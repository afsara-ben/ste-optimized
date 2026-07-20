"""Validation: cadence panel during training + full gated panel (plan §6/§4.8).

All generation goes through the batched path. Controls (raw-v / unsteered arms)
depend only on frozen inputs and seeds — they are generated ONCE and cached by
a panel hash, then reused for every checkpoint.

Gates (plan Stage 4): emotion-prob delta > 0 with a 95% paired-bootstrap CI
excluding 0; mean speaker similarity >= 0.85 AND degradation <= 0.05; ISR
(intelligible-speech rate == EOS-terminated fraction here) >= 80% and >=
control - 5pp; Whisper WER degradation <= the configured absolute threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .backend import QwenTTSBackend
from .config import ExperimentConfig
from .data import load_bases
from .experts import ExpertSuite, resample_to_expert
from .extraction import load_contrasts
from .transform import LowRankTransform, load_transform
from .wer_metrics import matched_control_wer, word_error_counts
from .whisper_asr import WhisperASRExpert


_EVAL_SPLITS = frozenset({"validation", "test"})


def _validate_eval_split(split: str) -> str:
    if split not in _EVAL_SPLITS:
        raise ValueError(
            f"evaluation split must be one of {sorted(_EVAL_SPLITS)}, got {split!r}"
        )
    return split


def _contrast_path_for_split(path: str | Path, split: str) -> Path:
    """Replace the final standalone split token in the artifact filename."""

    split = _validate_eval_split(split)
    source = Path(path)
    matches = list(re.finditer(
        r"(?<![A-Za-z0-9])(train|validation|test)(?![A-Za-z0-9])",
        source.name,
    ))
    if not matches:
        # Historical validation-only configs were allowed to point directly at
        # an artifact whose name had no split token.  Preserve that behavior,
        # but never silently use it as a test artifact.
        if split == "validation":
            return source
        raise ValueError(
            "cannot resolve test contrasts from a contrasts_path without a "
            f"train/validation/test filename token: {source}"
        )
    match = matches[-1]
    resolved_name = (
        source.name[:match.start()] + split + source.name[match.end():]
    )
    return source.with_name(resolved_name)


def _panel_rows(cfg: ExperimentConfig, limit: int | None,
                split: str = "validation") -> list[dict]:
    split = _validate_eval_split(split)
    contrasts = load_contrasts(
        str(_contrast_path_for_split(cfg.data.contrasts_path, split))
    )
    bases = load_bases(cfg.data.dataset_dir, split)
    rows = []
    # Materialize the complete canonical mapping first.  A contrast therefore
    # keeps the same held-out base in cadence/screening and full evaluation.
    for i, c in enumerate(contrasts):
        speaker = c["pair_id"].split(":")[0]
        elig = [b for b in bases if b.speaker != speaker] or bases
        b = elig[i % len(elig)]
        rows.append({"contrast": c, "base": b})
    if limit is None:
        return rows
    return _speaker_interleaved(rows)[:limit]


def _speaker_interleaved(rows: list[dict]) -> list[dict]:
    """Round-robin rows by contrast speaker, preserving within-speaker order."""

    groups: dict[str, list[dict]] = {}
    for row in rows:
        speaker = row["contrast"]["pair_id"].split(":")[0]
        groups.setdefault(speaker, []).append(row)
    ordered = []
    speakers = sorted(groups)
    for offset in range(max((len(group) for group in groups.values()), default=0)):
        for speaker in speakers:
            group = groups[speaker]
            if offset < len(group):
                ordered.append(group[offset])
    return ordered


@torch.no_grad()
def _generate_and_score(backend: QwenTTSBackend, experts: ExpertSuite,
                        asr: WhisperASRExpert | None,
                        cfg: ExperimentConfig, rows: list[dict],
                        vectors: torch.Tensor | None, alpha: float,
                        seed: int) -> list[dict]:
    """Generate/score ordered rows in deterministic, memory-bounded chunks."""

    batch_rows = cfg.eval.generation_batch_rows
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or \
            batch_rows <= 0:
        raise ValueError("eval.generation_batch_rows must be a positive integer")
    if vectors is not None and (
        vectors.dim() < 2 or vectors.shape[0] != len(rows)
    ):
        raise ValueError(
            "evaluation vectors must have one leading row per panel item: "
            f"{tuple(vectors.shape)} vs {len(rows)} rows"
        )

    out = []
    for start in range(0, len(rows), batch_rows):
        stop = min(start + batch_rows, len(rows))
        chunk_vectors = None if vectors is None else vectors[start:stop]
        out.extend(_generate_and_score_batch(
            backend, experts, asr, cfg, rows[start:stop], chunk_vectors,
            alpha, seed=_evaluation_chunk_seed(seed, start),
        ))
    return out


def _evaluation_chunk_seed(seed: int, start: int) -> int:
    """Keep the historical seed for chunk zero and derive later seeds stably."""

    if start == 0:
        return int(seed)
    ident = f"ste-eval-chunk-v1:{int(seed)}:{start}".encode()
    # torch.manual_seed accepts signed 64-bit values.  Staying in the positive
    # range also keeps this portable to generators with stricter seed checks.
    return int.from_bytes(hashlib.sha256(ident).digest()[:8], "little") % (2**63)


@torch.no_grad()
def _generate_and_score_batch(
    backend: QwenTTSBackend,
    experts: ExpertSuite,
    asr: WhisperASRExpert | None,
    cfg: ExperimentConfig,
    rows: list[dict],
    vectors: torch.Tensor | None,
    alpha: float,
    seed: int,
) -> list[dict]:
    prompt_rows = [{"base_id": r["base"].base_id,
                    "target_text": r["base"].target_text,
                    "reference_text": r["base"].reference_text,
                    "reference_audio": r["base"].reference_audio}
                   for r in rows]
    entries = backend.prepare_voice_clone_prompts(prompt_rows)
    gen = backend.generate_prepared_batch(entries, vectors, cfg.sampling,
                                          seed=seed, alpha=alpha)
    out = []
    waves, refs, keep = [], [], []
    for i, r in enumerate(rows):
        rec = {"pair_id": r["contrast"]["pair_id"],
               "base_id": r["base"].base_id, "alpha": alpha,
               "target_text": r["base"].target_text,
               "terminated": gen.terminated[i], "frames": gen.lengths[i]}
        try:
            # reference-conditioned decode: match native inference output
            wav, sr = backend.decode_hard(gen.codes[i],
                                          ref_codes=entries[i].ref_code)
            waves.append(resample_to_expert(wav.to(backend.device), sr))
            refs.append(r["base"].reference_audio)
            keep.append(i)
        except Exception as exc:
            rec["decode_error"] = f"{type(exc).__name__}: {exc}"
            rec["transcript"] = None
        out.append(rec)
    if waves:
        _, prob = experts.emotion.loss(waves, cfg.data.emotion)
        _, sim = experts.speaker.loss(waves, refs)
        for j, i in enumerate(keep):
            out[i]["emotion_prob"] = float(prob[j])
            out[i]["speaker_sim"] = float(sim[j])
        if asr is not None and cfg.eval.compute_wer:
            transcripts = asr.transcribe(waves)
            for transcript, i in zip(transcripts, keep):
                out[i]["transcript"] = transcript
    return out


def cadence_metric(trainer) -> float:
    """Small fixed panel during training; metric = mean emotion-prob delta of
    T(v) rows over cached raw-v control rows (higher is better)."""
    cfg = trainer.cfg
    rows = _panel_rows(cfg, cfg.eval.cadence_rows)
    v = torch.stack([r["contrast"]["v"].to(torch.float32) for r in rows]
                    ).to(trainer.backend.device)
    control = _cached_controls(
        trainer.backend, trainer.experts, trainer.asr, cfg, rows, v
    )
    u = trainer.transform(v)
    treated = _generate_and_score(
        trainer.backend, trainer.experts, trainer.asr, cfg, rows,
        u, cfg.train.alpha, seed=cfg.train.seed
    )
    deltas = [t.get("emotion_prob", 0.0) - c.get("emotion_prob", 0.0)
              for t, c in zip(treated, control)
              if t["terminated"] and c["terminated"]]
    if not deltas:
        return -1.0
    if cfg.eval.compute_wer:
        comparison = _wer_comparison(control, treated, cfg, min_reference_words=1)
        if not comparison["pass"]:
            return -1.0
    return float(sum(deltas) / len(deltas))


def _cached_controls(backend, experts, asr, cfg, rows, vectors) -> list[dict]:
    ident = json.dumps([[r["contrast"]["pair_id"], r["base"].base_id]
                        for r in rows]) + cfg.fingerprint()
    key = hashlib.sha256(ident.encode()).hexdigest()[:16]
    cache = Path(cfg.eval.control_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"controls-{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    control = _generate_and_score(backend, experts, asr, cfg, rows, vectors,
                                  cfg.train.alpha, seed=cfg.train.seed)
    path.write_text(json.dumps(control))
    return control


def full_panel(cfg: ExperimentConfig, transform: LowRankTransform,
               out_path: str | Path) -> dict:
    """Full validation report: per-pair T(v) vs raw v at alpha=1, plus the
    exported T(v_global) / raw v_global over the alpha sweep, with gates."""
    backend = QwenTTSBackend(cfg.model)
    experts = ExpertSuite.load(cfg.model.device)
    asr = _load_asr(cfg) if cfg.eval.compute_wer else None
    transform = transform.to(backend.device)
    rows = _panel_rows(cfg, None)
    v = torch.stack([r["contrast"]["v"].to(torch.float32) for r in rows]
                    ).to(backend.device)

    control = _cached_controls(backend, experts, asr, cfg, rows, v)
    with torch.no_grad():
        u = transform(v)
    treated = _generate_and_score(backend, experts, asr, cfg, rows, u,
                                  cfg.train.alpha, seed=cfg.train.seed)

    v_global = v.mean(0, keepdim=True)
    sweeps = {}
    for alpha in cfg.eval.alphas:
        with torch.no_grad():
            ug = transform(v_global)
        sweeps[f"T(v_global)@{alpha}"] = _generate_and_score(
            backend, experts, asr, cfg, rows, ug.expand(len(rows), -1), alpha,
            seed=cfg.train.seed)
        sweeps[f"v_global@{alpha}"] = _generate_and_score(
            backend, experts, asr, cfg, rows, v_global.expand(len(rows), -1), alpha,
            seed=cfg.train.seed)

    report = {
        "per_pair": {"control": control, "treated": treated},
        "sweeps": sweeps,
        "gates": _gates(control, treated, cfg),
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report


@dataclass
class _TransformSnapshot:
    """One unique transform state and every run artifact that aliases it."""

    transform: LowRankTransform
    provenance: dict
    artifacts: list[str]
    completed_updates: int | None
    digest: str


def checkpoint_panel(cfg: ExperimentConfig, run_dir: str | Path,
                     out_path: str | Path | None = None,
                     rows_limit: int | None = None,
                     alphas: tuple[float, ...] | list[float] | None = None,
                     split: str = "validation") -> dict:
    """Evaluate every unique transform snapshot on one fixed held-out panel.

    Qwen, the emotion/speaker experts, and Whisper are each loaded exactly once.
    Raw-vector controls are generated once per alpha and reused for every
    checkpoint.  Identical periodic/best/final exports are evaluated once while
    all artifact aliases remain recorded in the report.
    """

    split = _validate_eval_split(split)
    run = Path(run_dir)
    snapshots = _load_run_snapshots(run)
    if rows_limit is not None and (
        isinstance(rows_limit, bool) or not isinstance(rows_limit, int)
        or rows_limit <= 0
    ):
        raise ValueError("checkpoint panel rows_limit must be a positive integer")
    evaluated_alphas = _validated_alphas(
        cfg.eval.alphas if not alphas else alphas
    )
    provisional_screen = rows_limit is not None
    wer_min_reference_words = (
        1 if provisional_screen else cfg.asr.min_validation_reference_words
    )

    backend = QwenTTSBackend(cfg.model)
    experts = ExpertSuite.load(cfg.model.device)
    asr = _load_asr(cfg) if cfg.eval.compute_wer else None

    rows = _panel_rows(cfg, rows_limit, split=split)
    if not rows:
        raise ValueError(f"{split} checkpoint panel is empty")
    v = torch.stack([
        row["contrast"]["v"].to(torch.float32) for row in rows
    ]).to(backend.device)
    panel_items = [
        {
            "pair_id": row["contrast"]["pair_id"],
            "base_id": row["base"].base_id,
            "target_text": row["base"].target_text,
        }
        for row in rows
    ]
    panel_hash = hashlib.sha256(
        json.dumps({"split": split, "items": panel_items}, sort_keys=True).encode()
    ).hexdigest()[:16]

    controls = {
        alpha: _generate_and_score(
            backend, experts, asr, cfg, rows, v, alpha,
            seed=cfg.train.seed,
        )
        for alpha in evaluated_alphas
    }
    candidates = []
    for snapshot in snapshots:
        transform = snapshot.transform.to(backend.device)
        with torch.no_grad():
            transformed = transform(v)
        for alpha in evaluated_alphas:
            treated = _generate_and_score(
                backend, experts, asr, cfg, rows, transformed, alpha,
                seed=cfg.train.seed,
            )
            row_diagnostics = _matched_row_diagnostics(
                controls[alpha], treated, cfg
            )
            candidates.append({
                "artifacts": snapshot.artifacts,
                "completed_updates": snapshot.completed_updates,
                "transform_digest": snapshot.digest,
                "alpha": alpha,
                "metrics": _gates(
                    controls[alpha], treated, cfg,
                    min_reference_words=wer_min_reference_words,
                ),
                "rows": row_diagnostics,
            })
        # The module is tiny, but moving it back makes the single-model-memory
        # contract explicit when many checkpoints are present.
        snapshot.transform.to("cpu")

    report = {
        "schema": "ste-optimized/checkpoint-panel/v1",
        "run_dir": str(run.resolve()),
        "panel": {
            "split": split,
            "mode": "provisional_screen" if provisional_screen else "full",
            "hash": panel_hash,
            "rows": len(rows),
            "requested_rows": rows_limit,
            "items": panel_items,
            "seed": cfg.train.seed,
            "alphas": evaluated_alphas,
            "generation_batch_rows": cfg.eval.generation_batch_rows,
            "isr_definition": "EOS-terminated fraction",
        },
        "snapshots": len(snapshots),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection": _select_checkpoint_candidate(candidates),
        "per_checkpoint_alpha_selection": _per_pair_alpha_selection(candidates),
        "wer_contract": {
            "enabled": cfg.eval.compute_wer,
            "model_id": cfg.asr.model_id if cfg.eval.compute_wer else None,
            "revision": cfg.asr.revision if cfg.eval.compute_wer else None,
            "max_absolute_degradation": (
                cfg.asr.max_wer_degradation
                if cfg.eval.compute_wer else None
            ),
            "min_reference_words": (
                wer_min_reference_words if cfg.eval.compute_wer else None
            ),
            "configured_full_panel_min_reference_words": (
                cfg.asr.min_validation_reference_words
                if cfg.eval.compute_wer else None
            ),
            "provisional_screen": provisional_screen,
            "comparison": "matched raw-vector control at the same alpha",
        },
    }
    destination = Path(out_path) if out_path is not None \
        else run / "checkpoint_panel.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    return report


def _load_run_snapshots(run_dir: Path) -> list[_TransformSnapshot]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"training run directory does not exist: {run_dir}")
    paths = sorted((run_dir / "checkpoints").glob("transform-*.pt"))
    paths.extend(
        path for path in (
            run_dir / "best_transform.pt",
            run_dir / "final_transform.pt",
        ) if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(
            f"no transform snapshots found under training run: {run_dir}"
        )

    by_digest: dict[str, _TransformSnapshot] = {}
    order: list[str] = []
    for path in paths:
        try:
            transform, provenance = load_transform(path)
        except Exception as exc:
            raise ValueError(f"cannot load transform snapshot {path}: {exc}") from exc
        digest = _transform_digest(transform)
        artifact = str(path.relative_to(run_dir))
        if digest in by_digest:
            by_digest[digest].artifacts.append(artifact)
            continue
        update = provenance.get("completed_updates")
        if update is None:
            match = re.fullmatch(r"transform-(\d+)\.pt", path.name)
            update = int(match.group(1)) if match else None
        if update is not None and (
            isinstance(update, bool) or not isinstance(update, int) or update < 0
        ):
            raise ValueError(
                f"invalid completed_updates in transform snapshot {path}: {update!r}"
            )
        by_digest[digest] = _TransformSnapshot(
            transform=transform,
            provenance=provenance,
            artifacts=[artifact],
            completed_updates=update,
            digest=digest,
        )
        order.append(digest)
    return [by_digest[digest] for digest in order]


def _transform_digest(transform: LowRankTransform) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(transform.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:16]


def _validated_alphas(values) -> list[float]:
    alphas = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("evaluation alphas must be finite positive numbers")
        alpha = float(value)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("evaluation alphas must be finite positive numbers")
        if alpha not in alphas:
            alphas.append(alpha)
    if not alphas:
        raise ValueError("at least one evaluation alpha is required")
    return alphas


def _select_checkpoint_candidate(candidates: list[dict]) -> dict:
    """Select the strongest gated candidate, retaining a diagnostic fallback."""

    def has_finite_score(candidate: dict) -> bool:
        value = candidate.get("metrics", {}).get("emotion_delta_mean")
        try:
            return value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    scored = [candidate for candidate in candidates if has_finite_score(candidate)]
    feasible = [candidate for candidate in scored
                if candidate["metrics"].get("pass") is True]

    def score(candidate: dict) -> tuple[float, float, float, float, float]:
        metrics = candidate["metrics"]
        wer_delta = metrics.get("wer", {}).get("delta", 0.0)
        update = candidate.get("completed_updates")
        return (
            float(metrics["emotion_delta_mean"]),
            float(metrics.get("emotion_delta_ci", [float("-inf")])[0]),
            float(metrics.get("speaker_sim", float("-inf"))),
            -float(wer_delta),
            float(update if update is not None else -1),
        )

    def compact(candidate: dict | None) -> dict | None:
        if candidate is None:
            return None
        return {
            "artifacts": candidate["artifacts"],
            "completed_updates": candidate["completed_updates"],
            "transform_digest": candidate["transform_digest"],
            "alpha": candidate["alpha"],
            "metrics": candidate["metrics"],
        }

    best_feasible = max(feasible, key=score) if feasible else None
    best_observed = max(scored, key=score) if scored else None
    return {
        "criterion": (
            "maximum mean emotion-probability delta among candidates passing "
            "emotion, speaker, ISR, and WER gates"
        ),
        "feasible": best_feasible is not None,
        "best": compact(best_feasible),
        "best_observed": compact(best_observed),
    }


def _matched_row_diagnostics(control: list[dict], treated: list[dict],
                             cfg: ExperimentConfig) -> list[dict]:
    """Build compact, fail-closed matched evidence for one candidate arm."""

    if len(control) != len(treated):
        raise ValueError(
            f"diagnostic arms must be matched: {len(control)} != {len(treated)}"
        )
    diagnostics = []
    for index, (control_row, treated_row) in enumerate(zip(control, treated)):
        control_key = (control_row.get("pair_id"), control_row.get("base_id"))
        treated_key = (treated_row.get("pair_id"), treated_row.get("base_id"))
        if control_key != treated_key or None in control_key:
            raise ValueError(
                f"diagnostic row {index} is not a matched pair/base: "
                f"{control_key!r} != {treated_key!r}"
            )
        target = treated_row.get("target_text", control_row.get("target_text"))
        if not isinstance(target, str):
            raise ValueError(f"diagnostic row {index} has no target text")
        control_target = control_row.get("target_text")
        if control_target is not None and control_target != target:
            raise ValueError(f"diagnostic row {index} target text is not matched")

        angry_control = _finite_or_none(control_row.get("emotion_prob"))
        angry_treated = _finite_or_none(treated_row.get("emotion_prob"))
        angry_delta = (
            angry_treated - angry_control
            if angry_control is not None and angry_treated is not None else None
        )
        speaker_control = _finite_or_none(control_row.get("speaker_sim"))
        speaker_treated = _finite_or_none(treated_row.get("speaker_sim"))
        speaker_degradation = (
            speaker_control - speaker_treated
            if speaker_control is not None and speaker_treated is not None else None
        )

        control_transcript = control_row.get("transcript")
        treated_transcript = treated_row.get("transcript")
        control_wer = _utterance_wer(target, control_transcript)
        treated_wer = _utterance_wer(target, treated_transcript)
        wer_delta = (
            treated_wer["wer"] - control_wer["wer"]
            if treated_wer["wer"] is not None
            and control_wer["wer"] is not None else None
        )
        wer_scored = (
            cfg.eval.compute_wer
            and "transcript" in control_row and "transcript" in treated_row
            and wer_delta is not None
        )
        constraints = {
            "terminated": bool(
                control_row.get("terminated") and treated_row.get("terminated")
            ),
            "anger_scored": angry_delta is not None,
            "speaker_scored": speaker_degradation is not None,
            "speaker_abs": (
                speaker_treated is not None and speaker_treated >= 0.85
            ),
            "speaker_degradation": (
                speaker_degradation is not None
                and speaker_degradation <= 0.05
            ),
            "wer_scored": wer_scored,
            "wer": (
                wer_scored
                and wer_delta <= cfg.asr.max_wer_degradation
            ),
        }
        constraints["pass"] = all(constraints.values())
        failures = [
            name for name, passed in constraints.items()
            if name != "pass" and not passed
        ]
        diagnostics.append({
            "pair_id": control_key[0],
            "base_id": control_key[1],
            "target_text": target,
            "terminated": {
                "control": bool(control_row.get("terminated")),
                "treated": bool(treated_row.get("terminated")),
            },
            "angry_prob": {
                "control": angry_control,
                "treated": angry_treated,
                "delta": angry_delta,
            },
            "speaker_sim": {
                "control": speaker_control,
                "treated": speaker_treated,
                "degradation": speaker_degradation,
            },
            "transcript": {
                "control": control_transcript,
                "treated": treated_transcript,
            },
            "wer": {
                "control": control_wer,
                "treated": treated_wer,
                "delta": wer_delta,
            },
            "constraints": constraints,
            "failed_constraints": failures,
        })
    return diagnostics


def _finite_or_none(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _utterance_wer(target: str, transcript: str | None) -> dict:
    counts = word_error_counts(target, transcript)
    wer = (
        counts.errors / counts.reference_words
        if counts.reference_words else None
    )
    return {
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "errors": counts.errors,
        "reference_words": counts.reference_words,
        "wer": wer,
    }


def _per_pair_alpha_selection(candidates: list[dict]) -> list[dict]:
    """Choose a constraint-feasible alpha independently for every pair/base."""

    checkpoints: dict[str, dict] = {}
    for candidate in candidates:
        digest = candidate["transform_digest"]
        checkpoint = checkpoints.setdefault(digest, {
            "artifacts": candidate["artifacts"],
            "completed_updates": candidate["completed_updates"],
            "transform_digest": digest,
            "pairs": {},
            "pair_order": [],
        })
        for row in candidate.get("rows", []):
            key = (row["pair_id"], row["base_id"])
            if key not in checkpoint["pairs"]:
                checkpoint["pairs"][key] = []
                checkpoint["pair_order"].append(key)
            checkpoint["pairs"][key].append({
                "alpha": float(candidate["alpha"]),
                "row": row,
            })

    summaries = []
    for checkpoint in checkpoints.values():
        selections = []
        uncovered = []
        not_improved = []
        for pair_id, base_id in checkpoint.pop("pair_order"):
            arms = checkpoint["pairs"][(pair_id, base_id)]
            feasible = [
                arm for arm in arms
                if arm["row"].get("constraints", {}).get("pass") is True
                and _finite_or_none(
                    arm["row"].get("angry_prob", {}).get("delta")
                ) is not None
            ]
            if not feasible:
                uncovered.append({"pair_id": pair_id, "base_id": base_id})
                continue
            selected = max(feasible, key=lambda arm: (
                float(arm["row"]["angry_prob"]["delta"]),
                -float(arm["alpha"]),
            ))
            row = selected["row"]
            selection = {
                "pair_id": pair_id,
                "base_id": base_id,
                "alpha": selected["alpha"],
                "feasible_alpha_count": len(feasible),
                "angry_delta": row["angry_prob"]["delta"],
                "treated_angry_prob": row["angry_prob"]["treated"],
                "treated_speaker_sim": row["speaker_sim"]["treated"],
                "speaker_degradation": row["speaker_sim"]["degradation"],
                "wer_delta": row["wer"]["delta"],
            }
            selections.append(selection)
            if float(selection["angry_delta"]) <= 0:
                not_improved.append({"pair_id": pair_id, "base_id": base_id})

        total = len(checkpoint["pairs"])
        covered = len(selections)
        improved = covered - len(not_improved)
        checkpoint.pop("pairs")
        summaries.append({
            **checkpoint,
            "total_pairs": total,
            "covered_count": covered,
            "coverage": covered / total if total else 0.0,
            "improved_count": improved,
            "improvement_rate": improved / total if total else 0.0,
            "all_pairs_covered": total > 0 and covered == total,
            "all_pairs_improved": total > 0 and improved == total,
            "uncovered_pairs": uncovered,
            "not_improved_pairs": not_improved,
            "pair_selections": selections,
        })
    return summaries


def _gates(control: list[dict], treated: list[dict], cfg: ExperimentConfig,
           n_boot: int = 2000, seed: int = 0,
           min_reference_words: int | None = None) -> dict:
    if len(control) != len(treated):
        raise ValueError(
            f"gate arms must be matched: {len(control)} != {len(treated)}"
        )
    pairs = [(t["emotion_prob"] - c["emotion_prob"],
              t["speaker_sim"], c["speaker_sim"],
              t["emotion_prob"], c["emotion_prob"])
             for t, c in zip(treated, control)
             if t.get("terminated") and c.get("terminated")
             and "emotion_prob" in t and "emotion_prob" in c
             and "speaker_sim" in t and "speaker_sim" in c]
    isr_t = sum(1 for t in treated if t.get("terminated")) / max(len(treated), 1)
    isr_c = sum(1 for c in control if c.get("terminated")) / max(len(control), 1)
    gates = {
        "scored_pairs": len(pairs),
        "isr": isr_t,
        "isr_control": isr_c,
        "isr_degradation": isr_c - isr_t,
        "gate_isr": isr_t >= 0.80 and isr_t >= isr_c - 0.05,
    }
    if pairs:
        deltas = torch.tensor([p[0] for p in pairs])
        g = torch.Generator().manual_seed(seed)
        boots = torch.stack([
            deltas[
                torch.randint(len(deltas), (len(deltas),), generator=g)
            ].mean()
            for _ in range(n_boot)
        ])
        ci_low = boots.quantile(0.025).item()
        ci_high = boots.quantile(0.975).item()
        sim_t = torch.tensor([p[1] for p in pairs]).mean().item()
        sim_c = torch.tensor([p[2] for p in pairs]).mean().item()
        emotion_t = torch.tensor([p[3] for p in pairs]).mean().item()
        emotion_c = torch.tensor([p[4] for p in pairs]).mean().item()
        gates.update({
            "emotion_prob": emotion_t,
            "emotion_prob_control": emotion_c,
            "emotion_delta_mean": deltas.mean().item(),
            "emotion_delta_ci": [ci_low, ci_high],
            # This is the most literal per-contrast "accuracy" available for
            # an all-angry panel: the fraction of matched rows whose learned
            # vector raises angry probability over its raw-vector control.
            "emotion_improved_pairs": int((deltas > 0).sum().item()),
            "emotion_improved_rate": float((deltas > 0).float().mean().item()),
            "gate_emotion": ci_low > 0,
            "speaker_sim": sim_t,
            "speaker_sim_control": sim_c,
            "speaker_degradation": sim_c - sim_t,
            "gate_speaker_abs": sim_t >= 0.85,
            "gate_speaker_degradation": (sim_c - sim_t) <= 0.05,
        })
    else:
        gates.update({
            "reason": "no jointly terminated emotion/speaker-scored pairs",
            "emotion_prob": None,
            "emotion_prob_control": None,
            "emotion_delta_mean": None,
            "emotion_delta_ci": [None, None],
            "emotion_improved_pairs": 0,
            "emotion_improved_rate": 0.0,
            "gate_emotion": False,
            "speaker_sim": None,
            "speaker_sim_control": None,
            "speaker_degradation": None,
            "gate_speaker_abs": False,
            "gate_speaker_degradation": False,
        })
    if cfg.eval.compute_wer:
        wer = _wer_comparison(
            control, treated, cfg,
            min_reference_words=min_reference_words,
        )
        gates["wer"] = wer
        gates["gate_wer"] = wer["pass"]
    gates["pass"] = all(v for k, v in gates.items() if k.startswith("gate_"))
    return gates


def _load_asr(cfg: ExperimentConfig) -> WhisperASRExpert:
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(cfg.asr.dtype)
    if dtype is None:
        raise ValueError(f"unsupported ASR dtype: {cfg.asr.dtype!r}")
    return WhisperASRExpert(
        device=cfg.model.device,
        model_id=cfg.asr.model_id,
        revision=cfg.asr.revision,
        dtype=dtype,
        language=cfg.asr.language,
        task=cfg.asr.task,
    )


def _wer_comparison(control: list[dict], treated: list[dict],
                    cfg: ExperimentConfig,
                    min_reference_words: int | None = None) -> dict:
    if len(control) != len(treated):
        raise ValueError("WER requires matched control and treated rows")
    references = [row["target_text"] for row in treated]
    comparison = matched_control_wer(
        references,
        [row.get("transcript") for row in treated],
        [row.get("transcript") for row in control],
        max_delta=cfg.asr.max_wer_degradation,
        min_reference_words=(
            cfg.asr.min_validation_reference_words
            if min_reference_words is None else min_reference_words
        ),
    )
    return comparison.as_dict()
