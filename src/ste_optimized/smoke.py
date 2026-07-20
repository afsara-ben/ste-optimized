"""Bounded, real-contrast angry-transform smoke workflow.

This is deliberately narrower than the production trainer: it proves that one
real angry-minus-neutral contrast can be transformed, trained through the full
Qwen replay/codec/expert graph, and reused on neutral utterances that were not
used for the optimizer updates.

Every Qwen generation call is batched.  In particular, an optimizer update is
one native four-row generation (K=1, M=4), and each validation arm is one
native generation over the same four held-out bases.  There is no per-row Qwen
generation loop in this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
import torch
import yaml

from .backend import PromptEntry, QwenTTSBackend
from .config import ExperimentConfig
from .data import BaseRecord, PairRecord, load_bases, load_pairs
from .distributed import DistributedContext
from .experts import ExpertSuite, resample_to_expert
from .hooks import DecodeActivationCapture
from .training import Trainer
from .transform import save_transform


TRAIN_BASES = 4
VALIDATION_BASES = 4
MAX_SMOKE_UPDATES = 10
MAX_SPEAKER_DEGRADATION = 0.02
MIN_MEAN_SPEAKER_SIM = 0.85
MIN_ANGER_GAIN = 0.0


class SmokeFailure(RuntimeError):
    """The workflow ran, but did not establish the requested claim."""


def run_smoke(
    cfg: ExperimentConfig,
    pair_id: str = "0011:000254",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train and validate one real angry transform, then export its artifacts.

    Success means that the empirically selected ``checkpoint x alpha`` arm:

    * scores higher for angry than both the unsteered arm and every matched
      raw-v alpha arm;
    * satisfies EOS-termination and absolute/relative speaker constraints;
    * uses the same learned ``T(v)`` for all held-out validation utterances.

    The word "optimal" in the resulting payload is intentionally scoped to the
    checkpoints and alphas measured by this run; no finite smoke test can prove
    a universal mathematical optimum over every possible neutral utterance.
    """
    _validate_smoke_config(cfg)
    cuda = _require_cuda(cfg.model.device)

    started_at = datetime.now(timezone.utc)
    out = Path(output_dir) if output_dir is not None else (
        Path("runs") / f"angry-smoke-{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    out = out.resolve()
    if (out / "train" / "checkpoint.pt").exists():
        raise SmokeFailure(
            f"refusing to mix a smoke run with existing trainer state: {out}; "
            "choose a new --output directory"
        )
    out.mkdir(parents=True, exist_ok=True)
    cfg.train.output_dir = str(out / "train")
    resolved_config = out / "resolved_config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )

    run_t0 = time.perf_counter()
    backend = QwenTTSBackend(cfg.model)
    experts = ExpertSuite.load(cfg.model.device)

    pair = _find_pair(cfg, pair_id)
    contrast, contrast_source = _load_or_extract_selected_contrast(
        cfg, pair, out, backend, experts
    )
    vector = contrast["v"].detach().to(torch.float32).cpu()
    if vector.shape != (cfg.model.hidden_size,) or not torch.isfinite(vector).all():
        raise SmokeFailure(
            f"contrast {pair_id} has invalid vector shape/value: {tuple(vector.shape)}"
        )
    if float(vector.norm()) == 0.0:
        raise SmokeFailure(f"contrast {pair_id} is the zero vector")

    train_bases = _select_diverse_bases(
        load_bases(cfg.data.dataset_dir, "train"), pair.speaker, TRAIN_BASES
    )
    validation_bases = _select_diverse_bases(
        load_bases(cfg.data.dataset_dir, "validation"),
        pair.speaker,
        VALIDATION_BASES,
    )
    if {b.base_id for b in train_bases} & {b.base_id for b in validation_bases}:
        raise SmokeFailure("training and validation base IDs overlap")

    # Build validation prompts once.  Every arm below passes this full list to
    # one native batched generate call.
    validation_entries = backend.prepare_voice_clone_prompts(
        [_base_prompt_row(b) for b in validation_bases], cfg.sampling.language
    )
    eval_seed = _stable_seed(cfg.train.seed, pair_id, "heldout-validation")
    alphas = tuple(sorted(set(float(a) for a in cfg.eval.alphas)))
    audio_root = out / "audio"

    unsteered = _generate_score_arm(
        backend,
        experts,
        cfg,
        validation_bases,
        validation_entries,
        vector=None,
        alpha=0.0,
        seed=eval_seed,
        arm="unsteered",
        audio_root=audio_root,
    )
    raw_arms = []
    for alpha in alphas:
        raw_arms.append(
            _generate_score_arm(
                backend,
                experts,
                cfg,
                validation_bases,
                validation_entries,
                vector=vector,
                alpha=alpha,
                seed=eval_seed,
                arm=f"raw_alpha_{_alpha_slug(alpha)}",
                audio_root=audio_root,
            )
        )

    baseline_angry = max(
        [unsteered["mean_angry_prob"]]
        + [arm["mean_angry_prob"] for arm in raw_arms]
    )

    # Exactly one real contrast and exactly four fixed train bases.  The
    # BatchSampler expands this one contrast to all four rows (K=1, M=4), and
    # Trainer.run_update performs one native batched Qwen generation.
    trainer = Trainer(
        cfg,
        DistributedContext(mode="none"),
        backend=backend,
        experts=experts,
        contrasts=[contrast],
        bases=train_bases,
    )

    candidate_history: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_state: dict[str, torch.Tensor] | None = None
    stop_reason = "max_updates"
    train_t0 = time.perf_counter()
    attempts = 0

    while trainer.completed < cfg.train.max_updates:
        elapsed = time.perf_counter() - train_t0
        if cfg.train.max_attempts is not None and attempts >= cfg.train.max_attempts:
            stop_reason = "max_attempts"
            break
        if cfg.train.max_wall_seconds is not None and elapsed >= cfg.train.max_wall_seconds:
            stop_reason = "max_wall_seconds"
            break

        attempts += 1
        trainer.attempts = attempts
        record = trainer.run_update()
        training_records.append(_jsonable(record))
        if record.get("skipped"):
            continue

        checkpoint = _evaluate_checkpoint(
            backend=backend,
            experts=experts,
            cfg=cfg,
            transform=trainer.transform,
            raw_vector=vector,
            bases=validation_bases,
            entries=validation_entries,
            alphas=alphas,
            seed=eval_seed,
            update=trainer.completed,
            unsteered=unsteered,
            audio_root=audio_root,
        )
        candidate_history.append(checkpoint)
        checkpoint_best = checkpoint.get("best_feasible")
        if checkpoint_best is not None and (
            selected is None
            or checkpoint_best["mean_angry_prob"] > selected["mean_angry_prob"]
        ):
            selected = copy.deepcopy(checkpoint_best)
            selected["update"] = trainer.completed
            selected_state = {
                name: value.detach().cpu().clone()
                for name, value in trainer.transform.state_dict().items()
            }

        # Stop at the first checkpoint that proves the requested directional
        # effect.  This is the shortest honest smoke; a longer production run
        # can search a larger checkpoint/alpha space later.
        if selected is not None and selected["mean_angry_prob"] > (
            baseline_angry + MIN_ANGER_GAIN
        ):
            stop_reason = "success_gate"
            break

    training_seconds = time.perf_counter() - train_t0
    success = bool(
        selected is not None
        and selected["mean_angry_prob"] > baseline_angry + MIN_ANGER_GAIN
    )

    report: dict[str, Any] = {
        "schema": "ste-optimized/angry-smoke-report/v1",
        "status": "pass" if success else "fail",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair": {
            "pair_id": pair.pair_id,
            "speaker": pair.speaker,
            "split": pair.split,
            "text": pair.text,
            "neutral_audio": pair.neutral_audio,
            "emotional_audio": pair.emotional_audio,
        },
        "contrast": {
            "source": contrast_source,
            "artifact": str(out / "selected_contrast.pt"),
            "norm": float(vector.norm()),
            "emotion_prob": contrast.get("emotion_prob"),
            "speaker_sim": contrast.get("speaker_sim"),
        },
        "cuda": cuda,
        "model_load_seconds": backend.load_seconds,
        "train_base_ids": [b.base_id for b in train_bases],
        "validation_base_ids": [b.base_id for b in validation_bases],
        "batch_contract": {
            "contrasts_per_update": 1,
            "bases_per_contrast": 4,
            "native_qwen_calls_per_update": 1,
            "sequential_qwen_generation": False,
            "validation_rows_per_arm": len(validation_bases),
        },
        "controls": {"unsteered": unsteered, "raw_alpha_sweep": raw_arms},
        "control_max_mean_angry_prob": baseline_angry,
        "checkpoint_history": candidate_history,
        "selected": selected,
        "training": {
            "requested_max_updates": cfg.train.max_updates,
            "completed_updates": trainer.completed,
            "attempts": attempts,
            "stop_reason": stop_reason,
            "wall_seconds": training_seconds,
            "records": training_records,
        },
        "constraints": {
            "minimum_termination_rate": cfg.train.min_row_survival,
            "minimum_mean_speaker_similarity": MIN_MEAN_SPEAKER_SIM,
            "maximum_speaker_degradation_vs_unsteered": MAX_SPEAKER_DEGRADATION,
            "minimum_angry_gain_over_best_control": MIN_ANGER_GAIN,
        },
        "total_wall_seconds": time.perf_counter() - run_t0,
        "resolved_config": str(resolved_config),
        "claim_scope": (
            "empirical optimum over this run's measured checkpoints and alpha "
            "grid on four held-out validation bases"
        ),
    }

    # Always retain the last transform and the full measurements for debugging
    # a failed smoke.  The deployment artifacts are emitted only on success.
    save_transform(
        out / "last_transform.pt",
        trainer.transform,
        provenance=_provenance(cfg, pair, trainer.completed, "diagnostic-last"),
    )

    if success:
        assert selected is not None and selected_state is not None
        trainer.transform.load_state_dict(selected_state)
        learned_path = out / "learned_angry_transform.pt"
        save_transform(
            learned_path,
            trainer.transform,
            provenance=_provenance(
                cfg,
                pair,
                int(selected["update"]),
                "empirically-selected",
                selected,
            ),
        )
        with torch.no_grad():
            transformed = trainer.transform(vector.to(backend.device)).float().cpu()
        alpha = float(selected["alpha"])
        payload_path = out / "angry_steering_payload.pt"
        torch.save(
            {
                "schema": "ste-optimized/angry-steering/v1",
                "pair_id": pair.pair_id,
                "raw_v": vector,
                "transformed_v": transformed,
                # Apply this vector at alpha=1, or apply transformed_v using
                # selected_alpha in DecodeStepSteering.
                "scaled_vector": transformed * alpha,
                "selected_alpha": alpha,
                "layer": cfg.model.layer,
                "steer_frame0_predictor": cfg.model.steer_frame0_predictor,
                "model_id": cfg.model.model_id,
                "model_revision": cfg.model.model_revision,
                "hidden_size": cfg.model.hidden_size,
                "transform_path": str(learned_path),
                "selected_update": int(selected["update"]),
                "validation_base_ids": [b.base_id for b in validation_bases],
                "selection": selected,
                "inference_contract": {
                    "scaled_vector_alpha": 1.0,
                    "transformed_vector_alpha": alpha,
                    "reuse_same_vector_for_each_neutral_batch_row": True,
                },
            },
            payload_path,
        )
        report["artifacts"] = {
            "transform": str(learned_path),
            "steering_payload": str(payload_path),
            "audio_dir": str(audio_root),
        }
    else:
        report["failure_reason"] = (
            "no evaluated learned checkpoint/alpha achieved a higher mean "
            "angry probability than both unsteered and the best matched raw-v "
            "arm while satisfying termination and speaker constraints"
        )

    report_path = out / "report.json"
    report_path.write_text(
        json.dumps(_jsonable(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    report["report_path"] = str(report_path)

    if not success:
        raise SmokeFailure(f"angry-transform smoke failed; see {report_path}")
    return report


def _validate_smoke_config(cfg: ExperimentConfig) -> None:
    train = cfg.train
    if train.contrasts_per_update != 1 or train.bases_per_contrast != TRAIN_BASES:
        raise ValueError(
            "smoke requires train.contrasts_per_update=1 and "
            f"train.bases_per_contrast={TRAIN_BASES}"
        )
    if train.max_updates < 1 or train.max_updates > MAX_SMOKE_UPDATES:
        raise ValueError(
            f"smoke max_updates must be in [1, {MAX_SMOKE_UPDATES}], got "
            f"{train.max_updates}"
        )
    if train.max_attempts is None or train.max_attempts < train.max_updates:
        raise ValueError("smoke requires max_attempts >= max_updates")
    if train.max_wall_seconds is None or train.max_wall_seconds <= 0:
        raise ValueError("smoke requires a positive max_wall_seconds")
    if not cfg.eval.alphas or any(float(a) <= 0 for a in cfg.eval.alphas):
        raise ValueError("smoke eval.alphas must contain positive values")
    if cfg.distributed.mode != "none":
        raise ValueError("the one-device smoke requires distributed.mode=none")


def _require_cuda(device: str) -> dict[str, Any]:
    if not str(device).startswith("cuda"):
        raise SmokeFailure(f"smoke requires a CUDA device, configured {device!r}")
    if not torch.cuda.is_available():
        raise SmokeFailure(
            "PyTorch cannot access CUDA: torch.cuda.is_available() is false; "
            "check the NVIDIA driver, CUDA_VISIBLE_DEVICES, and CUDA PyTorch build"
        )
    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise SmokeFailure(
            f"configured {device!r}, but only {torch.cuda.device_count()} CUDA "
            "device(s) are visible"
        )
    props = torch.cuda.get_device_properties(index)
    return {
        "torch_cuda_version": torch.version.cuda,
        "device": str(dev),
        "device_name": props.name,
        "device_count": torch.cuda.device_count(),
        "total_memory_bytes": props.total_memory,
    }


def _find_pair(cfg: ExperimentConfig, pair_id: str) -> PairRecord:
    matches = [
        pair
        for split in ("train", "validation", "test", "reserve")
        for pair in load_pairs(cfg.data.dataset_dir, split)
        if pair.pair_id == pair_id
    ]
    if len(matches) != 1:
        raise SmokeFailure(
            f"expected exactly one real manifest pair {pair_id!r}, found "
            f"{len(matches)} in {cfg.data.dataset_dir}"
        )
    return matches[0]


def _load_or_extract_selected_contrast(
    cfg: ExperimentConfig,
    pair: PairRecord,
    out: Path,
    backend: QwenTTSBackend,
    experts: ExpertSuite,
) -> tuple[dict[str, Any], str]:
    selected_path = out / "selected_contrast.pt"
    candidates: list[Path] = []
    if cfg.data.contrasts_path:
        candidates.append(Path(cfg.data.contrasts_path))
    candidates.append(
        Path(cfg.data.dataset_dir)
        / f"contrasts-{cfg.data.emotion}-{pair.split}.pt"
    )
    for path in candidates:
        if not path.exists():
            continue
        row = _selected_row_from_artifact(path, pair.pair_id, cfg)
        if row is not None:
            _save_single_contrast(selected_path, row, cfg, pair, str(path))
            return row, f"existing:{path.resolve()}"

    row = _extract_exact_pair(cfg, pair, out, backend, experts)
    _save_single_contrast(selected_path, row, cfg, pair, "direct-extraction")
    return row, "direct-extraction"


def _selected_row_from_artifact(
    path: Path, pair_id: str, cfg: ExperimentConfig
) -> dict[str, Any] | None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "ste-optimized/contrasts/v1":
        raise SmokeFailure(f"unknown contrast schema in {path}")
    if payload.get("layer") != cfg.model.layer:
        raise SmokeFailure(
            f"contrast layer mismatch in {path}: {payload.get('layer')} != "
            f"{cfg.model.layer}"
        )
    if payload.get("model_id") not in (None, cfg.model.model_id):
        raise SmokeFailure(f"contrast model mismatch in {path}")
    if payload.get("model_revision") not in (None, cfg.model.model_revision):
        raise SmokeFailure(f"contrast model revision mismatch in {path}")
    return next((row for row in payload.get("rows", []) if row["pair_id"] == pair_id), None)


def _extract_exact_pair(
    cfg: ExperimentConfig,
    pair: PairRecord,
    out: Path,
    backend: QwenTTSBackend,
    experts: ExpertSuite,
) -> dict[str, Any]:
    rows = [
        {
            "base_id": f"smoke-extract:{pair.pair_id}:neutral",
            "target_text": pair.text,
            "reference_text": pair.text,
            "reference_audio": pair.neutral_audio,
        },
        {
            "base_id": f"smoke-extract:{pair.pair_id}:angry",
            "target_text": pair.text,
            "reference_text": pair.text,
            "reference_audio": pair.emotional_audio,
        },
    ]
    entries = backend.prepare_voice_clone_prompts(rows, cfg.sampling.language)
    capture = DecodeActivationCapture(
        backend.tts.model, cfg.model.layer, batch_size=2
    )
    generated = backend.generate_prepared_batch(
        entries,
        vectors=None,
        sampling=cfg.sampling,
        seed=_stable_seed(cfg.train.seed, pair.pair_id, "contrast-extraction"),
        capture_hook=capture,
    )
    if not all(generated.terminated):
        raise SmokeFailure(
            f"selected contrast pair {pair.pair_id} did not EOS-terminate: "
            f"lengths={generated.lengths}, cap={cfg.sampling.max_frames}"
        )
    means = capture.mean(generated.lengths)
    vector = (means[1] - means[0]).to(torch.float32).cpu()

    decoded = []
    waves16 = []
    extraction_audio = out / "audio" / "contrast_extraction"
    extraction_audio.mkdir(parents=True, exist_ok=True)
    for label, codes, entry in zip(("neutral", "angry"), generated.codes, entries):
        wav, sr = backend.decode_hard(codes, ref_codes=entry.ref_code)
        wav = wav.detach().float().cpu().reshape(-1)
        wav_path = extraction_audio / f"{label}.wav"
        sf.write(wav_path, wav.numpy(), sr)
        decoded.append(str(wav_path))
        waves16.append(resample_to_expert(wav.to(backend.device), sr))
    with torch.no_grad():
        _, angry = experts.emotion.loss([waves16[1]], cfg.data.emotion)
        _, speaker = experts.speaker.loss(
            [waves16[1]], [pair.emotional_audio]
        )
    return {
        "pair_id": pair.pair_id,
        "v": vector,
        "layer": cfg.model.layer,
        "emotion_prob": float(angry[0]),
        "speaker_sim": float(speaker[0]),
        "source_split": pair.split,
        "source_text": pair.text,
        "generated_audio": decoded,
        "extraction_lengths": generated.lengths,
    }


def _save_single_contrast(
    path: Path,
    row: dict[str, Any],
    cfg: ExperimentConfig,
    pair: PairRecord,
    source: str,
) -> None:
    torch.save(
        {
            "schema": "ste-optimized/contrasts/v1",
            "emotion": cfg.data.emotion,
            "layer": cfg.model.layer,
            "model_id": cfg.model.model_id,
            "model_revision": cfg.model.model_revision,
            "selected_pair_id": pair.pair_id,
            "selected_pair_split": pair.split,
            "source": source,
            "rows": [row],
        },
        path,
    )


def _select_diverse_bases(
    bases: Iterable[BaseRecord], contrast_speaker: str, count: int
) -> list[BaseRecord]:
    """Deterministically round-robin speakers instead of taking one block."""
    by_speaker: dict[str, list[BaseRecord]] = defaultdict(list)
    for base in sorted(bases, key=lambda b: (b.speaker, b.base_id)):
        if base.speaker != contrast_speaker:
            by_speaker[base.speaker].append(base)
    selected: list[BaseRecord] = []
    depth = 0
    while len(selected) < count and by_speaker:
        added = False
        for speaker in sorted(by_speaker):
            rows = by_speaker[speaker]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != count:
        raise SmokeFailure(
            f"only {len(selected)} cross-speaker neutral bases available; "
            f"the smoke requires exactly {count}"
        )
    return selected


def _base_prompt_row(base: BaseRecord) -> dict[str, str]:
    return {
        "base_id": base.base_id,
        "target_text": base.target_text,
        "reference_text": base.reference_text,
        "reference_audio": base.reference_audio,
        "reference_speaker": base.speaker,
    }


@torch.no_grad()
def _generate_score_arm(
    backend: QwenTTSBackend,
    experts: ExpertSuite,
    cfg: ExperimentConfig,
    bases: list[BaseRecord],
    entries: list[PromptEntry],
    vector: torch.Tensor | None,
    alpha: float,
    seed: int,
    arm: str,
    audio_root: Path,
) -> dict[str, Any]:
    if len(bases) < VALIDATION_BASES or len(entries) != len(bases):
        raise ValueError("every validation arm requires the complete held-out batch")
    vectors = None
    if vector is not None:
        one = vector.detach().to(backend.device, torch.float32).reshape(1, -1)
        vectors = one.expand(len(entries), -1)

    # This is the only Qwen call in an arm: all held-out neutral utterances are
    # generated together using the same raw or transformed vector.
    generated = backend.generate_prepared_batch(
        entries, vectors, cfg.sampling, seed=seed, alpha=alpha
    )
    arm_dir = audio_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    waves16: list[torch.Tensor] = []
    refs: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, (base, entry, codes) in enumerate(
        zip(bases, entries, generated.codes)
    ):
        wav, sr = backend.decode_hard(codes, ref_codes=entry.ref_code)
        wav = wav.detach().float().cpu().reshape(-1)
        wav_path = arm_dir / f"{index:02d}-{_safe_name(base.base_id)}.wav"
        sf.write(wav_path, wav.numpy(), sr)
        waves16.append(resample_to_expert(wav.to(backend.device), sr))
        refs.append(base.reference_audio)
        rows.append(
            {
                "base_id": base.base_id,
                "speaker": base.speaker,
                "target_text": base.target_text,
                "terminated": bool(generated.terminated[index]),
                "frames": int(generated.lengths[index]),
                "audio": str(wav_path),
            }
        )
    _, angry = experts.emotion.loss(waves16, cfg.data.emotion)
    _, speaker = experts.speaker.loss(waves16, refs)
    for row, prob, sim in zip(rows, angry.tolist(), speaker.tolist()):
        row["angry_prob"] = float(prob)
        row["speaker_sim"] = float(sim)
    return {
        "arm": arm,
        "alpha": float(alpha),
        "seed": int(seed),
        "rows": rows,
        "mean_angry_prob": _mean(float(p) for p in angry.tolist()),
        "mean_speaker_sim": _mean(float(s) for s in speaker.tolist()),
        "termination_rate": _mean(1.0 if ok else 0.0 for ok in generated.terminated),
        "generation_wall_seconds": generated.wall_seconds,
        "native_qwen_batch_size": len(entries),
    }


def _evaluate_checkpoint(
    *,
    backend: QwenTTSBackend,
    experts: ExpertSuite,
    cfg: ExperimentConfig,
    transform,
    raw_vector: torch.Tensor,
    bases: list[BaseRecord],
    entries: list[PromptEntry],
    alphas: tuple[float, ...],
    seed: int,
    update: int,
    unsteered: dict[str, Any],
    audio_root: Path,
) -> dict[str, Any]:
    with torch.no_grad():
        transformed = transform(raw_vector.to(backend.device)).float().cpu()
    arms: list[dict[str, Any]] = []
    for alpha in alphas:
        arm = _generate_score_arm(
            backend,
            experts,
            cfg,
            bases,
            entries,
            vector=transformed,
            alpha=alpha,
            seed=seed,
            arm=f"learned_update_{update:03d}_alpha_{_alpha_slug(alpha)}",
            audio_root=audio_root,
        )
        arm["constraints"] = _arm_constraints(arm, unsteered, cfg)
        arms.append(arm)
    feasible = [arm for arm in arms if arm["constraints"]["pass"]]
    best = max(feasible, key=lambda arm: arm["mean_angry_prob"], default=None)
    return {
        "update": update,
        "transform_delta_l2": float((transformed - raw_vector).norm()),
        "transformed_norm": float(transformed.norm()),
        "arms": arms,
        "best_feasible": copy.deepcopy(best),
    }


def _arm_constraints(
    arm: dict[str, Any], unsteered: dict[str, Any], cfg: ExperimentConfig
) -> dict[str, Any]:
    degradation = unsteered["mean_speaker_sim"] - arm["mean_speaker_sim"]
    gates = {
        "termination": arm["termination_rate"] >= cfg.train.min_row_survival,
        "speaker_absolute": arm["mean_speaker_sim"] >= MIN_MEAN_SPEAKER_SIM,
        "speaker_degradation": degradation <= MAX_SPEAKER_DEGRADATION,
    }
    return {
        **gates,
        "speaker_degradation_vs_unsteered": degradation,
        "pass": all(gates.values()),
    }


def _provenance(
    cfg: ExperimentConfig,
    pair: PairRecord,
    update: int,
    status: str,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "pair_id": pair.pair_id,
        "pair_split": pair.split,
        "completed_updates": update,
        "status": status,
        "selection": selection,
    }


def _stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256(":".join([str(base), *parts]).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _alpha_slug(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(sum(rows) / len(rows)) if rows else float("nan")


def _jsonable(value: Any) -> Any:
    """Strict-JSON conversion (notably turns non-finite floats into null)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
