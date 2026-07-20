"""Validation: cadence panel during training + full gated panel (plan §6/§4.8).

All generation goes through the batched path. Controls (raw-v / unsteered arms)
depend only on frozen inputs and seeds — they are generated ONCE and cached by
a panel hash, then reused for every checkpoint.

Gates (plan Stage 4): emotion-prob delta > 0 with a 95% paired-bootstrap CI
excluding 0; mean speaker similarity >= 0.85 AND degradation <= 0.02; ISR
(intelligible-speech rate == EOS-terminated fraction here; plug an ASR for the
stricter definition) >= 80% and >= control - 5pp; optional WER gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .backend import QwenTTSBackend
from .config import ExperimentConfig
from .data import load_bases
from .experts import ExpertSuite, resample_to_expert
from .extraction import load_contrasts
from .transform import LowRankTransform


def _panel_rows(cfg: ExperimentConfig, limit: int | None) -> list[dict]:
    contrasts = load_contrasts(cfg.data.contrasts_path.replace(
        "train", "validation")) if "train" in cfg.data.contrasts_path \
        else load_contrasts(cfg.data.contrasts_path)
    bases = load_bases(cfg.data.dataset_dir, "validation")
    rows = []
    for i, c in enumerate(contrasts if limit is None else contrasts[:limit]):
        speaker = c["pair_id"].split(":")[0]
        elig = [b for b in bases if b.speaker != speaker] or bases
        b = elig[i % len(elig)]
        rows.append({"contrast": c, "base": b})
    return rows


@torch.no_grad()
def _generate_and_score(backend: QwenTTSBackend, experts: ExpertSuite,
                        cfg: ExperimentConfig, rows: list[dict],
                        vectors: torch.Tensor | None, alpha: float,
                        seed: int) -> list[dict]:
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
               "terminated": gen.terminated[i], "frames": gen.lengths[i]}
        if gen.terminated[i]:
            # reference-conditioned decode: match native inference output
            wav, sr = backend.decode_hard(gen.codes[i],
                                          ref_codes=entries[i].ref_code)
            waves.append(resample_to_expert(wav.to(backend.device), sr))
            refs.append(r["base"].reference_audio)
            keep.append(i)
        out.append(rec)
    if waves:
        _, prob = experts.emotion.loss(waves, cfg.data.emotion)
        _, sim = experts.speaker.loss(waves, refs)
        for j, i in enumerate(keep):
            out[i]["emotion_prob"] = float(prob[j])
            out[i]["speaker_sim"] = float(sim[j])
    return out


def cadence_metric(trainer) -> float:
    """Small fixed panel during training; metric = mean emotion-prob delta of
    T(v) rows over cached raw-v control rows (higher is better)."""
    cfg = trainer.cfg
    rows = _panel_rows(cfg, cfg.eval.cadence_rows)
    v = torch.stack([r["contrast"]["v"].to(torch.float32) for r in rows]
                    ).to(trainer.backend.device)
    control = _cached_controls(trainer.backend, trainer.experts, cfg, rows, v)
    u = trainer.transform(v)
    treated = _generate_and_score(trainer.backend, trainer.experts, cfg, rows,
                                  u, cfg.train.alpha, seed=cfg.train.seed)
    deltas = [t.get("emotion_prob", 0.0) - c.get("emotion_prob", 0.0)
              for t, c in zip(treated, control)
              if t["terminated"] and c["terminated"]]
    return float(sum(deltas) / len(deltas)) if deltas else -1.0


def _cached_controls(backend, experts, cfg, rows, vectors) -> list[dict]:
    ident = json.dumps([[r["contrast"]["pair_id"], r["base"].base_id]
                        for r in rows]) + cfg.fingerprint()
    key = hashlib.sha256(ident.encode()).hexdigest()[:16]
    cache = Path(cfg.eval.control_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"controls-{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    control = _generate_and_score(backend, experts, cfg, rows, vectors,
                                  cfg.train.alpha, seed=cfg.train.seed)
    path.write_text(json.dumps(control))
    return control


def full_panel(cfg: ExperimentConfig, transform: LowRankTransform,
               out_path: str | Path) -> dict:
    """Full validation report: per-pair T(v) vs raw v at alpha=1, plus the
    exported T(v_global) / raw v_global over the alpha sweep, with gates."""
    backend = QwenTTSBackend(cfg.model)
    experts = ExpertSuite.load(cfg.model.device)
    transform = transform.to(backend.device)
    rows = _panel_rows(cfg, None)
    v = torch.stack([r["contrast"]["v"].to(torch.float32) for r in rows]
                    ).to(backend.device)

    control = _cached_controls(backend, experts, cfg, rows, v)
    with torch.no_grad():
        u = transform(v)
    treated = _generate_and_score(backend, experts, cfg, rows, u,
                                  cfg.train.alpha, seed=cfg.train.seed)

    v_global = v.mean(0, keepdim=True)
    sweeps = {}
    for alpha in cfg.eval.alphas:
        with torch.no_grad():
            ug = transform(v_global)
        sweeps[f"T(v_global)@{alpha}"] = _generate_and_score(
            backend, experts, cfg, rows, ug.expand(len(rows), -1), alpha,
            seed=cfg.train.seed)
        sweeps[f"v_global@{alpha}"] = _generate_and_score(
            backend, experts, cfg, rows, v_global.expand(len(rows), -1), alpha,
            seed=cfg.train.seed)

    report = {
        "per_pair": {"control": control, "treated": treated},
        "sweeps": sweeps,
        "gates": _gates(control, treated),
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report


def _gates(control: list[dict], treated: list[dict], n_boot: int = 2000,
           seed: int = 0) -> dict:
    pairs = [(t["emotion_prob"] - c["emotion_prob"],
              t["speaker_sim"], c["speaker_sim"])
             for t, c in zip(treated, control)
             if t.get("terminated") and c.get("terminated")
             and "emotion_prob" in t and "emotion_prob" in c]
    if not pairs:
        return {"pass": False, "reason": "no scored pairs"}
    deltas = torch.tensor([p[0] for p in pairs])
    g = torch.Generator().manual_seed(seed)
    boots = torch.stack([
        deltas[torch.randint(len(deltas), (len(deltas),), generator=g)].mean()
        for _ in range(n_boot)])
    ci_low, ci_high = boots.quantile(0.025).item(), boots.quantile(0.975).item()
    sim_t = torch.tensor([p[1] for p in pairs]).mean().item()
    sim_c = torch.tensor([p[2] for p in pairs]).mean().item()
    isr_t = sum(1 for t in treated if t.get("terminated")) / max(len(treated), 1)
    isr_c = sum(1 for c in control if c.get("terminated")) / max(len(control), 1)
    gates = {
        "emotion_delta_mean": deltas.mean().item(),
        "emotion_delta_ci": [ci_low, ci_high],
        "gate_emotion": ci_low > 0,
        "speaker_sim": sim_t, "gate_speaker_abs": sim_t >= 0.85,
        "gate_speaker_degradation": (sim_c - sim_t) <= 0.02,
        "isr": isr_t, "gate_isr": isr_t >= 0.80 and isr_t >= isr_c - 0.05,
    }
    gates["pass"] = all(v for k, v in gates.items() if k.startswith("gate_"))
    return gates
