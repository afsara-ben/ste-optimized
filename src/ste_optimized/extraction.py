"""Contrast extraction — batched (does NOT assume any prior extraction).

Definition (mean-decode contrast, layer L):
    for one pair (speaker, index, text) with a neutral and an emotional
    recording of the SAME text:
      h_emo = mean over decode steps of the layer-L hidden state while
              voice-cloning the EMOTIONAL recording (reference audio = the
              emotional recording, reference text = the pair text) speaking
              the pair text;
      h_neu = same with the NEUTRAL recording as reference;
      v     = h_emo - h_neu           (float32, [hidden])

Both renditions must EOS-terminate below the frame cap; failed pairs are
recorded and excluded. Each pair's generated emotional clone is expert-scored
once (emotion probability, speaker similarity vs the reference) — these become
the FIXED loss weights w_j (constants; experts never receive gradients here).

Batched: pairs are processed 2 rows at a time per pair (neutral+emotional
renditions interleaved) in generation batches of `batch_pairs` pairs, using the
same batched-generation path as training (~25x over scalar extraction).
Resumable: one JSONL event per pair; finished pairs are skipped on rerun.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .backend import QwenTTSBackend
from .config import ExperimentConfig
from .data import PairRecord, load_pairs
from .experts import EXPERT_SAMPLE_RATE, ExpertSuite, resample_to_expert
from .hooks import DecodeActivationCapture


def extract_contrasts(
    cfg: ExperimentConfig, split: str, out_path: str | Path,
    batch_pairs: int = 8, backend: QwenTTSBackend | None = None,
    experts: ExpertSuite | None = None,
) -> Path:
    backend = backend or QwenTTSBackend(cfg.model)
    experts = experts or ExpertSuite.load(cfg.model.device)
    pairs = load_pairs(cfg.data.dataset_dir, split)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events_path = out_path.with_suffix(".events.jsonl")
    done = _finished_pair_ids(events_path)
    todo = [p for p in pairs if p.pair_id not in done]

    for start in range(0, len(todo), batch_pairs):
        chunk = todo[start:start + batch_pairs]
        rows, owners = [], []
        for p in chunk:
            for rendition, audio in (("neutral", p.neutral_audio),
                                     ("emotional", p.emotional_audio)):
                rows.append({"base_id": f"extract:{p.pair_id}:{rendition}",
                             "target_text": p.text, "reference_text": p.text,
                             "reference_audio": audio})
                owners.append((p, rendition))
        entries = backend.prepare_voice_clone_prompts(rows)
        capture = DecodeActivationCapture(
            backend.tts.model, cfg.model.layer, batch_size=len(entries))
        t0 = time.perf_counter()
        batch = backend.generate_prepared_batch(
            entries, vectors=None, sampling=cfg.sampling,
            seed=_pair_seed(cfg.train.seed, chunk[0].pair_id),
            capture_hook=capture)
        means = capture.mean(batch.lengths)  # [2*P, hidden] fp32 cpu

        with events_path.open("a") as fh:
            for i in range(0, len(owners), 2):
                pair = owners[i][0]
                ok = batch.terminated[i] and batch.terminated[i + 1]
                event = {"pair_id": pair.pair_id, "speaker": pair.speaker,
                         "split": split, "status": "ok" if ok else "max_frames",
                         "tokens": [batch.lengths[i], batch.lengths[i + 1]],
                         "wall_seconds": round(time.perf_counter() - t0, 3)}
                if ok:
                    v = (means[i + 1] - means[i]).to(torch.float32)
                    scores = _score_pair(backend, experts, cfg,
                                         batch.codes[i + 1],
                                         pair.emotional_audio,
                                         entries[i + 1].ref_code)
                    event.update(scores)
                    torch.save({"pair_id": pair.pair_id, "v": v,
                                "layer": cfg.model.layer, **scores},
                               _row_path(out_path, pair.pair_id))
                fh.write(json.dumps(event) + "\n")
    _consolidate(out_path, events_path, cfg)
    return out_path


@torch.no_grad()
def _score_pair(backend, experts, cfg, emo_codes, reference_audio,
                ref_codes) -> dict:
    # reference-conditioned decode: match native inference output exactly
    wav, sr = backend.decode_hard(emo_codes, ref_codes=ref_codes)
    wav16 = resample_to_expert(wav.to(backend.device), sr)
    _, prob = experts.emotion.loss([wav16], cfg.data.emotion)
    _, sim = experts.speaker.loss([wav16], [reference_audio])
    return {"emotion_prob": round(float(prob[0]), 4),
            "speaker_sim": round(float(sim[0]), 4)}


def _consolidate(out_path: Path, events_path: Path, cfg: ExperimentConfig) -> None:
    rows = []
    for line in events_path.read_text().splitlines():
        ev = json.loads(line)
        if ev["status"] != "ok":
            continue
        payload = torch.load(_row_path(out_path, ev["pair_id"]),
                             map_location="cpu", weights_only=True)
        rows.append(payload)
    torch.save({"schema": "ste-optimized/contrasts/v1",
                "emotion": cfg.data.emotion, "layer": cfg.model.layer,
                "model_id": cfg.model.model_id,
                "model_revision": cfg.model.model_revision,
                "rows": rows}, out_path)


def load_contrasts(path: str | Path) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "ste-optimized/contrasts/v1":
        raise ValueError(f"unknown contrast schema in {path}")
    return payload["rows"]


def _row_path(out_path: Path, pair_id: str) -> Path:
    d = out_path.parent / (out_path.stem + ".rows")
    d.mkdir(exist_ok=True)
    return d / (pair_id.replace(":", "_") + ".pt")


def _finished_pair_ids(events_path: Path) -> set[str]:
    if not events_path.exists():
        return set()
    return {json.loads(l)["pair_id"] for l in events_path.read_text().splitlines() if l}


def _pair_seed(base_seed: int, pair_id: str) -> int:
    import hashlib
    h = hashlib.sha256(f"{base_seed}:{pair_id}".encode()).digest()
    return int.from_bytes(h[:4], "little")
