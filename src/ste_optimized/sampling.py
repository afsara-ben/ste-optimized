"""Balanced, resumable sampling of (contrast, base) rows for batched updates.

Per plan §3/§4:
- contrasts shuffle without replacement per epoch, per-speaker balanced;
- each contrast draws M cross-speaker bases (base speaker != contrast speaker),
  resampled every epoch (rotation), so coverage of the ~|bases| eligible bases
  accumulates across epochs;
- fully deterministic given (seed, epoch, update);
- state (epoch, cursor) lives in the training checkpoint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from .data import BaseRecord


@dataclass
class UpdateBatch:
    contrast_ids: list[str]
    vectors: torch.Tensor            # [K, hidden] fp32
    weights: list[float]             # fixed per-contrast weights w_j
    rows: list[dict]                 # len K*M: prompt rows for the backend
    row_contrast: list[int]          # row -> index into contrast_ids
    epoch: int
    update_in_epoch: int


def _rng(seed: int, *parts) -> torch.Generator:
    h = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    g = torch.Generator()
    g.manual_seed(int.from_bytes(h[:8], "little", signed=False) % (2**63 - 1))
    return g


class BatchSampler:
    def __init__(self, contrasts: list[dict], bases: list[BaseRecord],
                 contrasts_per_update: int, bases_per_contrast: int,
                 seed: int, min_emotion_prob: float, min_speaker_sim: float) -> None:
        self.all = contrasts
        self.pool = [c for c in contrasts
                     if c.get("emotion_prob", 1.0) >= min_emotion_prob
                     and c.get("speaker_sim", 1.0) >= min_speaker_sim]
        if not self.pool:
            raise ValueError("no contrasts pass the fixed-weight quality filter")
        self.excluded = len(contrasts) - len(self.pool)
        self.bases = bases
        self.K = contrasts_per_update
        self.M = bases_per_contrast
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self._order: list[int] = []
        self._reshuffle()

    def updates_per_epoch(self) -> int:
        return (len(self.pool) + self.K - 1) // self.K

    def _reshuffle(self) -> None:
        g = _rng(self.seed, "epoch", self.epoch)
        # per-speaker balance: interleave speaker queues in shuffled order
        by_speaker: dict[str, list[int]] = {}
        for i, c in enumerate(self.pool):
            by_speaker.setdefault(c["pair_id"].split(":")[0], []).append(i)
        queues = []
        for spk in sorted(by_speaker):
            idx = torch.tensor(by_speaker[spk])
            queues.append(idx[torch.randperm(len(idx), generator=g)].tolist())
        order: list[int] = []
        while any(queues):
            for q in queues:
                if q:
                    order.append(q.pop())
        self._order = order
        self.cursor = 0

    def _eligible_bases(self, contrast_speaker: str) -> list[BaseRecord]:
        elig = [b for b in self.bases if b.speaker != contrast_speaker]
        if len(elig) < self.M:
            raise ValueError(
                f"only {len(elig)} cross-speaker bases for speaker "
                f"{contrast_speaker}; need M={self.M}")
        return elig

    def next_batch(self) -> UpdateBatch:
        if self.cursor >= len(self._order):
            self.epoch += 1
            self._reshuffle()
        take = self._order[self.cursor:self.cursor + self.K]
        update_in_epoch = self.cursor // self.K
        self.cursor += len(take)

        contrast_ids, vectors, weights, rows, row_contrast = [], [], [], [], []
        for slot, idx in enumerate(take):
            c = self.pool[idx]
            speaker = c["pair_id"].split(":")[0]
            elig = self._eligible_bases(speaker)
            g = _rng(self.seed, "bases", self.epoch, c["pair_id"])
            perm = torch.randperm(len(elig), generator=g)[: self.M]
            contrast_ids.append(c["pair_id"])
            vectors.append(c["v"].to(torch.float32))
            weights.append(float(c.get("emotion_prob", 1.0)))
            for b_i in perm.tolist():
                b = elig[b_i]
                rows.append({"base_id": b.base_id, "target_text": b.target_text,
                             "reference_text": b.reference_text,
                             "reference_audio": b.reference_audio,
                             "reference_speaker": b.speaker})
                row_contrast.append(slot)
        return UpdateBatch(
            contrast_ids=contrast_ids, vectors=torch.stack(vectors),
            weights=weights, rows=rows, row_contrast=row_contrast,
            epoch=self.epoch, update_in_epoch=update_in_epoch)

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state["epoch"])
        self._reshuffle()
        self.cursor = int(state["cursor"])
