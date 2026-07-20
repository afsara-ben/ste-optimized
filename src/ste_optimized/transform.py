"""Rank-16 low-rank steering transform (the only trainable module).

T(v) = v + U @ (D @ v)

Identity initialization: D = deterministic orthonormal random rows, U = 0.
NEVER zero-initialize both factors — dL/dU ∝ Dv = 0 and dL/dD ∝ Uᵀ(…) = 0 is an
exact saddle and training never starts (plan §1).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LowRankTransform(nn.Module):
    def __init__(self, hidden_size: int = 1024, rank: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        # Kept in fp32 regardless of model dtype (plan: fp32 trainable params).
        self.down = nn.Parameter(torch.zeros(rank, hidden_size, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(hidden_size, rank, dtype=torch.float32))

    @torch.no_grad()
    def initialize_identity(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        raw = torch.randn(self.hidden_size, self.rank, generator=gen)
        q, _ = torch.linalg.qr(raw)  # orthonormal columns [hidden, rank]
        self.down.copy_(q.T.to(torch.float32))
        self.up.zero_()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        squeeze = v.dim() == 1
        if squeeze:
            v = v.unsqueeze(0)
        if v.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got {v.shape[-1]}")
        x = v.to(torch.float32)
        out = x + (x @ self.down.T) @ self.up.T
        out = out.to(v.dtype) if v.dtype != torch.float32 else out
        return out.squeeze(0) if squeeze else out

    def regularization(
        self, v: torch.Tensor, w_identity: float, w_cosine: float, w_norm: float
    ) -> torch.Tensor:
        """Small penalties keeping T(v) near v in direction and norm (plan §3)."""
        x = v.to(torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        u = x + (x @ self.down.T) @ self.up.T
        identity = (u - x).pow(2).sum(-1).mean() / self.hidden_size
        cosine = (1 - torch.nn.functional.cosine_similarity(u, x, dim=-1)).mean()
        norm_ratio = (u.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).log().pow(2).mean()
        return w_identity * identity + w_cosine * cosine + w_norm * norm_ratio

    def metadata(self) -> dict[str, Any]:
        return {"hidden_size": self.hidden_size, "rank": self.rank,
                "parameters": self.hidden_size * self.rank * 2}


def save_transform(path, transform: LowRankTransform, provenance: dict[str, Any]) -> None:
    torch.save(
        {"state_dict": transform.state_dict(), "metadata": transform.metadata(),
         "provenance": provenance, "schema": "ste-optimized/transform/v1"},
        path,
    )


def load_transform(path) -> tuple[LowRankTransform, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "ste-optimized/transform/v1":
        raise ValueError(f"unknown transform schema in {path}")
    meta = payload["metadata"]
    transform = LowRankTransform(meta["hidden_size"], meta["rank"])
    transform.load_state_dict(payload["state_dict"])
    return transform, payload.get("provenance", {})
