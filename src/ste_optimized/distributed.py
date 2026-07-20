"""Multi-GPU policy (plan §2).

Recommended: **seed-parallel** — one independent process per GPU, different
`train.seed` and `train.output_dir`, zero coupling:

    CUDA_VISIBLE_DEVICES=0 ste-optimized train -c cfg.yaml --seed 42 --output runs/s42 &
    CUDA_VISIBLE_DEVICES=1 ste-optimized train -c cfg.yaml --seed 43 --output runs/s43 &

Optional: **ddp_rows** — shard each update's K x M rows across ranks under
torchrun and average transform gradients. Historically measured at only ~1.13x
for this workload (generation-bound, 32k trainable params); provided because
the option was requested, defaulted off.

    torchrun --standalone --nproc-per-node=2 -m ste_optimized train \
        -c cfg.yaml --distributed ddp_rows
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    mode: str = "none"          # none | seed_parallel | ddp_rows
    rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    # rows are sharded round-robin so every rank sees a balanced mix of
    # contrasts; gradients are averaged, so the effective update equals the
    # single-process one over all surviving rows.
    def shard_rows(self, rows: list) -> list:
        if self.mode != "ddp_rows" or self.world_size == 1:
            return rows
        return [r for i, r in enumerate(rows) if i % self.world_size == self.rank]

    def allreduce_grads(self, module: torch.nn.Module) -> None:
        if self.mode != "ddp_rows" or self.world_size == 1:
            return
        for p in module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= self.world_size

    def global_fraction(self, numerator: int, denominator: int) -> float:
        if self.mode != "ddp_rows" or self.world_size == 1:
            return numerator / max(denominator, 1)
        t = torch.tensor([numerator, denominator], dtype=torch.float64)
        if torch.cuda.is_available():
            t = t.cuda()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t[0] / t[1].clamp_min(1))


def init_distributed(mode: str, backend: str = "nccl") -> DistributedContext:
    if mode != "ddp_rows":
        return DistributedContext(mode=mode)
    if "RANK" not in os.environ:
        raise RuntimeError("ddp_rows requires launching under torchrun")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % max(torch.cuda.device_count(), 1))
    return DistributedContext(mode=mode, rank=rank, world_size=world)


def device_for_rank(ctx: DistributedContext, requested: str) -> str:
    if ctx.mode == "ddp_rows":
        return f"cuda:{ctx.rank % max(torch.cuda.device_count(), 1)}"
    return requested
