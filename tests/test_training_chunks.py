"""CPU regression for the multi-chunk gradient-accumulation pattern.

Bug it pins (found 2026-07-20): computing u = T(v) ONCE and slicing it per
chunk means the first chunk's backward frees the shared transform graph, and
the second chunk's backward crashes ("backward through the graph a second
time"). The fix recomputes T(v[chunk]) fresh inside every chunk; gradients
still accumulate to the exact same totals as one big backward.
"""

import pytest
import torch

from ste_optimized.transform import LowRankTransform


def _loss(u: torch.Tensor) -> torch.Tensor:
    # stand-in for replay+codec+experts: any differentiable row-wise loss
    return (u.sin() * u).sum()


def test_shared_graph_double_backward_raises():
    """The OLD pattern must fail — documents the failure mode."""
    t = LowRankTransform(16, 2)
    t.initialize_identity(3)
    v = torch.randn(4, 16)
    u_all = t(v)                       # one shared forward
    chunks = [[0, 1], [2, 3]]
    _loss(u_all[chunks[0]]).backward() # frees the shared transform graph
    with pytest.raises(RuntimeError, match="econd time|freed"):
        _loss(u_all[chunks[1]]).backward()


def test_per_chunk_recompute_matches_single_backward():
    """The FIXED pattern: fresh T(v[chunk]) per chunk; accumulated grads equal
    one single-pass backward over all rows."""
    v = torch.randn(4, 16)
    chunks = [[0, 1], [2, 3]]

    ref = LowRankTransform(16, 2); ref.initialize_identity(3)
    with torch.no_grad():
        ref.up.add_(0.05)
    _loss(ref(v)).backward()

    chunked = LowRankTransform(16, 2); chunked.initialize_identity(3)
    with torch.no_grad():
        chunked.up.add_(0.05)
    for idx in chunks:
        u = chunked(v[torch.tensor(idx)])   # fresh graph per chunk
        _loss(u).backward()                 # accumulates into .grad

    assert torch.allclose(ref.down.grad, chunked.down.grad, atol=1e-6)
    assert torch.allclose(ref.up.grad, chunked.up.grad, atol=1e-6)


def test_three_chunks_uneven_sizes():
    v = torch.randn(5, 16)
    t = LowRankTransform(16, 2); t.initialize_identity(1)
    with torch.no_grad():
        t.up.add_(0.02)
    for idx in ([0, 1], [2], [3, 4]):
        _loss(t(v[torch.tensor(idx)])).backward()
    assert t.up.grad is not None and torch.isfinite(t.up.grad).all()
    assert t.up.grad.abs().sum() > 0
