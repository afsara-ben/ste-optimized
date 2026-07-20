import torch

from ste_optimized.transform import LowRankTransform, load_transform, save_transform


def test_identity_initialization_is_exact_identity():
    t = LowRankTransform(hidden_size=64, rank=4)
    t.initialize_identity(seed=7)
    v = torch.randn(5, 64)
    assert torch.allclose(t(v), v)
    assert torch.allclose(t(torch.zeros(64)), torch.zeros(64))  # T(0) = 0


def test_identity_init_is_not_a_saddle():
    """D must be nonzero so dL/dU != 0 at initialization."""
    t = LowRankTransform(hidden_size=64, rank=4)
    t.initialize_identity(seed=7)
    assert t.down.abs().sum() > 0
    v = torch.randn(3, 64)
    loss = t(v).pow(2).sum()
    loss.backward()
    assert t.up.grad is not None and t.up.grad.abs().sum() > 0


def test_gradients_flow_and_shapes():
    t = LowRankTransform(hidden_size=32, rank=2)
    t.initialize_identity(seed=1)
    v = torch.randn(4, 32)
    out = t(v)
    assert out.shape == (4, 32)
    out.sum().backward()
    assert t.down.grad is not None and t.up.grad is not None


def test_deterministic_initialization():
    a = LowRankTransform(64, 4); a.initialize_identity(3)
    b = LowRankTransform(64, 4); b.initialize_identity(3)
    assert torch.equal(a.down, b.down)


def test_save_load_roundtrip(tmp_path):
    t = LowRankTransform(32, 2)
    t.initialize_identity(seed=5)
    with torch.no_grad():
        t.up.add_(0.01)
    path = tmp_path / "t.pt"
    save_transform(path, t, provenance={"note": "test"})
    loaded, prov = load_transform(path)
    v = torch.randn(2, 32)
    assert torch.allclose(loaded(v), t(v))
    assert prov["note"] == "test"


def test_regularization_zero_at_identity():
    t = LowRankTransform(32, 2)
    t.initialize_identity(seed=5)
    v = torch.randn(3, 32)
    reg = t.regularization(v, 1.0, 1.0, 1.0)
    assert reg.item() < 1e-5  # fp32 cosine of a vector with itself ~1e-7 off
