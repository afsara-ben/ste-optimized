"""CPU tests for the steering hooks using a fake talker layer."""

from types import SimpleNamespace

import torch
from torch import nn

from ste_optimized.hooks import (DecodeActivationCapture, DecodeStepSteering,
                                 MaskedReplaySteering)


class _Layer(nn.Module):
    def forward(self, hidden):
        return (hidden,)  # decoder layers return tuples


def _fake_tts(n_layers: int = 1):
    layers = nn.ModuleList([_Layer() for _ in range(n_layers)])
    talker = SimpleNamespace(model=SimpleNamespace(layers=layers))
    return SimpleNamespace(talker=talker), layers


def test_decode_step_steering_skips_prefill_and_renormalizes():
    tts, layers = _fake_tts()
    vectors = torch.randn(3, 8)
    hook = DecodeStepSteering(tts, 0, vectors, alpha=1.0)
    prefill = torch.randn(3, 5, 8)
    decode = torch.randn(3, 1, 8)
    with hook:
        out_prefill = layers[0](prefill)[0]
        out_decode = layers[0](decode)[0]
    assert torch.equal(out_prefill, prefill)          # prefill untouched
    assert hook.calls == 1
    assert not torch.equal(out_decode, decode)        # decode steered
    # per-position L2 norm restored
    assert torch.allclose(out_decode.norm(dim=-1), decode.norm(dim=-1),
                          rtol=1e-4, atol=1e-5)
    # shift direction contains the steering vector component
    delta = (out_decode - decode)[:, 0, :]
    assert (delta * vectors).sum(-1).abs().sum() > 0


def test_masked_replay_steering_touches_only_masked_positions():
    tts, layers = _fake_tts()
    B, S, H = 2, 6, 8
    vectors = torch.randn(B, H)
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[0, 2:5] = True
    mask[1, 4:6] = True
    hook = MaskedReplaySteering(tts, 0, vectors, mask, alpha=1.0)
    hidden = torch.randn(B, S, H)
    with hook:
        out = layers[0](hidden)[0]
    assert hook.calls == int(mask.sum())
    assert torch.equal(out[~mask], hidden[~mask])     # unmasked untouched
    assert not torch.equal(out[mask], hidden[mask])
    assert torch.allclose(out[mask].norm(dim=-1), hidden[mask].norm(dim=-1),
                          rtol=1e-4, atol=1e-5)


def test_steer_last_prefill_flag_steers_frame0_predictor_only():
    """steer_frame0_predictor: the LAST prefill position is steered (frame-0
    predictor), the rest of the prompt stays untouched; decode steps behave
    as before."""
    tts, layers = _fake_tts()
    vectors = torch.randn(2, 8)
    hook = DecodeStepSteering(tts, 0, vectors, alpha=1.0,
                              steer_last_prefill=True)
    prefill = torch.randn(2, 5, 8)
    with hook:
        out = layers[0](prefill)[0]
    assert hook.calls == 1
    assert torch.equal(out[:, :-1], prefill[:, :-1])       # prompt untouched
    assert not torch.equal(out[:, -1], prefill[:, -1])     # frame-0 predictor steered
    assert torch.allclose(out[:, -1].norm(dim=-1), prefill[:, -1].norm(dim=-1),
                          rtol=1e-4, atol=1e-5)            # renormalised
    # decode step still steered
    decode = torch.randn(2, 1, 8)
    with hook:
        out_d = layers[0](decode)[0]
    assert not torch.equal(out_d, decode)


def test_masked_steering_batch_mismatch_raises():
    tts, layers = _fake_tts()
    hook = MaskedReplaySteering(tts, 0, torch.randn(2, 8),
                                torch.ones(2, 4, dtype=torch.bool), alpha=1.0)
    try:
        with hook:
            layers[0](torch.randn(2, 5, 8))
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_capture_mean_uses_per_row_lengths():
    tts, layers = _fake_tts()
    cap = DecodeActivationCapture(tts, 0, batch_size=2)
    steps = [torch.randn(2, 1, 4) for _ in range(5)]
    with cap:
        layers[0](torch.randn(2, 7, 4))               # prefill ignored
        for s in steps:
            layers[0](s)
    means = cap.mean([3, 5])
    expected0 = torch.stack([s[0, 0] for s in steps[:3]]).mean(0)
    expected1 = torch.stack([s[1, 0] for s in steps[:5]]).mean(0)
    assert torch.allclose(means[0], expected0, atol=1e-6)
    assert torch.allclose(means[1], expected1, atol=1e-6)
