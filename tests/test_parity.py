"""GPU trust gates, ordered by the initial milestone (2026-07-20).

MILESTONE: one angry transform, ONE FIXED BATCH, and this acceptance test:
the complete reference-conditioned waveform loss produces a finite, nonzero
expert-only gradient in T_theta, changes its parameters, decreases on repeated
optimization, and leaves every frozen model without parameter gradients.
THEN multi-chunk accumulation and native/replay parity — before DDP,
bucketing, or other emotions.

Run order:
    1. test_acceptance_one_fixed_batch          (THE milestone gate)
    2. test_multichunk_accumulation
    3. test_prompt_assembly_matches_native_prefill
       test_soft_codec_matches_hard_decode
       test_emotion2vec_head_parity

Run with:  STE_OPT_REF_WAV=<speech.wav> pytest -m gpu tests/test_parity.py
Needs one CUDA device; models download on first use.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.gpu

REF_WAV = os.environ.get("STE_OPT_REF_WAV", "")
REF_TEXT = os.environ.get("STE_OPT_REF_TEXT", "Something about the house.")
ATTN_IMPLEMENTATION = os.environ.get("STE_OPT_ATTN_IMPLEMENTATION", "sdpa")


def _skip_reasons():
    if not torch.cuda.is_available():
        return "no CUDA device"
    if not REF_WAV or not os.path.exists(REF_WAV):
        return "set STE_OPT_REF_WAV to a speech wav"
    return None


@pytest.fixture(scope="module")
def backend():
    reason = _skip_reasons()
    if reason:
        pytest.skip(reason)
    from ste_optimized.backend import QwenTTSBackend
    from ste_optimized.config import ModelConfig
    return QwenTTSBackend(ModelConfig(attn_implementation=ATTN_IMPLEMENTATION))


@pytest.fixture(scope="module")
def entries(backend):
    rows = [
        {"base_id": f"parity:{i}", "target_text": t,
         "reference_text": REF_TEXT,
         "reference_audio": REF_WAV}
        for i, t in enumerate(["The quick brown fox.",
                               "A bird sang in the tree."])
    ]
    return backend.prepare_voice_clone_prompts(rows)


def _frozen_modules(backend, experts=None):
    st = getattr(backend.tts.model, "speech_tokenizer", None)
    mods = {"talker": backend.talker, "codec": getattr(st, "model", None)}
    if experts is not None:
        mods["emotion2vec"] = experts.emotion.model
        mods["wavlm"] = experts.speaker.model
    return mods


def _assert_frozen_grad_free(mods):
    for name, m in mods.items():
        if m is None:
            continue
        for p in m.parameters():
            assert not p.requires_grad, f"{name} has requires_grad=True"
            assert p.grad is None, f"{name} accumulated a gradient"


# --------------------------------------------------------------------------
# 1. THE MILESTONE ACCEPTANCE TEST — one fixed batch, complete loss path
# --------------------------------------------------------------------------
def test_acceptance_one_fixed_batch(backend, entries):
    """Complete reference-conditioned waveform loss on one fixed batch:
    (a) finite, nonzero expert-only gradient in T_theta;
    (b) an optimizer step changes T_theta's parameters;
    (c) the loss decreases under repeated optimization;
    (d) every frozen model (talker, codec, emotion2vec, WavLM) ends with no
        parameter gradients and requires_grad=False."""
    from ste_optimized.codec import decode_soft, output_sample_rate
    from ste_optimized.config import SamplingConfig
    from ste_optimized.experts import ExpertSuite, resample_to_expert
    from ste_optimized.replay import replay_chunk
    from ste_optimized.transform import LowRankTransform

    sampling = SamplingConfig(max_frames=64)
    experts = ExpertSuite.load(str(backend.device))
    sr = output_sample_rate(backend.tts.model.speech_tokenizer)
    t = LowRankTransform(backend.cfg.hidden_size, 16)
    t.initialize_identity(7)
    t.to(backend.device)
    opt = torch.optim.AdamW(t.parameters(), lr=2e-3)
    torch.manual_seed(0)
    v = (torch.randn(len(entries), backend.cfg.hidden_size,
                     device=backend.device) * 5)

    first_grad_checked = False
    losses = []
    for step in range(25):
        with torch.no_grad():
            u0 = t(v)
        gen = backend.generate_prepared_batch(entries, u0.detach(), sampling,
                                              seed=100)
        keep = [i for i, ok in enumerate(gen.terminated) if ok]
        if not keep:
            continue
        opt.zero_grad(set_to_none=True)
        u = t(v[torch.tensor(keep, device=v.device)])
        rep = replay_chunk(backend, [entries[i] for i in keep],
                           [gen.codes[i] for i in keep], u, sampling, 1.0)
        wavs = [decode_soft(backend.tts.model.speech_tokenizer.model,
                            oh.unsqueeze(0), ref_codes=entries[i].ref_code)[0]
                for oh, i in zip(rep.ste_onehots, keep)]
        wavs16 = [resample_to_expert(w, sr) for w in wavs]
        e_loss, _ = experts.emotion.loss(wavs16, "angry")
        s_loss, _ = experts.speaker.loss(
            wavs16, [entries[i].reference_audio for i in keep])
        loss = (e_loss + s_loss).mean()
        loss.backward()

        if not first_grad_checked:
            # (a) finite, nonzero expert-only gradient in T_theta
            for p in t.parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
            assert t.up.grad.abs().sum() > 0
            # (d) frozen models untouched
            _assert_frozen_grad_free(_frozen_modules(backend, experts))
            before = {n: p.detach().clone() for n, p in t.named_parameters()}
            opt.step()
            # (b) parameters actually change
            changed = any(not torch.equal(before[n], p.detach())
                          for n, p in t.named_parameters())
            assert changed, "optimizer step did not change T_theta"
            first_grad_checked = True
        else:
            opt.step()
        losses.append(float(loss))

    assert first_grad_checked, "no surviving generation in 25 attempts"
    assert len(losses) >= 8, "too many generation failures"
    # (c) decreases under repeated optimization (noisy: compare tails)
    assert min(losses[-4:]) < losses[0], f"loss did not decrease: {losses}"
    # (d) final check after all steps
    _assert_frozen_grad_free(_frozen_modules(backend, experts))


# --------------------------------------------------------------------------
# 2. Multi-chunk accumulation (run AFTER acceptance passes)
# --------------------------------------------------------------------------
def test_multichunk_accumulation(backend, entries):
    """Two chunks (chunk size 1) with per-chunk T(v) recompute: no freed-graph
    error, gradients accumulate, frozen models stay grad-free."""
    from ste_optimized.codec import decode_soft
    from ste_optimized.config import SamplingConfig
    from ste_optimized.replay import replay_chunk
    from ste_optimized.transform import LowRankTransform

    sampling = SamplingConfig(max_frames=48)
    t = LowRankTransform(backend.cfg.hidden_size, 16)
    t.initialize_identity(7)
    t.to(backend.device)
    torch.manual_seed(1)
    v = torch.randn(len(entries), backend.cfg.hidden_size, device=backend.device)
    with torch.no_grad():
        u0 = t(v)
    gen = backend.generate_prepared_batch(entries, u0.detach(), sampling, seed=7)
    keep = [i for i, ok in enumerate(gen.terminated) if ok]
    if len(keep) < 2:
        pytest.skip("need two surviving rows")
    for i in keep[:2]:                       # chunk size 1 -> two backwards
        u = t(v[i: i + 1])                   # fresh forward per chunk
        rep = replay_chunk(backend, [entries[i]], [gen.codes[i]], u,
                           sampling, 1.0)
        wav = decode_soft(backend.tts.model.speech_tokenizer.model,
                          rep.ste_onehots[0].unsqueeze(0),
                          ref_codes=entries[i].ref_code)[0]
        wav.pow(2).mean().backward()         # second backward must NOT raise
    for p in t.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    _assert_frozen_grad_free(_frozen_modules(backend))


# --------------------------------------------------------------------------
# 3. Native/replay + component parity
# --------------------------------------------------------------------------
def test_prompt_assembly_matches_native_prefill(backend, entries):
    """Our build_prompt_embed vs the prefill embeds native generate() builds,
    captured with a pre-hook on the talker backbone."""
    entry = entries[0]
    captured = {}

    def pre_hook(module, args, kwargs):
        ie = kwargs.get("inputs_embeds")
        if ie is not None and ie.shape[1] > 1 and "prefill" not in captured:
            captured["prefill"] = ie.detach()

    handle = backend.talker.model.register_forward_pre_hook(
        pre_hook, with_kwargs=True)
    try:
        from ste_optimized.config import SamplingConfig
        backend.generate_prepared_batch(
            [entry], vectors=None,
            sampling=SamplingConfig(max_frames=4), seed=0)
    finally:
        handle.remove()
    assert "prefill" in captured, "did not capture a native prefill"
    native = captured["prefill"]
    ours = entry.prompt_embed.to(native.device, native.dtype)
    assert native.shape == ours.shape, (native.shape, ours.shape)
    assert torch.allclose(native, ours, atol=5e-3), \
        "prompt assembly drifted from the installed qwen-tts version"


def test_soft_codec_matches_hard_decode(backend, entries):
    """decode_soft with exact one-hots must equal the frozen hard decode —
    both bare and with the reference-conditioned prefix+trim path."""
    from ste_optimized.codec import decode_soft
    st = backend.tts.model.speech_tokenizer
    V = st.model.decoder.quantizer.rvq_first.vq.layers[0]._codebook.codebook_size
    codes = torch.randint(0, V, (60, 16), device=backend.device)

    hard, _ = backend.decode_hard(codes.cpu())
    onehots = torch.nn.functional.one_hot(codes, V).to(torch.float32).unsqueeze(0)
    with torch.no_grad():
        soft = decode_soft(st.model, onehots)[0].cpu()
    n = min(hard.shape[-1], soft.shape[-1])
    assert torch.allclose(hard[:n].float(), soft[:n].float(), atol=2e-2), \
        "soft quantizer path drifted from the frozen decoder"

    ref = entries[0].ref_code.clamp_max(V - 1)
    hard_r, _ = backend.decode_hard(codes.cpu(), ref_codes=ref.cpu())
    with torch.no_grad():
        soft_r = decode_soft(st.model, onehots, ref_codes=ref)[0].cpu()
    n = min(hard_r.shape[-1], soft_r.shape[-1])
    assert torch.allclose(hard_r[:n].float(), soft_r[:n].float(), atol=2e-2), \
        "reference-conditioned soft decode drifted from the hard path"


def test_emotion2vec_head_parity(backend):
    import librosa
    from ste_optimized.experts import EXPERT_SAMPLE_RATE, Emotion2VecExpert
    wav, _ = librosa.load(REF_WAV, sr=EXPERT_SAMPLE_RATE, mono=True)
    expert = Emotion2VecExpert(device=str(backend.device))
    assert expert.parity_check(torch.from_numpy(wav).to(backend.device)), \
        "differentiable emotion head diverges from funasr inference"
