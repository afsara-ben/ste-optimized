from __future__ import annotations

import math

import pytest
import torch
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)

from ste_optimized.whisper_asr import (
    WHISPER_LARGE_V3_TURBO_ID,
    WHISPER_LARGE_V3_TURBO_REVISION,
    DifferentiableWhisperFeatureExtractor,
    WhisperASRExpert,
)


def _short_feature_extractor() -> WhisperFeatureExtractor:
    # A one-second frontend keeps the parity and gradient tests fast.  Its
    # formulas are the same as large-v3-turbo's 30-second, 128-mel frontend.
    return WhisperFeatureExtractor(
        feature_size=8,
        sampling_rate=16_000,
        hop_length=160,
        chunk_length=1,
        n_fft=400,
        dither=0.0,
    )


class _FakeWhisperTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.prefix_tokens = [1, 3, 4, 5]
        self.prefix_call = None

    def set_prefix_tokens(self, *, language, task, predict_timestamps):
        self.prefix_call = (language, task, predict_timestamps)

    def __call__(
        self,
        texts,
        *,
        add_special_tokens,
        padding,
        truncation,
        return_attention_mask,
        return_tensors,
    ):
        assert add_special_tokens
        assert padding
        assert truncation is False
        assert return_attention_mask
        assert return_tensors == "pt"
        rows = []
        for text in texts:
            lexical = [6 + (ord(char) % 17) for char in text if char != " "]
            rows.append(self.prefix_tokens + lexical + [self.eos_token_id])
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, values in enumerate(rows):
            ids[row, : len(values)] = torch.tensor(values)
            mask[row, : len(values)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def batch_decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return ["decoded" for _ in token_ids]


def _tiny_expert(*, include_control_tokens=False, include_eos=True):
    feature_extractor = _short_feature_extractor()
    config = WhisperConfig(
        vocab_size=32,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=32,
        max_source_positions=50,
        max_target_positions=64,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=2,
        decoder_start_token_id=1,
        suppress_tokens=None,
        begin_suppress_tokens=None,
    )
    model = WhisperForConditionalGeneration(config)
    tokenizer = _FakeWhisperTokenizer()
    expert = WhisperASRExpert(
        device="cpu",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        include_control_tokens=include_control_tokens,
        include_eos=include_eos,
    )
    return expert, tokenizer


def test_large_v3_turbo_defaults_are_pinned():
    assert WHISPER_LARGE_V3_TURBO_ID == "openai/whisper-large-v3-turbo"
    assert WHISPER_LARGE_V3_TURBO_REVISION == (
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    )


def test_torch_frontend_matches_hugging_face_and_keeps_gradient():
    torch.manual_seed(7)
    hf = _short_feature_extractor()
    waves = [torch.randn(3_211), torch.randn(8_003)]
    expected = hf(
        [wave.numpy() for wave in waves],
        sampling_rate=16_000,
        return_tensors="pt",
    ).input_features

    first = waves[0].clone().requires_grad_()
    second = waves[1].clone().requires_grad_()
    ours = DifferentiableWhisperFeatureExtractor(hf)([first, second])

    torch.testing.assert_close(ours, expected, atol=1e-5, rtol=1e-5)
    ours.square().mean().backward()
    for wave in (first, second):
        assert wave.grad is not None
        assert torch.isfinite(wave.grad).all()
        assert wave.grad.abs().sum() > 0


def test_teacher_forced_nll_reaches_waveform_but_not_frozen_model():
    torch.manual_seed(11)
    expert, tokenizer = _tiny_expert()
    waves = [
        (torch.randn(2_900) * 0.03).requires_grad_(),
        (torch.randn(4_100) * 0.03).requires_grad_(),
    ]

    row_nll = expert.loss(
        waves, ["angry words", "still clear"], use_checkpoint=True
    )

    assert row_nll.shape == (2,)
    assert torch.isfinite(row_nll).all()
    assert tokenizer.prefix_call == ("english", "transcribe", False)
    row_nll.mean().backward()

    for wave in waves:
        assert wave.grad is not None
        assert torch.isfinite(wave.grad).all()
        assert wave.grad.abs().sum() > 0
    for parameter in expert.model.parameters():
        assert not parameter.requires_grad
        assert parameter.grad is None
    assert not expert.model.training


def test_target_mask_excludes_controls_without_breaking_decoder_labels():
    expert, _ = _tiny_expert(include_control_tokens=False, include_eos=False)
    targets = expert.tokenize_targets(["a", "abcd"])

    # After dropping SOT, labels retain EN/TRANSCRIBE/NO_TIMESTAMPS so the
    # decoder remains correctly teacher-forced, but those controls are not
    # scored.  EOS is also retained in labels and independently masked here.
    assert targets.labels[0, :3].tolist() == [3, 4, 5]
    assert not targets.loss_mask[:, :3].any()
    assert targets.loss_mask.sum(dim=1).tolist() == [1, 4]


def test_nll_is_meaned_per_row_not_across_the_padded_batch():
    expert, _ = _tiny_expert()
    # Uniform logits make every selected token cost log(vocab_size).  The two
    # rows therefore have identical NLL despite different transcript lengths.
    with torch.no_grad():
        expert.model.proj_out.weight.zero_()
    row_nll = expert.loss(
        [torch.zeros(2_000), torch.zeros(3_000)],
        ["a", "a much longer row"],
        use_checkpoint=False,
    )
    torch.testing.assert_close(
        row_nll,
        torch.full((2,), math.log(expert.model.config.vocab_size)),
    )


def test_frontend_rejects_audio_longer_than_whisper_window():
    frontend = DifferentiableWhisperFeatureExtractor(_short_feature_extractor())
    too_long = torch.zeros(16_001)
    with pytest.raises(ValueError, match="longer than Whisper"):
        frontend([too_long])
