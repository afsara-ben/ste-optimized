"""Differentiable codec decode from STE one-hots.

The frozen 12 Hz tokenizer decoder (qwen_tts Qwen3TTSTokenizerV2Decoder) is:

    hidden = quantizer.decode(codes)   # embedding lookups, per codebook
    hidden = pre_conv -> pre_transformer -> upsample blocks -> decoder blocks

Only the very first step (hard `F.embedding` lookups inside
SplitResidualVectorQuantizer -> ... -> EuclideanCodebook.decode) is
non-differentiable w.r.t. codes. We replace exactly that step with soft
matrix products against the same codebook tables (`embedding_sum /
cluster_usage`), then run the ORIGINAL frozen modules for everything after —
so the hard path and the soft path agree bitwise when the one-hots are exact
one-hots (gpu parity test).

Structure (verified in modeling_qwen3_tts_tokenizer_v2.py):
    decoder.quantizer.rvq_first : ResidualVectorQuantizer over codebook 0
    decoder.quantizer.rvq_rest  : ResidualVectorQuantizer over codebooks 1..15
    each VectorQuantization: _codebook (EuclideanCodebook) + project_out
    each ResidualVectorQuantizer: sum of layer decodes -> output_proj (Conv1d)
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

CHUNK_FRAMES = 300
LEFT_CONTEXT = 25


def _reference_one_hot(
    ref_codes: torch.Tensor, device: torch.device, vocab_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode a constant reference prefix without mutating inference tensors."""
    import torch.nn.functional as F

    return F.one_hot(
        ref_codes.to(device).long().clamp(0, vocab_size - 1), vocab_size
    ).to(dtype)


def _codebook_table(vq_layer) -> torch.Tensor:
    cb = vq_layer._codebook
    return cb.embedding_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]


def _soft_rvq_decode(rvq, onehots: torch.Tensor) -> torch.Tensor:
    """rvq: ResidualVectorQuantizer; onehots [B, K, T, V] for its K layers.
    Mirrors rvq.decode(codes) with F.embedding replaced by onehots @ table.
    `onehots` must already be in the decoder's parameter dtype (see
    decode_soft) — feeding fp32 into the bf16 output_proj Conv1d crashes."""
    total = None
    for k, layer in enumerate(rvq.vq.layers):
        table = _codebook_table(layer).to(onehots.dtype)          # [V, D]
        quant = onehots[:, k] @ table                              # [B, T, D]
        quant = layer.project_out(quant)                           # [B, T, D']
        quant = quant.transpose(1, 2)                              # [B, D', T]
        total = quant if total is None else total + quant
    return rvq.output_proj(total)


def soft_quantizer_decode(decoder, onehots: torch.Tensor) -> torch.Tensor:
    """onehots [B, 16, T, V] -> quantized hidden [B, D, T], differentiable."""
    q = decoder.quantizer
    first = _soft_rvq_decode(q.rvq_first, onehots[:, : q.n_q_semantic])
    rest = _soft_rvq_decode(q.rvq_rest, onehots[:, q.n_q_semantic:])
    return first + rest


def _post_quantizer(decoder, hidden: torch.Tensor) -> torch.Tensor:
    """Everything decoder.forward does after quantizer.decode (verbatim)."""
    hidden = decoder.pre_conv(hidden).transpose(1, 2)
    hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
    hidden = hidden.permute(0, 2, 1)
    for blocks in decoder.upsample:
        for block in blocks:
            hidden = block(hidden)
    wav = hidden
    for block in decoder.decoder:
        wav = block(wav)
    return wav.clamp(min=-1, max=1)


def decode_soft(
    tokenizer_model, onehots: torch.Tensor, ref_codes: torch.Tensor | None = None,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """STE one-hots [B, T, 16, V] -> waveforms [B, samples] (differentiable).

    `ref_codes` [T_ref, 16] (long, constant — no gradient): the voice-clone
    reference codes. Native inference decodes cat([ref_code, generated]) and
    trims the reference span proportionally (verified,
    qwen_tts/inference/qwen3_tts_model.py:614-629); we mirror that so the
    experts score the SAME waveform normal Qwen output produces. Without the
    prefix, the decoder's in-chunk transformer sees different context and the
    audio differs materially.

    One-hots are cast to the decoder's parameter dtype up front (bf16 model +
    fp32 STE probs would crash at the quantizer's output_proj Conv1d).

    Chunked exactly like the frozen chunked_decode (chunk 300, left context 25)
    so long sequences stay memory-bounded; each chunk is activation-
    checkpointed (recomputed in backward).
    """
    import torch.nn.functional as F

    decoder = tokenizer_model.decoder
    dtype = next(decoder.parameters()).dtype
    x = onehots.permute(0, 2, 1, 3).to(dtype)  # [B, 16, T, V]
    ref_frames = 0
    if ref_codes is not None:
        V = x.shape[-1]
        ref_frames = int(ref_codes.shape[0])
        # ``ref_codes`` can originate from Qwen's inference-mode prompt
        # construction.  PyTorch forbids in-place updates to such tensors once
        # we are back in the grad-enabled replay pass, even after a dtype/device
        # conversion that happens to be a no-op.  Keep this out of place so the
        # constant reference prefix is safe in both inference and autograd.
        ref_oh = _reference_one_hot(ref_codes, x.device, V, dtype)
        ref_x = ref_oh.permute(1, 0, 2).unsqueeze(0)      # [1, 16, T_ref, V]
        x = torch.cat([ref_x.expand(x.shape[0], -1, -1, -1), x], dim=2)
    total_upsample = int(decoder.total_upsample)

    def run(chunk):
        hidden = soft_quantizer_decode(decoder, chunk)
        return _post_quantizer(decoder, hidden)

    wavs = []
    start = 0
    T = x.shape[2]
    while start < T:
        end = min(start + CHUNK_FRAMES, T)
        ctx = LEFT_CONTEXT if start - LEFT_CONTEXT > 0 else start
        chunk = x[:, :, start - ctx:end]
        if use_checkpoint and torch.is_grad_enabled():
            wav = checkpoint(run, chunk, use_reentrant=False)
        else:
            wav = run(chunk)
        wavs.append(wav[..., ctx * total_upsample:])
        start = end
    out = torch.cat(wavs, dim=-1).squeeze(1)
    if ref_frames:
        # mirror the native proportional trim exactly
        cut = int(ref_frames / max(T, 1) * out.shape[-1])
        out = out[..., cut:]
    return out


def output_sample_rate(speech_tokenizer) -> int:
    return int(speech_tokenizer.get_output_sample_rate())
