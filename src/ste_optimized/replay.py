"""Pass-2: batched, padded teacher-forced replay with fixed-hard STE.

Sequence layout per row r (plan §3 step 6; positions after LEFT padding):

    [pad ... pad | prompt_0 .. prompt_{P-1} | e(y_0) .. e(y_{T-2})]

- logits at the last prompt position predict y_0; frame-input position t
  predicts y_{t+1}.
- steering applies ONLY to the frame-input positions e(y_0)..e(y_{T-2}); the
  last prompt position (frame-0 predictor) stays unsteered, matching pass 1
  where the prefill is unsteered.
- LEFT padding matches the native batched generate() convention and keeps each
  row's frames contiguous at the right end.

Outputs per row: STE one-hots [T, 16, V_codec] feeding the differentiable
codec, built from talker (codebook 0) and subtalker (codebooks 1-15) replay
probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .backend import PromptEntry, QwenTTSBackend
from .config import SamplingConfig
from .hooks import MaskedReplaySteering
from .ste import LogitProcessingChain, apply_warpers, one_hot_ste


@dataclass
class ReplayOutput:
    ste_onehots: list[torch.Tensor]   # per row [T_r, 16, V_codec], graph-attached
    codes: list[torch.Tensor]         # per row [T_r, 16] (device)
    logp_cb0_mean: torch.Tensor       # [R] diagnostic per-row mean log p(sampled)


def _codec_vocab_size(backend: QwenTTSBackend) -> int:
    tc = backend.model_config.talker_config
    # residual codebook size from the code predictor's config; the talker
    # vocab additionally holds text/control tokens beyond the codec range.
    return backend.talker.code_predictor.config.vocab_size \
        if hasattr(backend.talker.code_predictor.config, "vocab_size") \
        else tc.vocab_size


def replay_chunk(
    backend: QwenTTSBackend,
    entries: list[PromptEntry],
    codes: list[torch.Tensor],
    vectors: torch.Tensor,
    sampling: SamplingConfig,
    alpha: float,
) -> ReplayOutput:
    """One steered teacher-forced forward over a chunk of rows.

    entries[i], codes[i] ([T_i,16], cpu) and vectors[i] describe row i.
    `vectors` carries gradient (it is T(v) — NOT detached here).
    """
    device = backend.device
    tc = backend.model_config.talker_config
    R = len(entries)
    lengths = [int(c.shape[0]) for c in codes]

    rows, prompt_lens = [], []
    with torch.no_grad():
        for e, c in zip(entries, codes):
            prompt = e.prompt_embed.to(device)              # [1, P, H]
            frames = backend.frame_embeds(c.to(device))     # [T, H]
            rows.append(torch.cat([prompt[0], frames[:-1]], dim=0) if c.shape[0] > 1
                        else prompt[0])
            prompt_lens.append(prompt.shape[1])

    L = max(r.shape[0] for r in rows)
    H = rows[0].shape[1]
    inputs = torch.zeros(R, L, H, device=device, dtype=rows[0].dtype)
    attn = torch.zeros(R, L, dtype=torch.long, device=device)
    steer_mask = torch.zeros(R, L, dtype=torch.bool, device=device)
    target_pos = []  # per row: positions whose logits predict y_0..y_{T-1}
    steer_frame0 = backend.cfg.steer_frame0_predictor
    for i, r in enumerate(rows):
        pad = L - r.shape[0]
        inputs[i, pad:] = r
        attn[i, pad:] = 1
        P, T = prompt_lens[i], lengths[i]
        # steer the frame-input positions e(y_0)..e(y_{T-2}); with the
        # steer_frame0_predictor flag, also the last prompt position (the
        # frame-0 predictor) — mirroring DecodeStepSteering's flag exactly.
        first = pad + P - 1 if steer_frame0 else pad + P
        steer_mask[i, first: pad + P + T - 1] = True
        target_pos.append(torch.arange(pad + P - 1, pad + P + T - 1, device=device))

    steering = MaskedReplaySteering(
        backend.tts.model, backend.cfg.layer, vectors.to(device), steer_mask, alpha)
    with steering:
        out = backend.talker(
            inputs_embeds=inputs, attention_mask=attn,
            use_cache=False, output_hidden_states=True)
    expected = int(steer_mask.sum().item())
    if steering.calls != expected:
        raise RuntimeError(f"steering covered {steering.calls} positions, expected {expected}")

    all_logits = out.logits                       # [R, L, V_talker]
    hidden_layers = out.hidden_states[0] if isinstance(out.hidden_states, tuple) \
        else out.hidden_states
    last_hidden = hidden_layers[-1]               # [R, L, H]

    # gather per-row targets, flatten frames across the chunk
    cb0_logits = torch.cat([all_logits[i, target_pos[i]] for i in range(R)], dim=0)
    frame_hidden = torch.cat([last_hidden[i, target_pos[i]] for i in range(R)], dim=0)
    flat_codes = torch.cat([c.to(device) for c in codes], dim=0)   # [N, 16]

    # ---- codebook 0: STE over the pass-1 processing chain -------------------
    V_talker = cb0_logits.shape[-1]
    chain = LogitProcessingChain(sampling, V_talker, tc.codec_eos_token_id)
    # chain works on [B, T, V]; treat the flattened frames as one row per
    # sequence segment to keep the history-dependent penalty per row.
    probs_cb0 = []
    offset = 0
    for i in range(R):
        T = lengths[i]
        seg_logits = cb0_logits[offset:offset + T].unsqueeze(0)
        seg_codes = flat_codes[offset:offset + T, 0].unsqueeze(0)
        valid = torch.ones(1, T, dtype=torch.bool, device=device)
        processed = chain.process(seg_logits, seg_codes, valid)
        probs_cb0.append(torch.softmax(processed, dim=-1)[0])
        offset += T
    probs_cb0 = torch.cat(probs_cb0, dim=0)       # [N, V_talker]

    # ---- codebooks 1-15: subtalker finetune path, per-example --------------
    # The nested subtalker generate() samples with its own warpers
    # (temperature/top-k/top-p, defaults 0.9/50/1.0); the STE derivative must
    # replay the same warping or the residual gradient is inconsistent with
    # the distribution the codes were drawn from.
    sub_logits, _sub_loss = backend.talker.forward_sub_talker_finetune(
        flat_codes, frame_hidden)                 # [N, 15, V_codec]
    sub_processed = apply_warpers(
        sub_logits.to(torch.float32), sampling.subtalker_temperature,
        sampling.subtalker_top_k, sampling.subtalker_top_p)
    probs_res = torch.softmax(sub_processed, dim=-1)

    # ---- assemble STE one-hots over the codec vocabulary -------------------
    V_codec = probs_res.shape[-1]
    y_cb0 = one_hot_ste(flat_codes[:, 0].clamp_max(V_codec - 1),
                        probs_cb0[:, :V_codec]
                        / probs_cb0[:, :V_codec].sum(-1, keepdim=True).clamp_min(1e-8))
    y_res = one_hot_ste(flat_codes[:, 1:], probs_res)          # [N, 15, V_codec]
    y_all = torch.cat([y_cb0.unsqueeze(1), y_res], dim=1)      # [N, 16, V_codec]

    logp = torch.log(probs_cb0.gather(
        1, flat_codes[:, :1].clamp_max(V_talker - 1)).clamp_min(1e-9)).squeeze(1)

    onehots, offset, per_row_logp = [], 0, []
    for i in range(R):
        T = lengths[i]
        onehots.append(y_all[offset:offset + T])
        per_row_logp.append(logp[offset:offset + T].mean())
        offset += T
    return ReplayOutput(
        ste_onehots=onehots,
        codes=[c.to(device) for c in codes],
        logp_cb0_mean=torch.stack(per_row_logp).detach(),
    )
