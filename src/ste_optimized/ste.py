"""Fixed-hard straight-through estimation and pass-1 logit-chain replay.

Fixed-hard STE (plan §3 step 7): forward value = the pass-1 sampled hard code,
gradient = the pass-2 replay probability:

    y = one_hot(code) - stop_gradient(p) + p,  p = softmax(processed logits)

`processed` MUST reproduce the exact logit-processing chain the codes were
sampled under, otherwise the STE derivative is inconsistent with the forward
value (plan §3 step 6). The chain below mirrors qwen_tts generate() defaults in
HF processor order (verified against modeling_qwen3_tts.py::generate):

    1. repetition penalty (1.05) over previously generated codebook-0 ids
    2. suppress_tokens: text-range ids [vocab-1024, vocab) except codec EOS
    3. min_new_tokens (2): EOS suppressed for the first steps
    4. temperature (0.9) -> top-k (50) -> top-p (1.0)

The gpu parity test (tests/test_parity.py) compares this chain against stored
per-step processed logits from a real generate() run and is the gate that must
pass before any training result is trusted.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import SamplingConfig


def one_hot_ste(codes: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """codes [..., ] long, probs [..., V] float -> STE one-hots [..., V]."""
    hard = F.one_hot(codes, num_classes=probs.shape[-1]).to(probs.dtype)
    return hard - probs.detach() + probs


def apply_warpers(x: torch.Tensor, temperature: float, top_k: int,
                  top_p: float) -> torch.Tensor:
    """HF warper chain (temperature -> top-k -> top-p), out-of-place.

    Used for codebook 0 (inside LogitProcessingChain) AND for the residual
    codebooks 1-15: the nested subtalker generate() samples with its own
    temperature/top-k/top-p (defaults 0.9/50/1.0), so the residual STE
    derivative must replay the same warping."""
    V = x.shape[-1]
    if temperature and temperature != 1.0:
        x = x / temperature
    if top_k and 0 < top_k < V:
        kth = torch.topk(x, top_k, dim=-1).values[..., -1:]
        x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
    if top_p and top_p < 1.0:
        sorted_x, idx = torch.sort(x, descending=True, dim=-1)
        probs_sorted = torch.softmax(sorted_x, dim=-1)
        cum = probs_sorted.cumsum(-1)
        remove = cum - probs_sorted >= top_p
        remove_orig = torch.zeros_like(remove).scatter(-1, idx, remove)
        x = torch.where(remove_orig, torch.full_like(x, float("-inf")), x)
    return x


class LogitProcessingChain:
    """Vectorised replay of the pass-1 processing chain over a whole batch.

    Applies to codebook-0 (talker) logits [B, T, V_talker] with per-row valid
    lengths. The repetition penalty is history-dependent: at step t it acts on
    the ids generated before t (the prompt side entered as embeddings and is
    not penalised, matching generate()).
    """

    def __init__(self, sampling: SamplingConfig, vocab_size: int,
                 codec_eos_id: int, suppress_text_range: bool = True) -> None:
        self.s = sampling
        self.vocab_size = vocab_size
        self.codec_eos_id = codec_eos_id
        self.suppress_text_range = suppress_text_range

    def process(self, logits: torch.Tensor, codes: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """logits [B, T, V] raw; codes [B, T] sampled cb0 ids; valid [B, T]
        bool. Returns processed logits (float32) with -inf where filtered.

        Implementation note: every step is OUT-OF-PLACE. The filter masks are
        constants of the stored trajectory (built under no_grad); gradient
        flows only through the surviving logit values.
        """
        x = logits.to(torch.float32)
        B, T, V = x.shape
        s = self.s
        neg_inf = torch.tensor(float("-inf"), device=x.device)

        # 1. repetition penalty over the strictly-previous generated ids.
        #    seen[b, t, v] = code v was generated before step t (constant mask).
        if s.repetition_penalty and s.repetition_penalty != 1.0:
            with torch.no_grad():
                seen = torch.zeros(B, T, V, dtype=torch.bool, device=x.device)
                rows = torch.arange(B, device=x.device)
                for t in range(1, T):
                    seen[:, t] = seen[:, t - 1]
                    prev = codes[:, t - 1].clamp(0, V - 1)
                    seen[rows, t, prev] |= valid[:, t - 1]
            penal = torch.where(x > 0, x / s.repetition_penalty,
                                x * s.repetition_penalty)
            x = torch.where(seen, penal, x)

        # 2. suppress text-range tokens (all but codec EOS).
        if self.suppress_text_range:
            lo = max(self.vocab_size - 1024, 0)
            mask = torch.zeros(V, dtype=torch.bool, device=x.device)
            mask[lo:self.vocab_size] = True
            mask[self.codec_eos_id] = False
            x = torch.where(mask.view(1, 1, V), neg_inf, x)

        # 3. EOS suppressed before min_new_tokens.
        if s.min_new_tokens > 0:
            t_cut = min(s.min_new_tokens, T)
            eos_mask = torch.zeros(T, V, dtype=torch.bool, device=x.device)
            eos_mask[:t_cut, self.codec_eos_id] = True
            x = torch.where(eos_mask.view(1, T, V), neg_inf, x)

        # 4. warpers: temperature -> top-k -> top-p.
        return apply_warpers(x, s.temperature, s.top_k, s.top_p)


def replay_probabilities(chain: LogitProcessingChain, logits: torch.Tensor,
                         codes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Processed softmax probabilities for the STE derivative. Gradient flows
    through `logits`; the processing masks are treated as constants (they were
    fixed by pass-1's sampled trajectory)."""
    processed = chain.process(logits, codes, valid)
    return torch.softmax(processed, dim=-1)
