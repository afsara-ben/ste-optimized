"""Layer-15 steering hooks with per-row vectors and norm restoration.

Steering contract (plan §3, verified against the live inference path):
- generation (pass 1): steer ONLY single-token decode steps; the prefill —
  including the last prompt position, whose hidden state produces frame-0
  logits — is left unsteered.
- replay (pass 2): steer exactly the frame-input positions, i.e. positions
  holding embeddings of frames y_0 … y_{T-2}; the last prompt position stays
  unsteered. A boolean mask expresses this per row.
- every steered hidden state is rescaled back to its ORIGINAL per-position L2
  norm ("renormalize"), matching the inference hook.

Hooks attach to the talker decoder layer `layer_index` via forward hooks and
modify its output hidden state in place of the tuple.
"""

from __future__ import annotations

import contextlib

import torch


def _renorm(shifted: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    orig_norm = original.norm(dim=-1, keepdim=True)
    new_norm = shifted.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return shifted * (orig_norm / new_norm)


def _apply(hidden: torch.Tensor, vectors: torch.Tensor, alpha: float) -> torch.Tensor:
    # hidden [B, S, H]; vectors [B, H] broadcast over S.
    shifted = hidden + alpha * vectors.to(hidden.dtype).unsqueeze(1)
    return _renorm(shifted, hidden)


def talker_layers(tts_model) -> torch.nn.ModuleList:
    """Decoder layers of the talker backbone (qwen_tts 0.1.1 layout)."""
    return tts_model.talker.model.layers


class _BaseSteering:
    def __init__(self, tts_model, layer_index: int) -> None:
        self._layer = talker_layers(tts_model)[layer_index]
        self._handle = None
        self.calls = 0

    def _hook(self, module, args, output):  # pragma: no cover - exercised on GPU
        raise NotImplementedError

    def __enter__(self):
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class DecodeStepSteering(_BaseSteering):
    """Pass-1 hook: per-row vectors applied on decode steps (seq_len == 1).

    With `steer_last_prefill=True` (the `steer_frame0_predictor` flag), the
    LAST prefill position — whose hidden state predicts frame 0 — is also
    steered; the rest of the prompt stays untouched. With native left padding,
    the last prompt token is the last position for every row.
    """

    def __init__(self, tts_model, layer_index: int, vectors: torch.Tensor,
                 alpha: float, steer_last_prefill: bool = False) -> None:
        super().__init__(tts_model, layer_index)
        if vectors.dim() != 2:
            raise ValueError("vectors must be [batch, hidden]")
        self.vectors = vectors
        self.alpha = alpha
        self.steer_last_prefill = steer_last_prefill

    def _hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != self.vectors.shape[0]:
            raise RuntimeError(
                f"steering rows {self.vectors.shape[0]} != batch {hidden.shape[0]}")
        if hidden.shape[1] == 1:  # decode step
            self.calls += 1
            steered = _apply(hidden, self.vectors, self.alpha)
        elif self.steer_last_prefill:  # prefill: steer frame-0 predictor only
            self.calls += 1
            last = _apply(hidden[:, -1:, :], self.vectors, self.alpha)
            steered = torch.cat([hidden[:, :-1, :], last], dim=1)
        else:  # prefill fully unsteered (historical convention)
            return output
        if isinstance(output, tuple):
            return (steered,) + tuple(output[1:])
        return steered


class MaskedReplaySteering(_BaseSteering):
    """Pass-2 hook: per-row vectors applied where `mask` is True.

    mask [B, S] must be True exactly at frame-input positions (plan §3 step 2);
    padded and prompt positions keep their original hidden states.
    """

    def __init__(self, tts_model, layer_index: int, vectors: torch.Tensor,
                 mask: torch.Tensor, alpha: float) -> None:
        super().__init__(tts_model, layer_index)
        self.vectors = vectors
        self.mask = mask
        self.alpha = alpha

    def _hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[:2] != self.mask.shape:
            raise RuntimeError(
                f"mask {tuple(self.mask.shape)} != hidden {tuple(hidden.shape[:2])}")
        self.calls += int(self.mask.sum().item())
        steered = _apply(hidden, self.vectors, self.alpha)
        merged = torch.where(self.mask.unsqueeze(-1), steered, hidden)
        if isinstance(output, tuple):
            return (merged,) + tuple(output[1:])
        return merged


class DecodeActivationCapture(_BaseSteering):
    """Extraction hook: record per-row hidden states of every decode step
    (seq_len == 1) at `layer_index`; used for mean-decode contrast vectors.

    Steps are stored per index because, in batched generation, rows that hit
    EOS early keep producing (padded) decode steps until the longest row
    finishes — the mean must only cover each row's own first T_r steps.
    """

    def __init__(self, tts_model, layer_index: int, batch_size: int) -> None:
        super().__init__(tts_model, layer_index)
        self.steps: list[torch.Tensor] = []
        self.batch_size = batch_size

    def _hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] != 1:
            return output
        self.steps.append(hidden[:, 0, :].detach().to(torch.float32).cpu())
        return output

    def mean(self, lengths: list[int]) -> torch.Tensor:
        """Per-row mean over that row's own decode steps only."""
        if not self.steps:
            raise RuntimeError("no decode steps captured")
        stacked = torch.stack(self.steps, dim=1)  # [B, T_max, H]
        out = torch.zeros(stacked.shape[0], stacked.shape[2])
        for row, t in enumerate(lengths):
            t = min(max(int(t), 1), stacked.shape[1])
            out[row] = stacked[row, :t].mean(dim=0)
        return out


@contextlib.contextmanager
def no_steering():
    yield None
