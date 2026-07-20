"""Frozen expert models scoring generated audio INSIDE the training loss.

- emotion2vec (funasr `Emotion2vec` torch module): differentiable path is
  `extract_features(...)["x"]` -> mean pool over valid frames -> `model.proj`
  linear head -> softmax over emotion labels. Loss = -log p(target emotion).
- WavLM-SV (transformers WavLMForXVector): x-vector embeddings; loss =
  1 - cosine(embedding(generated), embedding(reference)); reference embeddings
  are cached (they never change).

Both models are frozen (requires_grad False) but stay in the autograd graph so
gradients flow through them into the transform. Waveforms arrive at the codec
output rate (24 kHz) and are differentiably resampled to 16 kHz.

Gate for trust: `Emotion2VecExpert.parity_check()` compares this differentiable
head against funasr's own inference scores on the same clip (gpu test).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.checkpoint import checkpoint

EXPERT_SAMPLE_RATE = 16_000


def resample_to_expert(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == EXPERT_SAMPLE_RATE:
        return wav
    return AF.resample(wav, sr, EXPERT_SAMPLE_RATE)


def _pad_stack(waves: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(w.shape[-1] for w in waves)
    out = torch.zeros(len(waves), L, device=waves[0].device, dtype=torch.float32)
    mask = torch.zeros(len(waves), L, dtype=torch.bool, device=waves[0].device)
    for i, w in enumerate(waves):
        out[i, : w.shape[-1]] = w.to(torch.float32)
        mask[i, : w.shape[-1]] = True
    return out, mask


class Emotion2VecExpert:
    def __init__(self, device: str = "cuda:0",
                 model_id: str = "emotion2vec/emotion2vec_plus_large") -> None:
        from funasr import AutoModel  # heavy import kept local

        self.auto = AutoModel(model=model_id, hub="hf", disable_update=True,
                              device=device)
        self.model = self.auto.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        tokenizer = self.auto.kwargs.get("tokenizer")
        self.labels: list[str] = list(getattr(tokenizer, "token_list", []) or [])

    def label_index(self, emotion: str) -> int:
        matches = [i for i, lb in enumerate(self.labels)
                   if emotion.lower() in lb.lower()]
        if not matches:
            raise ValueError(f"emotion {emotion!r} not found in labels {self.labels}")
        return matches[0]

    def logits(self, waves: list[torch.Tensor], use_checkpoint: bool = True) -> torch.Tensor:
        """waves: 16 kHz mono tensors (graph-attached). Returns [B, C]."""
        batch, mask = _pad_stack(waves)
        # emotion2vec convention: per-utterance layer norm of the raw waveform.
        mean = (batch * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        var = ((batch - mean) ** 2 * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        source = (batch - mean) / (var + 1e-5).sqrt() * mask
        padding_mask = ~mask  # funasr: True = padded

        def run(src):
            feats = self.model.extract_features(src, padding_mask=padding_mask,
                                                mask=False)
            x = feats["x"]                                   # [B, T', D]
            fpad = feats.get("padding_mask")
            if fpad is not None:
                keep = (~fpad).to(x.dtype).unsqueeze(-1)
                pooled = (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)
            else:
                pooled = x.mean(1)
            return self.model.proj(pooled)

        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(run, source, use_reentrant=False)
        return run(source)

    def loss(self, waves: list[torch.Tensor], target_emotion: str
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (per-row -log p(target) [B], per-row p(target) [B])."""
        logits = self.logits(waves)
        logp = torch.log_softmax(logits, dim=-1)
        idx = self.label_index(target_emotion)
        return -logp[:, idx], logp.exp()[:, idx].detach()

    @torch.no_grad()
    def parity_check(self, wav_16k: torch.Tensor, atol: float = 5e-2) -> bool:
        """Differentiable head vs funasr inference on one clip (gpu test)."""
        ours = torch.softmax(self.logits([wav_16k]), dim=-1)[0].cpu()
        ref = self.auto.generate(input=wav_16k.detach().cpu().numpy(),
                                 fs=EXPERT_SAMPLE_RATE, granularity="utterance",
                                 extract_embedding=False)[0]
        ref_scores = torch.tensor(ref["scores"])
        keep = [i for i, lb in enumerate(self.labels) if not lb.startswith("unuse")]
        return torch.allclose(ours[keep] / ours[keep].sum(),
                              ref_scores / ref_scores.sum(), atol=atol)


class WavLMSpeakerExpert:
    def __init__(self, device: str = "cuda:0",
                 model_id: str = "microsoft/wavlm-base-plus-sv") -> None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)
        self._ref_cache: dict[str, torch.Tensor] = {}

    def embed(self, waves: list[torch.Tensor], use_checkpoint: bool = True) -> torch.Tensor:
        batch, mask = _pad_stack(waves)
        # WavLM feature extractor at inference = zero-mean/unit-var per utt.
        mean = (batch * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        var = ((batch - mean) ** 2 * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
        source = (batch - mean) / (var + 1e-7).sqrt() * mask

        def run(src):
            out = self.model(input_values=src,
                             attention_mask=mask.long()).embeddings
            return F.normalize(out, dim=-1)

        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(run, source, use_reentrant=False)
        return run(source)

    @torch.no_grad()
    def reference_embedding(self, audio_path: str) -> torch.Tensor:
        if audio_path not in self._ref_cache:
            import librosa
            wav, _ = librosa.load(audio_path, sr=EXPERT_SAMPLE_RATE, mono=True)
            ref = self.embed([torch.from_numpy(wav).to(self.device)],
                             use_checkpoint=False)[0]
            self._ref_cache[audio_path] = ref
        return self._ref_cache[audio_path]

    def loss(self, waves: list[torch.Tensor], reference_paths: list[str]
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (1 - cosine [B], cosine.detach() [B])."""
        gen = self.embed(waves)
        refs = torch.stack([self.reference_embedding(p) for p in reference_paths])
        cos = (gen * refs).sum(-1)
        return 1.0 - cos, cos.detach()


@dataclass
class ExpertSuite:
    emotion: Emotion2VecExpert
    speaker: WavLMSpeakerExpert

    @classmethod
    def load(cls, device: str) -> "ExpertSuite":
        return cls(Emotion2VecExpert(device=device),
                   WavLMSpeakerExpert(device=device))
