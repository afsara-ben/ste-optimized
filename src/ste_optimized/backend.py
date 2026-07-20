"""Qwen3-TTS backend: model loading, prompt cache, batched steered generation.

Pass 1 uses the package's NATIVE batched `model.generate()` (which left-pads
prompts internally — verified) under a `DecodeStepSteering` hook carrying one
steering vector per row. We never reimplement the decode loop.

For pass-2 replay we assemble the exact prefill embeddings `generate()` builds
for the non-streaming ICL voice-clone path (replicated from
qwen_tts/core/models/modeling_qwen3_tts.py::generate and ::generate_icl_prompt;
the gpu parity test in tests/test_parity.py is the gate that this assembly is
byte-consistent with the running package version).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .config import ModelConfig, SamplingConfig
from .hooks import DecodeStepSteering


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


@dataclass
class PromptEntry:
    """Cached, immutable per-(base, target-text) prompt state."""

    base_id: str
    target_text: str
    reference_text: str
    reference_audio: str
    items: Any                      # qwen prompt items (ref codes + x-vector)
    input_id: torch.Tensor          # tokenized assistant text (template incl.)
    ref_id: torch.Tensor            # tokenized reference text
    prompt_embed: torch.Tensor      # [1, P, H] assembled prefill, cpu
    prompt_len: int

    @property
    def ref_code(self) -> torch.Tensor:
        """Reference codec codes [T_ref, 16] — the constant prefix native
        inference prepends before decoding (see codec.decode_soft)."""
        return self.items[0].ref_code


@dataclass
class GenerationBatch:
    codes: list[torch.Tensor]       # per row [T_r, 16] (cpu, trimmed)
    lengths: list[int]
    terminated: list[bool]          # EOS before max_frames
    wall_seconds: float


class QwenTTSBackend:
    def __init__(self, cfg: ModelConfig) -> None:
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel

        t0 = time.perf_counter()
        path = snapshot_download(repo_id=cfg.model_id, revision=cfg.model_revision)
        self.tts = Qwen3TTSModel.from_pretrained(
            path, device_map=cfg.device, dtype=_torch_dtype(cfg.dtype),
            attn_implementation=cfg.attn_implementation)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.load_seconds = time.perf_counter() - t0
        self.talker = self.tts.model.talker
        self.model_config = self.tts.model.config
        self.attention_runtime = {
            "requested": cfg.attn_implementation,
            "model": getattr(self.model_config, "_attn_implementation", None),
            "talker": getattr(self.talker.config, "_attn_implementation", None),
            "code_predictor": getattr(
                getattr(self.talker, "code_predictor", None), "config", None
            ),
        }
        code_predictor_cfg = self.attention_runtime["code_predictor"]
        self.attention_runtime["code_predictor"] = getattr(
            code_predictor_cfg, "_attn_implementation", None
        )
        if cfg.attn_implementation == "flash_attention_2":
            import flash_attn

            self.attention_runtime["flash_attn_version"] = flash_attn.__version__
            configured = (
                self.attention_runtime["model"],
                self.attention_runtime["talker"],
            )
            if any(value != "flash_attention_2" for value in configured):
                raise RuntimeError(
                    "FlashAttention was requested but Qwen did not retain "
                    f"flash_attention_2: {self.attention_runtime}"
                )
        self._prompt_cache: dict[tuple[str, str], PromptEntry] = {}
        for p in self.tts.model.parameters():
            p.requires_grad_(False)
        self.tts.model.eval()
        # The speech tokenizer is a PLAIN-class wrapper holding its own
        # nn.Module — invisible to tts.model.parameters(). Freeze it
        # explicitly or the codec accumulates gradients every backward.
        st = getattr(self.tts.model, "speech_tokenizer", None)
        if st is not None and getattr(st, "model", None) is not None:
            st.model.eval()
            for p in st.model.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ embeds
    def _text_proj(self, ids: torch.Tensor) -> torch.Tensor:
        return self.talker.text_projection(self.talker.get_text_embeddings()(ids))

    def _codec_embed_ids(self, ids: list[int]) -> torch.Tensor:
        t = torch.tensor([ids], device=self.device, dtype=torch.long)
        return self.talker.get_input_embeddings()(t)

    def special_text_embeds(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.model_config
        ids = torch.tensor([[c.tts_bos_token_id, c.tts_eos_token_id,
                             c.tts_pad_token_id]], device=self.device)
        bos, eos, pad = self._text_proj(ids).chunk(3, dim=1)
        return bos, eos, pad

    def frame_embeds(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [T, 16] -> per-frame replay input embeddings [T, H]:
        sum of the 16 codebook embeddings + tts_pad conditioning
        (non-streaming trailing conditioning is the pad embed at every step)."""
        codes = codes.to(self.device)
        emb = self.talker.get_input_embeddings()(codes[:, 0])
        preds = self.talker.code_predictor.get_input_embeddings()
        for g in range(1, codes.shape[1]):
            emb = emb + preds[g - 1](codes[:, g])
        _, _, pad = self.special_text_embeds()
        return emb + pad[0, 0]

    # -------------------------------------------------------- prompt assembly
    def build_prompt_embed(self, input_id: torch.Tensor, ref_id: torch.Tensor,
                           ref_code: torch.Tensor, spk_embed: torch.Tensor,
                           language: str) -> torch.Tensor:
        """Non-streaming ICL voice-clone prefill, replicated from generate()."""
        tc = self.model_config.talker_config
        lang = language.lower()
        if lang not in tc.codec_language_id:
            raise ValueError(f"language {language!r} not supported")
        language_id = tc.codec_language_id[lang]

        bos, eos, pad = self.special_text_embeds()
        codec_emb_0 = self._codec_embed_ids(
            [tc.codec_think_id, tc.codec_think_bos_id, language_id, tc.codec_think_eos_id])
        codec_emb_1 = self._codec_embed_ids([tc.codec_pad_id, tc.codec_bos_id])
        codec_input = torch.cat(
            [codec_emb_0,
             spk_embed.to(self.device, codec_emb_0.dtype).view(1, 1, -1),
             codec_emb_1], dim=1)
        role = self._text_proj(input_id[:, :3].to(self.device))
        body = torch.cat(
            [pad.expand(-1, codec_input.shape[1] - 2, -1), bos], dim=1
        ) + codec_input[:, :-1]
        head = torch.cat([role, body], dim=1)
        icl, _trailing = self.tts.model.generate_icl_prompt(
            text_id=input_id[:, 3:-5].to(self.device),
            ref_id=ref_id[:, 3:-2].to(self.device),
            ref_code=ref_code.to(self.device),
            tts_pad_embed=pad, tts_eos_embed=eos, non_streaming_mode=True)
        return torch.cat([head, icl], dim=1)

    # ---------------------------------------------------------- prompt cache
    def prepare_voice_clone_prompts(
        self, rows: list[dict[str, str]], language: str = "English",
    ) -> list[PromptEntry]:
        """rows: dicts with base_id, target_text, reference_text,
        reference_audio. Reference encode + prompt assembly are cached by
        (base_id, target_text) — the warm path costs ~0."""
        out = []
        for row in rows:
            key = (row["base_id"], row["target_text"])
            if key not in self._prompt_cache:
                items = self.tts.create_voice_clone_prompt(
                    ref_audio=row["reference_audio"],
                    ref_text=row["reference_text"],
                    x_vector_only_mode=False)
                input_id = self.tts._tokenize_texts(
                    [self.tts._build_assistant_text(row["target_text"])])
                if isinstance(input_id, list):
                    input_id = input_id[0]
                if input_id.dim() == 1:
                    input_id = input_id.unsqueeze(0)
                ref_id = self.tts._tokenize_texts(
                    [self.tts._build_ref_text(row["reference_text"])])
                if isinstance(ref_id, list):
                    ref_id = ref_id[0]
                if ref_id.dim() == 1:
                    ref_id = ref_id.unsqueeze(0)
                item = items[0]
                with torch.no_grad():
                    embed = self.build_prompt_embed(
                        input_id, ref_id, item.ref_code,
                        item.ref_spk_embedding, language)
                self._prompt_cache[key] = PromptEntry(
                    base_id=row["base_id"], target_text=row["target_text"],
                    reference_text=row["reference_text"],
                    reference_audio=row["reference_audio"], items=items,
                    input_id=input_id, ref_id=ref_id,
                    prompt_embed=embed.to("cpu"), prompt_len=embed.shape[1])
            out.append(self._prompt_cache[key])
        return out

    # ------------------------------------------------------------ generation
    def generate_prepared_batch(
        self, entries: list[PromptEntry], vectors: torch.Tensor | None,
        sampling: SamplingConfig, seed: int, alpha: float = 1.0,
        capture_hook=None,
    ) -> GenerationBatch:
        """One native batched generate over all rows. `vectors` is [B, H]
        (per-row steering, applied at decode steps only) or None for unsteered.
        Reproducibility contract: one seed per ordered batch identity; batched
        sampling does NOT reproduce single-stream token streams (plan §5)."""
        B = len(entries)
        items = [e.items[0] for e in entries]
        prompt = self.tts._prompt_items_to_voice_clone_prompt(items)
        input_ids = [e.input_id for e in entries]
        ref_ids = [e.ref_id for e in entries]

        torch.manual_seed(seed)
        hooks = []
        if vectors is not None:
            hooks.append(DecodeStepSteering(
                self.tts.model, self.cfg.layer, vectors.to(self.device), alpha,
                steer_last_prefill=self.cfg.steer_frame0_predictor))
        if capture_hook is not None:
            hooks.append(capture_hook)
        t0 = time.perf_counter()
        try:
            for h in hooks:
                h.__enter__()
            with torch.no_grad():
                codes_list, _ = self.tts.model.generate(
                    input_ids=input_ids, ref_ids=ref_ids,
                    voice_clone_prompt=prompt, languages=[sampling.language] * B,
                    non_streaming_mode=True,
                    max_new_tokens=sampling.max_frames,
                    do_sample=sampling.do_sample, top_k=sampling.top_k,
                    top_p=sampling.top_p, temperature=sampling.temperature,
                    repetition_penalty=sampling.repetition_penalty,
                    subtalker_dosample=sampling.do_sample,
                    subtalker_top_k=sampling.subtalker_top_k,
                    subtalker_top_p=sampling.subtalker_top_p,
                    subtalker_temperature=sampling.subtalker_temperature)
        finally:
            for h in reversed(hooks):
                h.__exit__(None, None, None)
        wall = time.perf_counter() - t0
        codes = [c.detach().to("cpu") for c in codes_list]
        lengths = [int(c.shape[0]) for c in codes]
        terminated = [ln < sampling.max_frames for ln in lengths]
        return GenerationBatch(codes=codes, lengths=lengths,
                               terminated=terminated, wall_seconds=wall)

    # ---------------------------------------------------------------- decode
    def decode_hard(self, codes: torch.Tensor,
                    ref_codes: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
        """Frozen (non-differentiable) waveform decode.

        With `ref_codes`, mirrors native voice-clone inference exactly:
        decode cat([ref, generated]) then trim the reference span
        proportionally (qwen3_tts_model.py:614-629)."""
        if ref_codes is None:
            wavs, sr = self.tts.model.speech_tokenizer.decode(
                {"audio_codes": [codes]})
            return torch.from_numpy(wavs[0]), sr
        combined = torch.cat([ref_codes.to(codes.device, codes.dtype), codes], dim=0)
        wavs, sr = self.tts.model.speech_tokenizer.decode(
            {"audio_codes": [combined]})
        wav = torch.from_numpy(wavs[0])
        cut = int(ref_codes.shape[0] / max(combined.shape[0], 1) * wav.shape[0])
        return wav[cut:], sr
