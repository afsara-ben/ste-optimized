"""Differentiable Whisper ASR loss for generated waveforms.

Hugging Face's :class:`~transformers.WhisperFeatureExtractor` returns NumPy
arrays, including when it uses torch internally.  That is correct for normal
ASR inference but severs a training graph.  This module mirrors its frontend
with torch operations only, then runs a frozen Whisper model while retaining
gradients with respect to the waveform.

The default model and revision are deliberately immutable.  Loading another
revision is an explicit caller choice, rather than an accidental change when a
Hub branch moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import nn
from torch.utils.checkpoint import checkpoint


WHISPER_SAMPLE_RATE = 16_000
WHISPER_LARGE_V3_TURBO_ID = "openai/whisper-large-v3-turbo"
WHISPER_LARGE_V3_TURBO_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


class DifferentiableWhisperFeatureExtractor(nn.Module):
    """Torch-native equivalent of a ``WhisperFeatureExtractor``.

    The feature-extractor instance remains the source of truth for the mel
    filter bank and all frontend dimensions.  Its NumPy mel matrix is copied
    once into a registered float32 buffer.  Audio longer than the configured
    Whisper window is rejected by default: truncating audio while retaining a
    full transcript would create an invalid teacher-forced objective.
    """

    def __init__(self, feature_extractor: Any) -> None:
        super().__init__()
        required = (
            "mel_filters",
            "sampling_rate",
            "n_fft",
            "hop_length",
            "n_samples",
            "nb_max_frames",
            "padding_value",
            "dither",
        )
        missing = [name for name in required if not hasattr(feature_extractor, name)]
        if missing:
            raise TypeError(
                "feature_extractor is not Whisper-compatible; missing "
                + ", ".join(missing)
            )

        mel_filters = torch.as_tensor(
            feature_extractor.mel_filters, dtype=torch.float32
        ).clone()
        if mel_filters.ndim != 2:
            raise ValueError("Whisper mel_filters must be a rank-2 matrix")

        self.register_buffer("mel_filters", mel_filters)
        self.sampling_rate = int(feature_extractor.sampling_rate)
        self.n_fft = int(feature_extractor.n_fft)
        self.hop_length = int(feature_extractor.hop_length)
        self.n_samples = int(feature_extractor.n_samples)
        self.nb_max_frames = int(feature_extractor.nb_max_frames)
        self.padding_value = float(feature_extractor.padding_value)
        self.dither = float(feature_extractor.dither)

        expected_bins = 1 + self.n_fft // 2
        if mel_filters.shape[0] != expected_bins:
            raise ValueError(
                f"mel filter bank has {mel_filters.shape[0]} frequency bins; "
                f"expected {expected_bins} for n_fft={self.n_fft}"
            )
        if self.n_samples // self.hop_length != self.nb_max_frames:
            raise ValueError("Whisper frontend sample/frame dimensions are inconsistent")

    @property
    def feature_size(self) -> int:
        return int(self.mel_filters.shape[1])

    def forward(
        self,
        waves: Sequence[torch.Tensor],
        *,
        sampling_rate: int = WHISPER_SAMPLE_RATE,
        truncate: bool = False,
    ) -> torch.Tensor:
        """Return ``[batch, mel, frames]`` features without detaching ``waves``.

        Args:
            waves: Non-empty sequence of mono waveforms.  Each item may be
                ``[samples]`` or ``[1, samples]``.
            sampling_rate: Common input sample rate.  Resampling, when needed,
                is performed by differentiable torchaudio operators.
            truncate: Match Hugging Face's default overlength behavior when
                true.  The safer training default is to reject overlength
                waveforms.
        """
        if not waves:
            raise ValueError("waves must contain at least one waveform")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        padded: list[torch.Tensor] = []
        for row, wave in enumerate(waves):
            if not isinstance(wave, torch.Tensor):
                raise TypeError(f"wave {row} is not a torch.Tensor")
            if wave.ndim == 2 and wave.shape[0] == 1:
                wave = wave[0]
            if wave.ndim != 1:
                raise ValueError(
                    f"wave {row} must be mono [samples] or [1, samples], "
                    f"got shape {tuple(wave.shape)}"
                )
            if wave.numel() == 0:
                raise ValueError(f"wave {row} is empty")

            # Feature computation is float32 in WhisperFeatureExtractor even
            # when model weights are fp16/bf16.  Device/dtype copies preserve
            # autograd edges.
            source = wave.to(device=self.mel_filters.device, dtype=torch.float32)
            if sampling_rate != self.sampling_rate:
                source = AF.resample(source, sampling_rate, self.sampling_rate)

            if source.shape[-1] > self.n_samples:
                if not truncate:
                    duration = source.shape[-1] / self.sampling_rate
                    limit = self.n_samples / self.sampling_rate
                    raise ValueError(
                        f"wave {row} is {duration:.3f}s, longer than Whisper's "
                        f"{limit:.3f}s frontend window"
                    )
                source = source[: self.n_samples]
            source = F.pad(
                source,
                (0, self.n_samples - source.shape[-1]),
                value=self.padding_value,
            )
            padded.append(source)

        waveform = torch.stack(padded)
        if self.dither != 0.0:
            waveform = waveform + self.dither * torch.randn_like(waveform)

        window = torch.hann_window(
            self.n_fft, device=waveform.device, dtype=waveform.dtype
        )
        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=window,
            return_complex=True,
        )
        # HF removes the final STFT frame before applying the mel bank.
        magnitudes = stft[..., :-1].abs().square()
        mel_spec = self.mel_filters.transpose(0, 1) @ magnitudes
        log_spec = mel_spec.clamp_min(1e-10).log10()
        per_row_max = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, per_row_max - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        if log_spec.shape[-1] != self.nb_max_frames:
            raise RuntimeError(
                f"frontend produced {log_spec.shape[-1]} frames; "
                f"expected {self.nb_max_frames}"
            )
        return log_spec


@dataclass(frozen=True)
class WhisperTargetBatch:
    """Tokenized teacher-forcing inputs and the independently chosen loss mask."""

    labels: torch.Tensor
    loss_mask: torch.Tensor


class WhisperASRExpert:
    """Frozen Whisper with per-row teacher-forced token NLL.

    Model parameters never receive gradients, but the forward pass is not put
    under ``no_grad``: NLL gradients traverse Whisper, the torch-native
    frontend, and the input waveform.  By default the loss covers transcript
    tokens plus EOS while excluding fixed language/task/timestamp control
    tokens.  Those choices are explicit constructor flags.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        *,
        model_id: str = WHISPER_LARGE_V3_TURBO_ID,
        revision: str = WHISPER_LARGE_V3_TURBO_REVISION,
        dtype: torch.dtype | None = None,
        language: str = "english",
        task: str = "transcribe",
        include_control_tokens: bool = False,
        include_eos: bool = True,
        local_files_only: bool = False,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        feature_extractor: Any | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model_id = model_id
        self.revision = revision
        self.language = language
        self.task = task
        self.include_control_tokens = include_control_tokens
        self.include_eos = include_eos

        injected = (model, tokenizer, feature_extractor)
        if any(item is not None for item in injected) and not all(
            item is not None for item in injected
        ):
            raise ValueError(
                "model, tokenizer, and feature_extractor must be injected together"
            )

        if model is None:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            processor = WhisperProcessor.from_pretrained(
                model_id,
                revision=revision,
                language=language,
                task=task,
                local_files_only=local_files_only,
            )
            tokenizer = processor.tokenizer
            feature_extractor = processor.feature_extractor
            if dtype is None:
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )

        assert model is not None
        assert tokenizer is not None
        assert feature_extractor is not None
        self.model = model.to(self.device)
        if dtype is not None:
            self.model.to(dtype=dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.tokenizer = tokenizer
        if not hasattr(self.tokenizer, "set_prefix_tokens"):
            raise TypeError("tokenizer must provide Whisper set_prefix_tokens()")
        self.tokenizer.set_prefix_tokens(
            language=language, task=task, predict_timestamps=False
        )

        if isinstance(feature_extractor, DifferentiableWhisperFeatureExtractor):
            self.frontend = feature_extractor.to(self.device)
        else:
            self.frontend = DifferentiableWhisperFeatureExtractor(
                feature_extractor
            ).to(self.device)
        self._validate_frontend_model_contract()

    def _validate_frontend_model_contract(self) -> None:
        config = self.model.config
        model_mels = int(config.num_mel_bins)
        if self.frontend.feature_size != model_mels:
            raise ValueError(
                f"frontend has {self.frontend.feature_size} mel bins, model expects "
                f"{model_mels}"
            )
        # Whisper has conv strides 1 and 2, hence two frontend frames per
        # encoder position.  Reading the modules also keeps this correct for
        # injected test models.
        encoder = self.model.get_encoder()
        stride = int(encoder.conv1.stride[0]) * int(encoder.conv2.stride[0])
        expected_frames = int(config.max_source_positions) * stride
        if self.frontend.nb_max_frames != expected_frames:
            raise ValueError(
                f"frontend has {self.frontend.nb_max_frames} frames, model expects "
                f"{expected_frames}"
            )

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def tokenize_targets(self, target_texts: Sequence[str]) -> WhisperTargetBatch:
        """Build shifted-label targets without masking decoder prompt inputs."""
        if not target_texts:
            raise ValueError("target_texts must not be empty")
        if any(not isinstance(text, str) for text in target_texts):
            raise TypeError("every target text must be a string")

        encoded = self.tokenizer(
            list(target_texts),
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        attention = torch.as_tensor(encoded["attention_mask"], dtype=torch.bool)
        if input_ids.ndim != 2 or input_ids.shape != attention.shape:
            raise ValueError("tokenizer returned invalid input_ids/attention_mask")

        decoder_start = int(self.model.config.decoder_start_token_id)
        if input_ids.shape[1] < 2 or not torch.all(input_ids[:, 0] == decoder_start):
            raise ValueError(
                "Whisper targets must begin with decoder_start_token_id exactly once"
            )

        # The model inserts decoder_start_token_id when it shifts labels.  HF's
        # Whisper fine-tuning collators therefore remove the tokenizer's first
        # start-of-transcript token before passing labels.
        labels = input_ids[:, 1:].clone()
        valid = attention[:, 1:].clone()
        labels[~valid] = -100
        loss_mask = valid.clone()

        prefix_tokens = list(self.tokenizer.prefix_tokens)
        if not prefix_tokens or int(prefix_tokens[0]) != decoder_start:
            raise ValueError("tokenizer prefix does not start with decoder_start_token_id")
        if not self.include_control_tokens:
            control_count = max(len(prefix_tokens) - 1, 0)
            loss_mask[:, :control_count] = False

        if not self.include_eos:
            # EOS is the last valid target in each row.  Remove it from the
            # scoring mask without changing labels used for decoder inputs.
            lengths = valid.long().sum(dim=1)
            rows = torch.arange(labels.shape[0])
            loss_mask[rows, lengths - 1] = False

        counts = loss_mask.sum(dim=1)
        if torch.any(counts == 0):
            bad = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"target rows {bad} contain no tokens selected for ASR NLL")

        max_target_positions = int(self.model.config.max_target_positions)
        if labels.shape[1] > max_target_positions:
            raise ValueError(
                f"tokenized target length {labels.shape[1]} exceeds Whisper limit "
                f"{max_target_positions}; refusing to truncate target_text"
            )

        return WhisperTargetBatch(
            labels=labels.to(self.device), loss_mask=loss_mask.to(self.device)
        )

    def loss(
        self,
        waves: Sequence[torch.Tensor],
        target_texts: Sequence[str],
        *,
        sampling_rate: int = WHISPER_SAMPLE_RATE,
        use_checkpoint: bool = True,
    ) -> torch.Tensor:
        """Return mean-token NLL for each waveform/transcript row, shape ``[B]``."""
        if len(waves) != len(target_texts):
            raise ValueError(
                f"got {len(waves)} waveforms but {len(target_texts)} target texts"
            )
        features = self.frontend(waves, sampling_rate=sampling_rate)
        features = features.to(dtype=self.model_dtype)
        targets = self.tokenize_targets(target_texts)

        def run(input_features: torch.Tensor) -> torch.Tensor:
            return self.model(
                input_features=input_features,
                labels=targets.labels,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            ).logits

        if use_checkpoint and torch.is_grad_enabled() and features.requires_grad:
            logits = checkpoint(run, features, use_reentrant=False)
        else:
            logits = run(features)

        token_nll = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(targets.labels)
        mask = targets.loss_mask.to(token_nll.dtype)
        return (token_nll * mask).sum(dim=1) / mask.sum(dim=1)

    @torch.no_grad()
    def transcribe(
        self,
        waves: Sequence[torch.Tensor],
        *,
        sampling_rate: int = WHISPER_SAMPLE_RATE,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Deterministic greedy transcription for diagnostics and WER."""
        features = self.frontend(waves, sampling_rate=sampling_rate).to(
            dtype=self.model_dtype
        )
        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "num_beams": 1,
            "language": self.language,
            "task": self.task,
            "return_timestamps": False,
            "use_cache": True,
        }
        if max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = max_new_tokens
        token_ids = self.model.generate(
            input_features=features, **generation_kwargs
        )
        return list(self.tokenizer.batch_decode(token_ids, skip_special_tokens=True))
