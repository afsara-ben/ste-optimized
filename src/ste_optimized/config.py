"""Configuration dataclasses and YAML loading.

Every run is fully described by one YAML file (see configs/angry.yaml). All
defaults here mirror BATCHED_REPLAY_PLAN_PORTABLE.md; deviations must be set in
the YAML and are recorded verbatim in every artifact for provenance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"

# Canonical speaker partition — mandatory for every artifact (plan §2).
SPEAKER_PARTITION: dict[str, tuple[str, ...]] = {
    "train": ("0011", "0014", "0017", "0020"),
    "validation": ("0012", "0016"),
    "test": ("0013", "0019"),
    "reserve": ("0015", "0018"),
}


@dataclass
class ModelConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"  # fallback: "sdpa"
    device: str = "cuda:0"
    layer: int = 15
    hidden_size: int = 1024
    # Steer the LAST prefill position (the frame-0 predictor) in addition to
    # decode steps — applied symmetrically in generation, replay, and
    # evaluation. Default False = historical convention (prefill unsteered).
    steer_frame0_predictor: bool = False


@dataclass
class SamplingConfig:
    """Pass-1 sampling parameters. MUST match the logit chain replayed by
    ste.LogitProcessingChain for the STE gradient to be consistent (plan §3
    step 6). Defaults are qwen_tts generate() defaults, verified against
    modeling_qwen3_tts.py::generate."""

    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9
    repetition_penalty: float = 1.05
    min_new_tokens: int = 2
    max_frames: int = 192  # ~2 x p95 of natural code lengths; NOT 512
    language: str = "English"
    # Nested subtalker sampling (residual codebooks 1-15). qwen_tts generate()
    # defaults; the residual STE derivative replays the same warping.
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0


@dataclass
class DataConfig:
    emotion: str = "angry"
    dataset_dir: str = "data/prepared"  # output of `build-data`
    contrasts_path: str = ""  # output of `extract`; empty => must run extract
    bases_per_speaker: int = 5
    min_pair_emotion_prob: float = 0.2  # exclusion thresholds for fixed weights
    min_pair_speaker_sim: float = 0.85


@dataclass
class TrainConfig:
    contrasts_per_update: int = 10  # K
    bases_per_contrast: int = 4     # M (floor 2; 5 only for historical-parity arm)
    chunk_rows: int = 8             # pass-2 gradient-accumulation chunk size
    min_row_survival: float = 0.8   # accept update if >=80% rows EOS-terminate
    max_updates: int = 300
    eval_every: int = 25
    early_stop_patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    speaker_weight: float = 1.0     # weight of WavLM speaker loss vs emotion loss
    alpha: float = 1.0              # steering strength during training
    rank: int = 16
    seed: int = 42
    reg_identity: float = 0.01
    reg_cosine: float = 0.01
    reg_norm: float = 0.01
    checkpoint_every: int = 5
    output_dir: str = "runs/train"


@dataclass
class DistributedConfig:
    """Multi-GPU policy. 'seed_parallel' (recommended, plan §2): launch one
    independent process per GPU with different seeds — no code coupling.
    'ddp_rows': shard the R rows of each update across ranks under torchrun and
    all-reduce transform gradients; measured historically at only ~1.13x for
    intra-update parallelism, kept as an option."""

    mode: str = "none"  # none | seed_parallel | ddp_rows
    backend: str = "nccl"


@dataclass
class EvalConfig:
    panel_seeds: int = 1
    alphas: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
    cadence_rows: int = 12          # small fixed panel evaluated during training
    control_cache_dir: str = "runs/control-cache"
    compute_wer: bool = False       # whisper download on first use
    whisper_model: str = "openai/whisper-large-v3-turbo"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _build(cls, payload: dict[str, Any]):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sections = {
        "model": ModelConfig,
        "sampling": SamplingConfig,
        "data": DataConfig,
        "train": TrainConfig,
        "distributed": DistributedConfig,
        "eval": EvalConfig,
    }
    kwargs = {}
    for name, cls in sections.items():
        section = raw.pop(name, {}) or {}
        if isinstance(section.get("alphas"), list):
            section["alphas"] = tuple(section["alphas"])
        kwargs[name] = _build(cls, section)
    if raw:
        raise ValueError(f"unknown top-level config sections: {sorted(raw)}")
    return ExperimentConfig(**kwargs)
