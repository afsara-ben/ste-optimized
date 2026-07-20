"""CPU-only contracts for short, bounded trainer smoke runs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import ste_optimized.training as training_module
from ste_optimized.config import ExperimentConfig
from ste_optimized.data import BaseRecord
from ste_optimized.distributed import DistributedContext
from ste_optimized.training import Trainer


class _SpeechTokenizer:
    def get_output_sample_rate(self) -> int:
        return 24_000


def test_init_accepts_preloaded_models_and_in_memory_data(
    tmp_path, monkeypatch,
):
    """A one-pair smoke can reuse extraction models without touching disk."""

    def unexpected(*_args, **_kwargs):
        raise AssertionError("preloaded/in-memory resource was loaded again")

    monkeypatch.setattr(training_module, "QwenTTSBackend", unexpected)
    monkeypatch.setattr(training_module.ExpertSuite, "load", unexpected)
    monkeypatch.setattr(training_module, "load_contrasts", unexpected)
    monkeypatch.setattr(training_module, "load_bases", unexpected)

    backend = SimpleNamespace(
        device=torch.device("cpu"),
        tts=SimpleNamespace(
            model=SimpleNamespace(speech_tokenizer=_SpeechTokenizer())),
    )
    experts = SimpleNamespace()
    contrasts = [{
        "pair_id": "0011:0001",
        "v": torch.arange(4, dtype=torch.float32),
        "emotion_prob": 1.0,
        "speaker_sim": 1.0,
    }]
    bases = [BaseRecord(
        base_id="0012:0002", speaker="0012", index="0002",
        target_text="target", reference_index="0003",
        reference_text="reference", reference_audio="neutral.wav",
        split="train",
    )]
    cfg = ExperimentConfig()
    cfg.asr.enabled = False
    cfg.model.device = "cpu"
    cfg.model.hidden_size = 4
    cfg.train.rank = 2
    cfg.train.contrasts_per_update = 1
    cfg.train.bases_per_contrast = 1
    cfg.train.output_dir = str(tmp_path)

    trainer = Trainer(
        cfg, DistributedContext(), backend=backend, experts=experts,
        contrasts=contrasts, bases=bases,
    )

    assert trainer.backend is backend
    assert trainer.experts is experts
    assert trainer.sr == 24_000
    batch = trainer.sampler.next_batch()
    assert batch.contrast_ids == ["0011:0001"]
    assert batch.rows[0]["base_id"] == "0012:0002"


def _loop_only_trainer(tmp_path, *, max_attempts=None, max_wall_seconds=None):
    """Construct only the state used by Trainer.train, without model loads."""
    trainer = object.__new__(Trainer)
    cfg = ExperimentConfig()
    cfg.train.max_updates = 20
    cfg.train.max_attempts = max_attempts
    cfg.train.max_wall_seconds = max_wall_seconds
    cfg.train.eval_every = 1
    trainer.cfg = cfg
    trainer.dist = SimpleNamespace(is_main=True)
    trainer.out = tmp_path
    trainer.completed = 0
    trainer.attempts = 0
    trainer.best_metric = -float("inf")
    trainer.evals_without_improvement = 0
    logs = []
    exports = []
    trainer._log = logs.append
    trainer._export = lambda path: exports.append(path)
    return trainer, logs, exports


def test_attempt_bound_counts_skips_and_only_evaluates_advanced_updates(tmp_path):
    trainer, logs, exports = _loop_only_trainer(tmp_path, max_attempts=3)
    outcomes = iter([False, True, False])

    def run_update():
        advanced = next(outcomes)
        if advanced:
            trainer.completed += 1
        return {"skipped": not advanced}

    eval_updates = []
    trainer.run_update = run_update
    trainer.train(cadence_eval=lambda current: eval_updates.append(
        current.completed) or 0.5)

    assert trainer.attempts == 3
    assert trainer.completed == 1
    assert eval_updates == [1]
    assert any(
        row["event"] == "bounded_stop" and row["reason"] == "max_attempts"
        for row in logs
    )
    assert exports == [
        tmp_path / "best_transform.pt",
        tmp_path / "final_transform.pt",
    ]


def test_zero_wall_bound_stops_before_update_and_still_exports(tmp_path):
    trainer, logs, exports = _loop_only_trainer(
        tmp_path, max_wall_seconds=0.0)

    def unexpected_update():
        raise AssertionError("wall-bounded trainer started an update")

    trainer.run_update = unexpected_update
    trainer.train()

    assert trainer.attempts == 0
    assert logs[0]["event"] == "bounded_stop"
    assert logs[0]["reason"] == "max_wall_seconds"
    assert exports == [tmp_path / "final_transform.pt"]


def test_failed_cadence_sentinel_is_not_exported_as_best(tmp_path):
    trainer, logs, exports = _loop_only_trainer(tmp_path, max_attempts=1)

    def run_update():
        trainer.completed += 1
        return {"skipped": False}

    trainer.run_update = run_update
    trainer.train(cadence_eval=lambda _current: -1.0)

    cadence = next(row for row in logs if row["event"] == "cadence_eval")
    assert cadence["eligible"] is False
    assert cadence["improved"] is False
    assert trainer.best_metric == -float("inf")
    assert exports == [tmp_path / "final_transform.pt"]


class _TwoParameterModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.values = nn.ParameterList([
            nn.Parameter(torch.zeros(1), requires_grad=False),
            nn.Parameter(torch.zeros(1), requires_grad=False),
        ])


@pytest.mark.parametrize("violation", ["requires_grad", "gradient"])
def test_gradient_contract_checks_every_frozen_parameter(violation):
    trainer = object.__new__(Trainer)
    trainer.transform = nn.Linear(1, 1)
    for parameter in trainer.transform.parameters():
        parameter.grad = torch.ones_like(parameter)

    talker = _TwoParameterModule()
    if violation == "requires_grad":
        talker.values[1].requires_grad_(True)
    else:
        talker.values[1].grad = torch.ones_like(talker.values[1])
    codec = _TwoParameterModule()
    emotion = _TwoParameterModule()
    speaker = _TwoParameterModule()
    trainer.backend = SimpleNamespace(
        talker=talker,
        tts=SimpleNamespace(model=SimpleNamespace(
            speech_tokenizer=SimpleNamespace(model=codec))),
    )
    trainer.experts = SimpleNamespace(
        emotion=SimpleNamespace(model=emotion),
        speaker=SimpleNamespace(model=speaker),
    )

    with pytest.raises(RuntimeError, match=r"talker.*parameter 1"):
        trainer._assert_gradient_contract()


def test_gradient_contract_rejects_zero_transform_gradient():
    trainer = object.__new__(Trainer)
    trainer.transform = nn.Linear(1, 1)
    for parameter in trainer.transform.parameters():
        parameter.grad = torch.zeros_like(parameter)

    frozen = _TwoParameterModule()
    trainer.backend = SimpleNamespace(
        talker=frozen,
        tts=SimpleNamespace(model=SimpleNamespace(
            speech_tokenizer=SimpleNamespace(model=frozen))),
    )
    trainer.experts = SimpleNamespace(
        emotion=SimpleNamespace(model=frozen),
        speaker=SimpleNamespace(model=frozen),
    )

    with pytest.raises(RuntimeError, match="exactly zero"):
        trainer._assert_gradient_contract()
