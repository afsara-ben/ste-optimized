"""CPU-only contracts for multi-checkpoint validation and selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ste_optimized.evaluation as evaluation
from ste_optimized.config import ExperimentConfig
from ste_optimized.transform import LowRankTransform, save_transform


def _transform(scale: float) -> LowRankTransform:
    transform = LowRankTransform(hidden_size=2, rank=1)
    with torch.no_grad():
        transform.down.copy_(torch.tensor([[1.0, 0.0]]))
        transform.up.copy_(torch.tensor([[scale], [0.0]]))
    return transform


def _save(path, transform: LowRankTransform, update: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_transform(path, transform, {"completed_updates": update})


def test_limited_panel_interleaves_speakers_without_changing_full_mapping(
    monkeypatch,
):
    contrasts = [
        {"pair_id": f"0012:a{index}", "v": torch.zeros(2)}
        for index in range(3)
    ] + [
        {"pair_id": f"0016:b{index}", "v": torch.zeros(2)}
        for index in range(3)
    ]
    bases = [
        SimpleNamespace(base_id="base:0", speaker="0090"),
        SimpleNamespace(base_id="base:1", speaker="0091"),
    ]
    monkeypatch.setattr(evaluation, "load_contrasts", lambda _path: contrasts)
    monkeypatch.setattr(
        evaluation, "load_bases", lambda _dataset, _split: bases
    )
    cfg = ExperimentConfig()
    cfg.data.contrasts_path = "contrasts-train.pt"

    full = evaluation._panel_rows(cfg, None)
    limited = evaluation._panel_rows(cfg, 4)

    assert [row["contrast"]["pair_id"] for row in full] == [
        "0012:a0", "0012:a1", "0012:a2",
        "0016:b0", "0016:b1", "0016:b2",
    ]
    assert [row["contrast"]["pair_id"] for row in limited] == [
        "0012:a0", "0016:b0", "0012:a1", "0016:b1",
    ]
    assert {row["contrast"]["pair_id"]: row["base"].base_id for row in limited} == {
        row["contrast"]["pair_id"]: row["base"].base_id for row in full
        if row["contrast"]["pair_id"] in {
            "0012:a0", "0016:b0", "0012:a1", "0016:b1",
        }
    }


def test_test_panel_resolves_filename_split_and_loads_matching_bases(monkeypatch):
    seen = {}
    contrasts = [{"pair_id": "0013:test", "v": torch.zeros(2)}]
    bases = [SimpleNamespace(base_id="0019:base", speaker="0019")]

    def load_contrasts(path):
        seen["contrasts"] = path
        return contrasts

    def load_bases(dataset, split):
        seen["bases"] = (dataset, split)
        return bases

    monkeypatch.setattr(evaluation, "load_contrasts", load_contrasts)
    monkeypatch.setattr(evaluation, "load_bases", load_bases)
    cfg = ExperimentConfig()
    cfg.data.dataset_dir = "data/prepared/angry"
    cfg.data.contrasts_path = (
        "data/training-artifacts/contrasts-angry-train.pt"
    )

    rows = evaluation._panel_rows(cfg, None, split="test")

    assert seen["contrasts"] == (
        "data/training-artifacts/contrasts-angry-test.pt"
    )
    assert seen["bases"] == ("data/prepared/angry", "test")
    assert rows[0]["contrast"]["pair_id"] == "0013:test"


def test_checkpoint_panel_rejects_non_heldout_split_before_loading(tmp_path):
    with pytest.raises(ValueError, match="evaluation split.*reserve"):
        evaluation.checkpoint_panel(
            ExperimentConfig(), tmp_path / "missing-run", split="reserve"
        )


def test_checkpoint_panel_loads_models_once_and_reuses_alpha_controls(
    tmp_path, monkeypatch,
):
    run = tmp_path / "run"
    first = _transform(0.5)
    second = _transform(1.0)
    _save(run / "checkpoints" / "transform-00001.pt", first, 1)
    _save(run / "best_transform.pt", first, 1)  # identical alias
    _save(run / "final_transform.pt", second, 2)

    cfg = ExperimentConfig()
    cfg.model.device = "cpu"
    cfg.model.hidden_size = 2
    cfg.train.rank = 1
    cfg.train.seed = 7
    cfg.eval.alphas = (0.5, 1.0)
    cfg.asr.min_validation_reference_words = 100

    loads = {"backend": 0, "experts": 0, "asr": 0}

    def load_backend(_cfg):
        loads["backend"] += 1
        return SimpleNamespace(device=torch.device("cpu"))

    def load_experts(_device):
        loads["experts"] += 1
        return object()

    def load_asr(_cfg):
        loads["asr"] += 1
        return object()

    rows = [
        {
            "contrast": {"pair_id": f"speaker:{i}", "v": torch.tensor([1., 0.])},
            "base": SimpleNamespace(
                base_id=f"base:{i}", target_text="clear angry speech"
            ),
        }
        for i in range(4)
    ]
    calls = []

    def score(_backend, _experts, _asr, _cfg, panel, vectors, alpha, seed):
        assert seed == 7
        strength = float(vectors[0, 0])
        calls.append((strength, alpha))
        return [
            {
                "pair_id": row["contrast"]["pair_id"],
                "base_id": row["base"].base_id,
                "target_text": row["base"].target_text,
                "terminated": True,
                "emotion_prob": 0.1 + strength * alpha * 0.1,
                "speaker_sim": 0.92,
                "transcript": row["base"].target_text,
            }
            for row in panel
        ]

    monkeypatch.setattr(evaluation, "QwenTTSBackend", load_backend)
    monkeypatch.setattr(
        evaluation, "ExpertSuite", SimpleNamespace(load=load_experts)
    )
    monkeypatch.setattr(evaluation, "_load_asr", load_asr)
    panel_limits = []

    def panel_rows(_cfg, limit, split="validation"):
        panel_limits.append((limit, split))
        return rows if limit is None else rows[:limit]

    monkeypatch.setattr(evaluation, "_panel_rows", panel_rows)
    monkeypatch.setattr(evaluation, "_generate_and_score", score)

    output = tmp_path / "reports" / "checkpoints.json"
    report = evaluation.checkpoint_panel(
        cfg, run, output, rows_limit=3, alphas=[0.25, 1.0], split="test"
    )

    assert loads == {"backend": 1, "experts": 1, "asr": 1}
    assert len(calls) == 6  # 2 controls + 2 unique transforms x 2 alphas
    assert sum(strength == 1.0 for strength, _alpha in calls) == 2
    assert panel_limits == [(3, "test")]
    assert report["panel"]["split"] == "test"
    assert report["panel"]["rows"] == 3
    assert report["panel"]["requested_rows"] == 3
    assert report["panel"]["alphas"] == [0.25, 1.0]
    assert report["panel"]["mode"] == "provisional_screen"
    assert report["wer_contract"]["min_reference_words"] == 1
    assert report["wer_contract"][
        "configured_full_panel_min_reference_words"
    ] == 100
    assert report["wer_contract"]["provisional_screen"] is True
    assert report["snapshots"] == 2
    assert report["candidate_count"] == 4
    assert report["candidates"][0]["artifacts"] == [
        "checkpoints/transform-00001.pt",
        "best_transform.pt",
    ]
    assert report["selection"]["feasible"] is True
    assert report["selection"]["best"]["completed_updates"] == 2
    assert report["selection"]["best"]["alpha"] == 1.0
    assert report["candidates"][0]["rows"][0]["pair_id"] == "speaker:0"
    assert report["candidates"][0]["rows"][0]["wer"]["treated"][
        "errors"
    ] == 0
    assert len(report["per_checkpoint_alpha_selection"]) == 2
    assert report["per_checkpoint_alpha_selection"][0]["coverage"] == 1.0
    assert report["per_checkpoint_alpha_selection"][0]["improved_count"] == 3
    assert output.is_file()


def test_generate_and_score_caps_contiguous_batches_and_matches_chunk_seeds(
    monkeypatch,
):
    cfg = ExperimentConfig()
    cfg.eval.generation_batch_rows = 32
    cfg.eval.compute_wer = True
    rows = [
        {
            "contrast": {"pair_id": f"speaker:{index}"},
            "base": SimpleNamespace(
                base_id=f"base:{index}",
                target_text=f"row {index}",
                reference_text="reference",
                reference_audio=f"ref-{index}.wav",
            ),
        }
        for index in range(65)
    ]
    vectors = torch.stack([
        torch.tensor([float(index), -float(index)]) for index in range(65)
    ])

    class Backend:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = []

        def prepare_voice_clone_prompts(self, prompt_rows):
            return [SimpleNamespace(
                base_id=row["base_id"], ref_code=torch.tensor([0])
            ) for row in prompt_rows]

        def generate_prepared_batch(
            self, entries, chunk_vectors, _sampling, seed, alpha,
        ):
            indexes = [int(entry.base_id.split(":")[1]) for entry in entries]
            self.calls.append({
                "indexes": indexes,
                "vectors": chunk_vectors[:, 0].tolist(),
                "seed": seed,
                "alpha": alpha,
            })
            return SimpleNamespace(
                codes=[torch.tensor([index]) for index in indexes],
                terminated=[True] * len(indexes),
                lengths=[1] * len(indexes),
            )

        def decode_hard(self, code, ref_codes):
            assert ref_codes.tolist() == [0]
            return code.to(torch.float32), 16_000

    class Emotion:
        def __init__(self):
            self.batch_sizes = []

        def loss(self, waves, _emotion):
            self.batch_sizes.append(len(waves))
            return torch.tensor(0.0), torch.tensor([
                float(wave[0]) / 100 for wave in waves
            ])

    class Speaker:
        def __init__(self):
            self.batch_sizes = []

        def loss(self, waves, refs):
            assert len(waves) == len(refs)
            self.batch_sizes.append(len(waves))
            return torch.tensor(0.0), torch.full((len(waves),), 0.9)

    class ASR:
        def __init__(self):
            self.batch_sizes = []

        def transcribe(self, waves):
            self.batch_sizes.append(len(waves))
            return [f"row {int(wave[0])}" for wave in waves]

    backend = Backend()
    emotion = Emotion()
    speaker = Speaker()
    asr = ASR()
    experts = SimpleNamespace(emotion=emotion, speaker=speaker)
    monkeypatch.setattr(evaluation, "resample_to_expert", lambda wave, _sr: wave)

    first = evaluation._generate_and_score(
        backend, experts, asr, cfg, rows, vectors, alpha=0.5, seed=99
    )
    second = evaluation._generate_and_score(
        backend, experts, asr, cfg, rows, vectors, alpha=1.0, seed=99
    )

    assert [len(call["indexes"]) for call in backend.calls] == [32, 32, 1] * 2
    assert backend.calls[0]["indexes"] == list(range(32))
    assert backend.calls[1]["indexes"] == list(range(32, 64))
    assert backend.calls[2]["indexes"] == [64]
    assert backend.calls[1]["vectors"] == [float(i) for i in range(32, 64)]
    first_seeds = [call["seed"] for call in backend.calls[:3]]
    second_seeds = [call["seed"] for call in backend.calls[3:]]
    assert first_seeds == second_seeds
    assert first_seeds[0] == 99
    assert len(set(first_seeds)) == 3
    assert emotion.batch_sizes == [32, 32, 1] * 2
    assert speaker.batch_sizes == [32, 32, 1] * 2
    assert asr.batch_sizes == [32, 32, 1] * 2
    assert [row["pair_id"] for row in first] == [
        f"speaker:{index}" for index in range(65)
    ]
    assert [row["pair_id"] for row in second] == [
        f"speaker:{index}" for index in range(65)
    ]
    assert first[-1]["transcript"] == "row 64"


def _gate_rows(treated_errors: int):
    reference_words = ["word"] * 100
    hypothesis_words = ["error"] * treated_errors + ["word"] * (
        100 - treated_errors
    )
    target = " ".join(reference_words)
    return (
        [{
            "target_text": target,
            "terminated": True,
            "emotion_prob": 0.20,
            "speaker_sim": 0.92,
            "transcript": target,
        }],
        [{
            "target_text": target,
            "terminated": True,
            "emotion_prob": 0.30,
            "speaker_sim": 0.91,
            "transcript": " ".join(hypothesis_words),
        }],
    )


def test_checkpoint_gates_accept_exact_six_point_wer_boundary():
    cfg = ExperimentConfig()
    cfg.asr.min_validation_reference_words = 100
    control, treated = _gate_rows(treated_errors=6)

    metrics = evaluation._gates(control, treated, cfg, n_boot=20)

    assert metrics["wer"]["delta"] == pytest.approx(0.06)
    assert metrics["gate_wer"] is True
    assert metrics["emotion_improved_pairs"] == 1
    assert metrics["emotion_improved_rate"] == 1.0
    assert metrics["speaker_sim_control"] == pytest.approx(0.92)
    assert metrics["speaker_degradation"] == pytest.approx(0.01)
    assert metrics["isr"] == 1.0
    assert metrics["pass"] is True


def test_matched_row_diagnostics_include_counts_and_fail_closed_constraints():
    cfg = ExperimentConfig()
    target = " ".join(["word"] * 100)
    treated_text = " ".join(["error"] * 6 + ["word"] * 94)
    control = [{
        "pair_id": "0012:one",
        "base_id": "0016:base",
        "target_text": target,
        "terminated": True,
        "emotion_prob": 0.20,
        "speaker_sim": 0.92,
        "transcript": target,
    }]
    treated = [{
        "pair_id": "0012:one",
        "base_id": "0016:base",
        "target_text": target,
        "terminated": True,
        "emotion_prob": 0.50,
        "speaker_sim": 0.88,
        "transcript": treated_text,
    }]

    row = evaluation._matched_row_diagnostics(control, treated, cfg)[0]

    assert row["angry_prob"] == {
        "control": pytest.approx(0.20),
        "treated": pytest.approx(0.50),
        "delta": pytest.approx(0.30),
    }
    assert row["speaker_sim"]["degradation"] == pytest.approx(0.04)
    assert row["wer"]["control"]["errors"] == 0
    assert row["wer"]["treated"]["substitutions"] == 6
    assert row["wer"]["treated"]["reference_words"] == 100
    assert row["wer"]["delta"] == pytest.approx(0.06)
    assert row["constraints"]["pass"] is True
    assert row["failed_constraints"] == []

    del treated[0]["speaker_sim"]
    failed = evaluation._matched_row_diagnostics(control, treated, cfg)[0]
    assert failed["constraints"]["pass"] is False
    assert "speaker_scored" in failed["failed_constraints"]
    assert "speaker_abs" in failed["failed_constraints"]


def test_per_pair_alpha_selection_reports_coverage_and_improvement_separately():
    def row(pair_id, angry_delta, passed):
        return {
            "pair_id": pair_id,
            "base_id": f"base:{pair_id}",
            "constraints": {"pass": passed},
            "angry_prob": {"delta": angry_delta, "treated": 0.5},
            "speaker_sim": {"treated": 0.9, "degradation": 0.01},
            "wer": {"delta": 0.02},
        }

    common = {
        "artifacts": ["checkpoints/transform-00005.pt"],
        "completed_updates": 5,
        "transform_digest": "digest",
    }
    candidates = [{
        **common,
        "alpha": 0.5,
        "rows": [
            row("pair:1", 0.1, True),
            row("pair:2", 0.4, False),
            row("pair:3", 0.8, False),
        ],
    }, {
        **common,
        "alpha": 1.0,
        "rows": [
            row("pair:1", 0.3, True),
            row("pair:2", -0.1, True),
            row("pair:3", 0.9, False),
        ],
    }]

    summary = evaluation._per_pair_alpha_selection(candidates)[0]

    assert summary["total_pairs"] == 3
    assert summary["covered_count"] == 2
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["improved_count"] == 1
    assert summary["improvement_rate"] == pytest.approx(1 / 3)
    assert summary["all_pairs_covered"] is False
    assert summary["all_pairs_improved"] is False
    assert summary["uncovered_pairs"] == [{
        "pair_id": "pair:3", "base_id": "base:pair:3"
    }]
    assert summary["not_improved_pairs"] == [{
        "pair_id": "pair:2", "base_id": "base:pair:2"
    }]
    assert summary["pair_selections"][0]["alpha"] == 1.0


def test_checkpoint_selection_excludes_higher_emotion_candidate_failing_wer():
    base = {
        "artifacts": ["checkpoint.pt"],
        "completed_updates": 5,
        "transform_digest": "a",
        "alpha": 1.0,
    }
    feasible = {
        **base,
        "metrics": {
            "emotion_delta_mean": 0.2,
            "emotion_delta_ci": [0.1, 0.3],
            "speaker_sim": 0.9,
            "wer": {"delta": 0.06},
            "pass": True,
        },
    }
    failed = {
        **base,
        "completed_updates": 10,
        "transform_digest": "b",
        "metrics": {
            "emotion_delta_mean": 0.5,
            "emotion_delta_ci": [0.4, 0.6],
            "speaker_sim": 0.9,
            "wer": {"delta": 0.07},
            "pass": False,
        },
    }

    selection = evaluation._select_checkpoint_candidate([feasible, failed])

    assert selection["best"]["transform_digest"] == "a"
    assert selection["best_observed"]["transform_digest"] == "b"
