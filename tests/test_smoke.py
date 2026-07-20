from ste_optimized.config import ExperimentConfig
from ste_optimized.smoke import _anger_success, _arm_constraints, _checkpoint_due


def _arm(probs):
    return {
        "rows": [
            {"base_id": f"b{i}", "angry_prob": probability}
            for i, probability in enumerate(probs)
        ],
        "termination_rate": 1.0,
        "mean_speaker_sim": 0.95,
    }


def test_mean_gain_cannot_hide_three_row_regressions():
    cfg = ExperimentConfig()
    unsteered = _arm([0.10, 0.10, 0.10, 0.10])
    learned = _arm([0.90, 0.09, 0.09, 0.09])

    constraints = _arm_constraints(learned, unsteered, cfg)

    assert constraints["rows_angrier_than_unsteered"] == 1
    assert not constraints["row_directionality"]
    assert not constraints["pass"]


def test_checkpoint_cadence_always_includes_final_update():
    assert not _checkpoint_due(1, max_updates=5, eval_every=5)
    assert _checkpoint_due(5, max_updates=5, eval_every=5)
    assert _checkpoint_due(6, max_updates=10, eval_every=3)


def test_angry_success_requires_mean_threshold_and_unsteered_gain():
    unsteered = {"mean_angry_prob": 0.01}
    assert not _anger_success({"mean_angry_prob": 0.49}, unsteered)
    assert _anger_success({"mean_angry_prob": 0.50}, unsteered)
