"""CPU tests: sampler balance/rotation/resume and leakage fail-closed."""

import pytest
import torch

from ste_optimized.data import BaseRecord, PairRecord, _leakage_checks, speaker_split
from ste_optimized.sampling import BatchSampler


def _contrasts(n_per_speaker=6, speakers=("0011", "0014", "0017", "0020")):
    rows = []
    for spk in speakers:
        for i in range(n_per_speaker):
            rows.append({"pair_id": f"{spk}:{i:06d}", "v": torch.randn(16),
                         "emotion_prob": 0.9, "speaker_sim": 0.95})
    return rows


def _bases(speakers=("0011", "0014", "0017", "0020"), per=4):
    out = []
    for spk in speakers:
        for i in range(per):
            out.append(BaseRecord(
                base_id=f"{spk}:b{i}", speaker=spk, index=f"b{i}",
                target_text=f"text {spk} {i}", reference_index=f"r{i}",
                reference_text=f"ref {spk} {i}",
                reference_audio=f"/audio/{spk}_{i}.wav", split="train"))
    return out


def _sampler(K=4, M=2, seed=42, contrasts=None):
    return BatchSampler(contrasts or _contrasts(), _bases(), K, M, seed,
                        min_emotion_prob=0.2, min_speaker_sim=0.85)


def test_batch_shape_and_cross_speaker_constraint():
    s = _sampler()
    b = s.next_batch()
    assert len(b.contrast_ids) == 4
    assert len(b.rows) == 8 and len(b.row_contrast) == 8
    for row, slot in zip(b.rows, b.row_contrast):
        contrast_speaker = b.contrast_ids[slot].split(":")[0]
        assert row["reference_speaker"] != contrast_speaker


def test_epoch_covers_every_contrast_once():
    s = _sampler()
    seen = []
    for _ in range(s.updates_per_epoch()):
        seen += s.next_batch().contrast_ids
    assert len(seen) == len(set(seen)) == 24


def test_base_rotation_across_epochs():
    s = _sampler(K=24)  # one update = one epoch
    first = s.next_batch()
    second = s.next_batch()
    assert second.epoch == 1
    # same contrast order key: compare base assignment for one contrast
    cid = first.contrast_ids[0]
    slot2 = second.contrast_ids.index(cid)
    bases1 = [r["base_id"] for r, sl in zip(first.rows, first.row_contrast)
              if first.contrast_ids[sl] == cid]
    bases2 = [r["base_id"] for r, sl in zip(second.rows, second.row_contrast)
              if second.contrast_ids[sl] == cid]
    assert bases1 != bases2  # rotated per epoch (seeded by epoch)


def test_resume_reproduces_next_batch():
    a = _sampler()
    a.next_batch()
    state = a.state_dict()
    expected = a.next_batch()
    b = _sampler()
    b.load_state_dict(state)
    got = b.next_batch()
    assert got.contrast_ids == expected.contrast_ids
    assert [r["base_id"] for r in got.rows] == [r["base_id"] for r in expected.rows]


def test_epoch_tail_is_carried_into_full_distinct_batch_and_resumes():
    # Ten contrasts at K=4 would historically emit a two-contrast tail.  The
    # third update must now carry two items from epoch 1 and remain full.
    contrasts = _contrasts(n_per_speaker=5, speakers=("0011", "0014"))
    a = _sampler(K=4, M=2, contrasts=contrasts)
    first = a.next_batch()
    second = a.next_batch()
    boundary = a.next_batch()

    assert len(first.contrast_ids) == len(second.contrast_ids) == 4
    assert len(boundary.contrast_ids) == 4
    assert len(boundary.rows) == 8
    assert len(set(boundary.contrast_ids)) == 4
    assert boundary.epoch == 0
    assert a.epoch == 1
    assert a.cursor == 2

    state = a.state_dict()
    expected = a.next_batch()
    b = _sampler(K=4, M=2, contrasts=contrasts)
    b.load_state_dict(state)
    got = b.next_batch()
    assert got.contrast_ids == expected.contrast_ids
    assert [row["base_id"] for row in got.rows] == [
        row["base_id"] for row in expected.rows
    ]


def test_sampler_rejects_pool_smaller_than_full_contrast_batch():
    with pytest.raises(ValueError, match="need K=4"):
        _sampler(
            K=4,
            contrasts=_contrasts(n_per_speaker=1, speakers=("0011", "0014")),
        )


def test_quality_filter_excludes_bad_pairs():
    rows = _contrasts()
    rows[0]["emotion_prob"] = 0.1
    rows[1]["speaker_sim"] = 0.5
    s = _sampler(contrasts=rows)
    assert s.excluded == 2
    all_ids = set()
    for _ in range(s.updates_per_epoch()):
        all_ids |= set(s.next_batch().contrast_ids)
    assert rows[0]["pair_id"] not in all_ids
    assert rows[1]["pair_id"] not in all_ids


def test_all_pass_filter_raises():
    rows = [{"pair_id": "0011:0", "v": torch.randn(16),
             "emotion_prob": 0.05, "speaker_sim": 0.1}]
    with pytest.raises(ValueError):
        _sampler(contrasts=rows)


def test_speaker_split_and_leakage_fail_closed():
    assert speaker_split("0011") == "train"
    assert speaker_split("0016") == "validation"
    with pytest.raises(ValueError):
        speaker_split("9999")
    bad_pair = PairRecord(pair_id="0011:1", speaker="0011", index="1",
                          text="t", split="validation", neutral_audio="a",
                          emotional_audio="b", emotion="angry")
    with pytest.raises(ValueError):
        _leakage_checks([bad_pair], [])
    bad_base = BaseRecord(base_id="0013:b0", speaker="0013", index="b0",
                          target_text="t", reference_index="r",
                          reference_text="rt", reference_audio="a",
                          split="train")
    with pytest.raises(ValueError):
        _leakage_checks([], [bad_base])
