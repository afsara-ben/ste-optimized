import math

import pytest

from ste_optimized.wer_metrics import (
    corpus_wer,
    matched_control_wer,
    normalize_english,
    wer_delta_passes,
    word_error_counts,
)


def test_whisper_english_normalization() -> None:
    assert normalize_english(" Um, [noise] I CAN'T, go! ") == "i can not go"
    assert normalize_english(None) == ""


def test_word_error_counts_include_substitution_deletion_and_insertion() -> None:
    counts = word_error_counts(
        "alpha beta gamma delta epsilon zeta",
        "alpha bravo gamma epsilon zeta yankee",
    )
    assert counts.substitutions == 1
    assert counts.deletions == 1
    assert counts.insertions == 1
    assert counts.reference_words == 6
    assert counts.errors == 3
    assert counts.wer == pytest.approx(0.5)


def test_corpus_wer_aggregates_counts_not_utterance_rates() -> None:
    # Row WERs are 1/2 and 2/3; their unweighted mean is 7/12.  Corpus WER is
    # instead (D + S + I) / N = 3/5.
    result = corpus_wer(["a b", "c d e"], ["a", "c x e z"])
    assert result.substitutions == 1
    assert result.deletions == 1
    assert result.insertions == 1
    assert result.reference_words == 5
    assert result.wer == pytest.approx(0.6)
    assert result.wer != pytest.approx((0.5 + 2 / 3) / 2)


@pytest.mark.parametrize("empty_hypothesis", ["", "   ", None, "um"])
def test_empty_hypothesis_is_all_deletions(empty_hypothesis) -> None:
    counts = corpus_wer(["keep all three"], [empty_hypothesis])
    assert counts.substitutions == 0
    assert counts.deletions == 3
    assert counts.insertions == 0
    assert counts.wer == 1.0


def test_empty_reference_rows_contribute_insertions_to_nonempty_corpus() -> None:
    result = corpus_wer(["", "alpha beta"], ["extra words", "alpha beta"])
    assert result.insertions == 2
    assert result.reference_words == 2
    assert result.wer == 1.0


def test_minimum_reference_word_validation() -> None:
    with pytest.raises(ValueError, match="require at least 3"):
        corpus_wer(["only two"], ["only two"], min_reference_words=3)
    with pytest.raises(ValueError, match="positive integer"):
        corpus_wer(["one"], ["one"], min_reference_words=0)
    with pytest.raises(ValueError, match="0 reference words"):
        corpus_wer(["um"], [""], min_reference_words=1)


def test_input_cardinality_is_validated() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        corpus_wer(["one"], [])
    with pytest.raises(ValueError, match="treated/control length mismatch"):
        matched_control_wer(["one"], ["one"], [])


def test_matched_control_delta_equality_passes_and_larger_delta_fails() -> None:
    words = ["token"] * 100
    reference = " ".join(words)
    control = reference

    equal_boundary = matched_control_wer(
        [reference], [" ".join(words[6:])], [control]
    )
    assert equal_boundary.control.wer == 0.0
    assert equal_boundary.treated.deletions == 6
    assert equal_boundary.delta == pytest.approx(0.06)
    assert equal_boundary.passed is True

    over_boundary = matched_control_wer(
        [reference], [" ".join(words[7:])], [control]
    )
    assert over_boundary.delta == pytest.approx(0.07)
    assert over_boundary.passed is False


def test_delta_gate_is_strict_except_for_inclusive_boundary() -> None:
    assert wer_delta_passes(0.16, 0.10) is True
    assert wer_delta_passes(0.1600001, 0.10) is False
    with pytest.raises(ValueError, match="finite"):
        wer_delta_passes(math.nan, 0.1)
