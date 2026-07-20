"""Deterministic word-error-rate helpers with no optional metric dependency.

Text is normalized with Whisper's English normalizer before word-level
Levenshtein alignment.  Corpus WER is always computed from aggregate edit
counts, ``(S + D + I) / N``; it is never an average of utterance WERs.

The dynamic-programming alignment lives here instead of depending on jiwer so
evaluation artifacts do not change with an optional package version.  Empty
hypotheses (including ``None`` and text normalized entirely away) deterministically
count as one deletion for every normalized reference word.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


DEFAULT_MAX_WER_DELTA = 0.06

# Whisper's English normalization algorithm is supplied by the mandatory
# transformers dependency.  Its final British-to-American spelling stage is
# intentionally an identity mapping here: this keeps the metric local and
# deterministic instead of making it depend on a tokenizer asset downloaded at
# evaluation time.  All other English Whisper rules (case, punctuation,
# contractions, fillers, numbers, and symbols) are unchanged.
_ENGLISH_NORMALIZER = EnglishTextNormalizer({})


@lru_cache(maxsize=4096)
def _normalize_cached(text: str) -> str:
    return _ENGLISH_NORMALIZER(text).strip()


def normalize_english(text: str | None) -> str:
    """Return Whisper-normalized English text.

    ``None`` is accepted for failed/empty ASR hypotheses and is normalized to
    the empty string.  Other non-string values are rejected rather than being
    silently stringified.
    """

    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"expected text to be str or None, got {type(text).__name__}")
    return _normalize_cached(text)


@dataclass(frozen=True)
class WERMetrics:
    """Word-level edit counts and their corpus WER denominator."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0

    def __post_init__(self) -> None:
        values = (
            self.substitutions,
            self.deletions,
            self.insertions,
            self.reference_words,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("WER counts must be integers")
        if any(value < 0 for value in values):
            raise ValueError("WER counts must be non-negative")

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            raise ValueError("WER is undefined with zero normalized reference words")
        return self.errors / self.reference_words

    def __add__(self, other: object) -> "WERMetrics":
        if not isinstance(other, WERMetrics):
            return NotImplemented
        return WERMetrics(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_words=self.reference_words + other.reference_words,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "reference_words": self.reference_words,
            "wer": self.wer,
        }


def _alignment_key(metrics: WERMetrics) -> tuple[int, int, int, int]:
    """Stable tie-break after minimum edit distance.

    Fewer insertions+deletions is preferred, which represents a replacement as
    a substitution when both have the same edit cost.  The final fields make
    otherwise equivalent alignments deterministic.
    """

    return (
        metrics.errors,
        metrics.insertions + metrics.deletions,
        metrics.deletions,
        metrics.insertions,
    )


def word_error_counts(
    reference: str,
    hypothesis: str | None,
    *,
    normalize: bool = True,
) -> WERMetrics:
    """Compute deterministic S/D/I/N counts for one matched utterance."""

    if not isinstance(reference, str):
        raise TypeError(f"expected reference to be str, got {type(reference).__name__}")
    if hypothesis is not None and not isinstance(hypothesis, str):
        raise TypeError(
            f"expected hypothesis to be str or None, got {type(hypothesis).__name__}"
        )

    if normalize:
        ref_words = normalize_english(reference).split()
        hyp_words = normalize_english(hypothesis).split()
    else:
        ref_words = reference.split()
        hyp_words = (hypothesis or "").split()

    # dp[i][j] aligns the first i reference words with the first j hypothesis
    # words.  Storing counts directly makes aggregate S/D/I provenance explicit.
    dp: list[list[WERMetrics]] = [
        [WERMetrics() for _ in range(len(hyp_words) + 1)]
        for _ in range(len(ref_words) + 1)
    ]
    for i in range(1, len(ref_words) + 1):
        dp[i][0] = WERMetrics(deletions=i, reference_words=i)
    for j in range(1, len(hyp_words) + 1):
        dp[0][j] = WERMetrics(insertions=j)

    for i, ref_word in enumerate(ref_words, start=1):
        for j, hyp_word in enumerate(hyp_words, start=1):
            if ref_word == hyp_word:
                prior = dp[i - 1][j - 1]
                dp[i][j] = WERMetrics(
                    substitutions=prior.substitutions,
                    deletions=prior.deletions,
                    insertions=prior.insertions,
                    reference_words=i,
                )
                continue

            diagonal = dp[i - 1][j - 1]
            deleted = dp[i - 1][j]
            inserted = dp[i][j - 1]
            candidates = (
                WERMetrics(
                    substitutions=diagonal.substitutions + 1,
                    deletions=diagonal.deletions,
                    insertions=diagonal.insertions,
                    reference_words=i,
                ),
                WERMetrics(
                    substitutions=deleted.substitutions,
                    deletions=deleted.deletions + 1,
                    insertions=deleted.insertions,
                    reference_words=i,
                ),
                WERMetrics(
                    substitutions=inserted.substitutions,
                    deletions=inserted.deletions,
                    insertions=inserted.insertions + 1,
                    reference_words=i,
                ),
            )
            dp[i][j] = min(candidates, key=_alignment_key)

    return dp[-1][-1]


def _validate_min_reference_words(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("min_reference_words must be a positive integer")


def corpus_wer(
    references: Iterable[str],
    hypotheses: Iterable[str | None],
    *,
    min_reference_words: int = 1,
) -> WERMetrics:
    """Compute corpus WER from aggregate S/D/I/N over matched utterances."""

    _validate_min_reference_words(min_reference_words)
    refs = list(references)
    hyps = list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError(
            f"reference/hypothesis length mismatch: {len(refs)} != {len(hyps)}"
        )
    if not refs:
        raise ValueError("at least one matched reference/hypothesis is required")

    total = WERMetrics()
    for reference, hypothesis in zip(refs, hyps):
        total = total + word_error_counts(reference, hypothesis)
    if total.reference_words < min_reference_words:
        raise ValueError(
            "normalized corpus has "
            f"{total.reference_words} reference words; require at least "
            f"{min_reference_words}"
        )
    return total


def wer_delta_passes(
    treated_wer: float,
    control_wer: float,
    *,
    max_delta: float = DEFAULT_MAX_WER_DELTA,
) -> bool:
    """Pass exactly when ``treated_wer - control_wer <= max_delta``."""

    values = {
        "treated_wer": treated_wer,
        "control_wer": control_wer,
        "max_delta": max_delta,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if float(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    return float(treated_wer) - float(control_wer) <= float(max_delta)


@dataclass(frozen=True)
class MatchedControlWER:
    """Corpus WER comparison for treated and matched-control hypotheses."""

    treated: WERMetrics
    control: WERMetrics
    max_delta: float = DEFAULT_MAX_WER_DELTA

    @property
    def delta(self) -> float:
        return self.treated.wer - self.control.wer

    @property
    def passed(self) -> bool:
        return wer_delta_passes(
            self.treated.wer, self.control.wer, max_delta=self.max_delta
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "treated": self.treated.as_dict(),
            "control": self.control.as_dict(),
            "delta": self.delta,
            "max_delta": self.max_delta,
            "pass": self.passed,
        }


def matched_control_wer(
    references: Iterable[str],
    treated_hypotheses: Iterable[str | None],
    control_hypotheses: Iterable[str | None],
    *,
    max_delta: float = DEFAULT_MAX_WER_DELTA,
    min_reference_words: int = 1,
) -> MatchedControlWER:
    """Compare treated and control corpus WER on the exact same references."""

    refs = list(references)
    treated = list(treated_hypotheses)
    control = list(control_hypotheses)
    if len(treated) != len(control):
        raise ValueError(
            f"treated/control length mismatch: {len(treated)} != {len(control)}"
        )
    # corpus_wer validates each arm against the same materialized reference list
    # and enforces the minimum normalized reference-word count.
    treated_metrics = corpus_wer(
        refs, treated, min_reference_words=min_reference_words
    )
    control_metrics = corpus_wer(
        refs, control, min_reference_words=min_reference_words
    )
    # Validate the caller's original value before converting it for the frozen
    # result object (notably, do not silently accept bools or numeric strings).
    wer_delta_passes(
        treated_metrics.wer, control_metrics.wer, max_delta=max_delta
    )
    result = MatchedControlWER(
        treated=treated_metrics,
        control=control_metrics,
        max_delta=float(max_delta),
    )
    return result
