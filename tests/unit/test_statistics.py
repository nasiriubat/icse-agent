"""Tests for the paired significance machinery.

The McNemar cases are checked against values computed by hand from the binomial distribution
rather than against whatever this implementation happens to return, because a significance test
that agrees only with itself is worse than no test at all.
"""

from __future__ import annotations

import math

import pytest

from memcheck.utilities.statistics import (
    EXACT_MCNEMAR_THRESHOLD,
    MAX_EXACT_SIGN_FLIP_STREAMS,
    holm_bonferroni,
    mcnemar,
    paired_difference_interval,
    stream_level_test,
    wilson_interval,
)


class TestStreamLevelTest:
    def test_identical_streams_are_no_difference(self) -> None:
        result = stream_level_test([0.0] * 10)
        assert result.p_value == 1.0
        assert result.mean == 0.0
        assert result.ci_low == result.ci_high == 0.0

    def test_every_stream_favouring_one_side_reaches_the_floor(self) -> None:
        """Ten streams all on one side: only the all-plus and all-minus assignments are as
        extreme, so the exact two-sided p is 2 / 2^10, the smallest the design can give."""
        result = stream_level_test([0.02, 0.05, 0.03, 0.04, 0.06, 0.02, 0.03, 0.05, 0.04, 0.03])
        assert result.p_value == pytest.approx(2 / 1024)
        assert result.n_streams == 10
        assert result.favouring_reference == 10

    def test_the_sign_of_the_effect_does_not_change_the_p_value(self) -> None:
        up = stream_level_test([0.1, 0.2, 0.3])
        down = stream_level_test([-0.1, -0.2, -0.3])
        assert up.p_value == down.p_value == pytest.approx(2 / 8)
        assert down.favouring_reference == 0

    def test_the_interval_brackets_the_mean_and_is_reproducible(self) -> None:
        values = [0.01, -0.02, 0.05, 0.03, 0.00, 0.04, -0.01, 0.02, 0.03, 0.06]
        first = stream_level_test(values)
        second = stream_level_test(values)
        assert first.ci_low <= first.mean <= first.ci_high
        assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)
        assert first.minimum == -0.02 and first.maximum == 0.06

    def test_a_single_stream_cannot_be_significant(self) -> None:
        assert stream_level_test([0.3]).p_value == 1.0

    def test_too_many_streams_for_exact_enumeration_are_refused(self) -> None:
        with pytest.raises(ValueError, match="too many streams"):
            stream_level_test([0.1] * (MAX_EXACT_SIGN_FLIP_STREAMS + 1))

    def test_an_empty_sample_yields_no_verdict(self) -> None:
        result = stream_level_test([])
        assert result.n_streams == 0 and math.isnan(result.p_value)


def _paired(b_count: int, c_count: int, concordant: int = 0) -> tuple[list[bool], list[bool]]:
    """Build a paired sample with exactly the requested discordant counts."""
    a = [True] * b_count + [False] * c_count + [True] * concordant
    b = [False] * b_count + [True] * c_count + [True] * concordant
    return a, b


class TestMcNemar:
    def test_identical_vectors_are_not_a_difference(self) -> None:
        outcomes = [True, False, True, True, False]
        result = mcnemar(outcomes, outcomes)
        assert result.p_value == 1.0
        assert result.effect_size == 0.0
        assert "exactly the same queries" in result.note

    def test_concordant_disagreement_is_still_no_difference(self) -> None:
        """Both wrong on the same queries is agreement, not evidence of a gap."""
        a = [True, True, False, False]
        b = [True, True, False, False]
        assert mcnemar(a, b).p_value == 1.0

    def test_exact_binomial_matches_a_hand_computed_value(self) -> None:
        """b=8, c=1 -> two-sided exact p = 2 * P(X <= 1 | n=9, p=0.5) = 2 * 10/512."""
        a, b = _paired(8, 1)
        result = mcnemar(a, b)
        assert result.p_value == pytest.approx(2 * 10 / 512)
        assert "exact binomial" in result.note
        # Cohen's g: 8/9 - 0.5.
        assert result.effect_size == pytest.approx(8 / 9 - 0.5)
        assert result.effect_label == "large"

    def test_a_perfectly_split_disagreement_is_maximally_insignificant(self) -> None:
        a, b = _paired(6, 6)
        result = mcnemar(a, b)
        assert result.p_value == 1.0
        assert result.effect_size == pytest.approx(0.0)

    def test_the_test_is_directional_in_its_effect_but_not_its_p_value(self) -> None:
        """Swapping the arguments flips the sign of g and leaves p alone."""
        a, b = _paired(9, 2)
        forward, reverse = mcnemar(a, b), mcnemar(b, a)
        assert forward.p_value == pytest.approx(reverse.p_value)
        assert forward.effect_size == pytest.approx(-reverse.effect_size)

    def test_large_samples_switch_to_the_corrected_chi_square(self) -> None:
        a, b = _paired(30, 10)
        result = mcnemar(a, b)
        assert "chi-square" in result.note
        # chi2 = (|30-10| - 1)^2 / 40 = 361/40 = 9.025 -> z = 3.0042, p ~= 0.00266.
        assert result.p_value == pytest.approx(0.002663, abs=1e-5)

    def test_the_exact_form_is_used_right_up_to_the_threshold(self) -> None:
        at_limit, _ = _paired(EXACT_MCNEMAR_THRESHOLD, 0)
        _, other = _paired(EXACT_MCNEMAR_THRESHOLD, 0)
        assert "exact binomial" in mcnemar(at_limit, other).note
        beyond_a, beyond_b = _paired(EXACT_MCNEMAR_THRESHOLD + 1, 0)
        assert "chi-square" in mcnemar(beyond_a, beyond_b).note

    def test_thin_evidence_is_flagged_rather_than_reported_silently(self) -> None:
        a, b = _paired(3, 0)
        result = mcnemar(a, b)
        assert any("little power" in warning for warning in result.warnings)

    def test_only_discordant_pairs_carry_information(self) -> None:
        """Padding with agreements changes n but must not change the p-value."""
        thin_a, thin_b = _paired(7, 1)
        padded_a, padded_b = _paired(7, 1, concordant=500)
        assert mcnemar(thin_a, thin_b).p_value == pytest.approx(mcnemar(padded_a, padded_b).p_value)
        assert mcnemar(padded_a, padded_b).n == 508

    def test_unequal_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="paired samples"):
            mcnemar([True, False], [True])

    def test_an_empty_sample_yields_no_verdict(self) -> None:
        result = mcnemar([], [])
        assert math.isnan(result.p_value)
        assert result.effect_label == "undefined"


class TestHolmBonferroni:
    def test_adjustment_is_monotone_and_order_preserving(self) -> None:
        raw = [0.01, 0.04, 0.03]
        adjusted = holm_bonferroni(raw)
        assert adjusted[0] == pytest.approx(0.03)
        # Holm enforces monotonicity: the 0.03 is lifted to the 0.04's level.
        assert adjusted[2] == pytest.approx(0.06)
        assert adjusted[1] == pytest.approx(0.06)

    def test_adjusted_values_never_shrink(self) -> None:
        raw = [0.001, 0.2, 0.5, 0.9]
        assert all(
            adjusted >= original
            for adjusted, original in zip(holm_bonferroni(raw), raw, strict=True)
        )


class TestWilsonInterval:
    def test_a_saturated_proportion_still_has_an_interval(self) -> None:
        """The normal approximation collapses at p=1; Wilson is used precisely because it does not."""
        low, high = wilson_interval(20, 20)
        assert 0.0 < low < 1.0
        assert high == pytest.approx(1.0)


class TestPairedDifferenceInterval:
    def test_identical_outcomes_give_a_zero_difference_inside_its_interval(self) -> None:
        outcomes = [True] * 30 + [False] * 10
        diff, low, high = paired_difference_interval(outcomes, outcomes)
        assert diff == 0.0
        assert low < 0.0 < high

    def test_interval_brackets_the_difference_and_is_antisymmetric(self) -> None:
        a = [True] * 40 + [False] * 10 + [True] * 8 + [False] * 2
        b = [True] * 40 + [False] * 10 + [False] * 8 + [True] * 2
        diff, low, high = paired_difference_interval(a, b)
        assert diff == pytest.approx(6 / 60)
        assert low < diff < high
        back, back_low, back_high = paired_difference_interval(b, a)
        assert back == pytest.approx(-diff)
        assert back_low == pytest.approx(-high)
        assert back_high == pytest.approx(-low)

    def test_strong_pairing_narrows_the_interval(self) -> None:
        # The same marginals, but concordant pairs carry the correlation the method uses.
        paired_a = [True] * 50 + [False] * 50
        paired_b = [True] * 45 + [False] * 55
        unpaired_b = [True] * 22 + [False] * 28 + [True] * 23 + [False] * 27
        _, low_paired, high_paired = paired_difference_interval(paired_a, paired_b)
        _, low_unpaired, high_unpaired = paired_difference_interval(paired_a, unpaired_b)
        assert high_paired - low_paired < high_unpaired - low_unpaired


class TestMcNemarLargeSample:
    def test_balanced_discordance_gives_p_of_one(self) -> None:
        a = [True] * 20 + [False] * 20
        b = [False] * 20 + [True] * 20
        assert mcnemar(a, b).p_value == pytest.approx(1.0)

    def test_tiny_p_values_do_not_underflow_to_zero(self) -> None:
        a = [True] * 200 + [False] * 5
        b = [False] * 200 + [True] * 5
        assert 0.0 < mcnemar(a, b).p_value < 1e-30
