"""Unit tests for butchc._utils."""

import math

import pytest

from butchc._utils import (
    MIN_BANDWIDTH_FRACTION,
    compute_trial_loss,
    effective_sample_size,
    is_finite_number,
    reflect,
    silverman_bandwidth,
    softmax,
    weighted_mean,
)


class TestSoftmax:
    def test_sums_to_one(self):
        assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_all_probabilities_positive(self):
        assert all(p > 0 for p in softmax([-50.0, 0.0, 50.0]))

    def test_uniform_logits_give_uniform_output(self):
        assert softmax([2.0] * 4) == pytest.approx([0.25] * 4)

    def test_identity_on_log_probs_at_temp_one(self):
        probs = [0.5, 0.3, 0.2]
        out = softmax([math.log(p) for p in probs], temp=1.0)
        assert out == pytest.approx(probs)

    def test_high_temp_flattens(self):
        sharp = softmax([0.0, 5.0], temp=1.0)
        flat = softmax([0.0, 5.0], temp=10.0)
        assert flat[1] - flat[0] < sharp[1] - sharp[0]

    def test_low_temp_sharpens(self):
        sharp = softmax([0.0, 1.0], temp=0.1)
        assert sharp[1] > 0.99

    def test_no_overflow_on_large_logits(self):
        assert sum(softmax([1e5, 1e5 + 1])) == pytest.approx(1.0)

    def test_rejects_non_positive_temp(self):
        with pytest.raises(ValueError):
            softmax([1.0, 2.0], temp=0.0)
        with pytest.raises(ValueError):
            softmax([1.0, 2.0], temp=-1.0)

    def test_rejects_empty_logits(self):
        with pytest.raises(ValueError):
            softmax([])


class TestWeightedMean:
    def test_uniform_weights_give_arithmetic_mean(self):
        assert weighted_mean([1.0, 2.0, 3.0], [1 / 3] * 3) == pytest.approx(2.0)

    def test_concentrated_weight_returns_that_point(self):
        assert weighted_mean([1.0, 9.0], [0.0, 1.0]) == pytest.approx(9.0)


class TestEffectiveSampleSize:
    def test_uniform_weights_give_full_n(self):
        assert effective_sample_size([0.1] * 10) == pytest.approx(10.0)

    def test_single_dominant_weight_gives_one(self):
        assert effective_sample_size([1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_never_exceeds_n(self):
        assert effective_sample_size([0.4, 0.3, 0.2, 0.1]) <= 4.0


class TestSilvermanBandwidth:
    def test_positive(self):
        assert silverman_bandwidth([0.0, 1.0, 2.0], [1 / 3] * 3, 2.0) > 0

    def test_floored_at_fraction_of_span(self):
        # Identical points have zero variance; bandwidth must not collapse.
        h = silverman_bandwidth([1.0] * 5, [0.2] * 5, span=10.0)
        assert h == pytest.approx(MIN_BANDWIDTH_FRACTION * 10.0)

    def test_never_exceeds_span(self):
        assert silverman_bandwidth([-100.0, 100.0], [0.5, 0.5], span=1.0) <= 1.0

    def test_grows_with_spread(self):
        tight = silverman_bandwidth([0.9, 1.0, 1.1], [1 / 3] * 3, 10.0)
        wide = silverman_bandwidth([-5.0, 0.0, 5.0], [1 / 3] * 3, 10.0)
        assert wide > tight

    def test_single_point_returns_floor(self):
        # Derived from the constant, not written out: a literal here pins the
        # test to whatever the floor happened to be when it was written, so
        # retuning the default fails a test that is not about the default.
        assert silverman_bandwidth([1.0], [1.0], span=4.0) == pytest.approx(
            MIN_BANDWIDTH_FRACTION * 4.0
        )


class TestReflect:
    def test_value_inside_is_unchanged(self):
        assert reflect(0.5, 0.0, 1.0) == pytest.approx(0.5)

    def test_overshoot_folds_back(self):
        assert reflect(1.2, 0.0, 1.0) == pytest.approx(0.8)

    def test_undershoot_folds_back(self):
        assert reflect(-0.3, 0.0, 1.0) == pytest.approx(0.3)

    def test_far_overshoot_stays_in_range(self):
        for x in (-37.4, 91.2, 1e3):
            assert 0.0 <= reflect(x, 0.0, 1.0) <= 1.0

    def test_no_mass_pile_up_at_bounds(self):
        # Clipping would map every overshoot onto the bound exactly;
        # reflection must not.
        assert reflect(1.5, 0.0, 1.0) != 1.0
        assert reflect(-0.5, 0.0, 1.0) != 0.0

    def test_degenerate_interval(self):
        assert reflect(5.0, 2.0, 2.0) == 2.0


class TestComputeTrialLoss:
    def test_empty_gives_zero(self):
        assert compute_trial_loss([]) == 0.0

    def test_mean_of_deltas(self):
        assert compute_trial_loss([0.1, 0.3]) == pytest.approx(0.2)


class TestIsFiniteNumber:
    @pytest.mark.parametrize("value", [0, 1.5, -3, 1e300])
    def test_accepts_finite_numbers(self, value):
        assert is_finite_number(value)

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf"), "1.0", None, True]
    )
    def test_rejects_everything_else(self, value):
        assert not is_finite_number(value)
