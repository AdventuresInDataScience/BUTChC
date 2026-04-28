"""Tests for butchc._utils: softmax, _silverman_bandwidth, _compute_trial_loss."""
import math
import pytest

from butchc._utils import (
    KDE_RESERVOIR_SIZE,
    _compute_trial_loss,
    _silverman_bandwidth,
    softmax,
)


class TestKDEReservoirSize:
    def test_is_positive_integer(self):
        assert isinstance(KDE_RESERVOIR_SIZE, int)
        assert KDE_RESERVOIR_SIZE > 0

    def test_value_is_fifty(self):
        assert KDE_RESERVOIR_SIZE == 50


class TestSoftmax:
    def test_sums_to_one(self):
        result = softmax([1.0, 2.0, 3.0])
        assert abs(sum(result) - 1.0) < 1e-9

    def test_length_preserved(self):
        logits = [0.5, 1.5, 2.5, 3.5]
        assert len(softmax(logits)) == len(logits)

    def test_all_probabilities_non_negative(self):
        result = softmax([1.0, 2.0, 3.0])
        assert all(p >= 0.0 for p in result)

    def test_all_probabilities_at_most_one(self):
        result = softmax([1.0, 2.0, 3.0])
        assert all(p <= 1.0 for p in result)

    def test_higher_logit_gets_higher_prob(self):
        result = softmax([1.0, 3.0])
        assert result[1] > result[0]

    def test_single_element_returns_one(self):
        result = softmax([42.0])
        assert abs(result[0] - 1.0) < 1e-9

    def test_equal_logits_give_uniform_distribution(self):
        result = softmax([5.0, 5.0, 5.0])
        for p in result:
            assert abs(p - 1 / 3) < 1e-9

    def test_high_temp_flattens_toward_uniform(self):
        logits = [1.0, 10.0]
        probs = softmax(logits, temp=100.0)
        assert abs(probs[0] - 0.5) < 0.05

    def test_low_temp_concentrates_on_maximum(self):
        logits = [1.0, 10.0]
        probs = softmax(logits, temp=0.01)
        assert probs[1] > 0.999

    def test_numerical_stability_with_large_positive_logits(self):
        result = softmax([1000.0, 1001.0])
        assert abs(sum(result) - 1.0) < 1e-9
        assert all(0.0 <= p <= 1.0 for p in result)

    def test_numerical_stability_with_large_negative_logits(self):
        result = softmax([-1000.0, -999.0])
        assert abs(sum(result) - 1.0) < 1e-9
        assert all(0.0 <= p <= 1.0 for p in result)

    def test_default_temp_matches_temp_one(self):
        logits = [1.0, 2.0, 3.0]
        assert softmax(logits) == softmax(logits, temp=1.0)

    def test_five_element_vector(self):
        logits = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = softmax(logits)
        assert abs(sum(result) - 1.0) < 1e-9
        assert len(result) == 5
        # Probabilities should be strictly increasing
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]


class TestSilvermanBandwidth:
    def test_single_point_returns_default(self):
        assert _silverman_bandwidth([5.0], [1.0]) == 1.0

    def test_empty_reservoir_returns_default(self):
        # Fewer than 2 points triggers default path
        assert _silverman_bandwidth([0.5], [1.0]) == 1.0

    def test_returns_positive_value(self):
        reservoir = [0.1, 0.5, 0.9]
        weights   = [1 / 3, 1 / 3, 1 / 3]
        assert _silverman_bandwidth(reservoir, weights) > 0

    def test_wider_spread_gives_larger_bandwidth(self):
        narrow = _silverman_bandwidth([0.4, 0.5, 0.6], [1 / 3, 1 / 3, 1 / 3])
        wide   = _silverman_bandwidth([0.0, 0.5, 1.0], [1 / 3, 1 / 3, 1 / 3])
        assert wide > narrow

    def test_identical_points_does_not_crash(self):
        # Zero variance: w_var = 0 → sqrt(1e-12) guard applied
        result = _silverman_bandwidth([0.5, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3])
        assert result > 0

    def test_heavily_skewed_weights_does_not_crash(self):
        reservoir = [0.1, 0.5, 0.9]
        weights   = [0.98, 0.01, 0.01]
        result = _silverman_bandwidth(reservoir, weights)
        assert result > 0

    def test_larger_reservoir_gives_smaller_bandwidth_same_spread(self):
        # More effective samples → Silverman rule gives smaller bandwidth
        small = _silverman_bandwidth([0.0, 1.0], [0.5, 0.5])
        large = _silverman_bandwidth(
            [i / 9 for i in range(10)],
            [0.1] * 10,
        )
        assert large < small

    def test_returns_float(self):
        result = _silverman_bandwidth([0.2, 0.8], [0.5, 0.5])
        assert isinstance(result, float)


class TestComputeTrialLoss:
    def test_empty_deltas_returns_zero(self):
        assert _compute_trial_loss([]) == 0.0

    def test_single_delta_returned_unchanged(self):
        assert _compute_trial_loss([0.5]) == 0.5

    def test_mean_of_three_deltas(self):
        assert abs(_compute_trial_loss([0.2, 0.4, 0.6]) - 0.4) < 1e-9

    def test_all_zero_deltas(self):
        assert _compute_trial_loss([0.0, 0.0, 0.0]) == 0.0

    def test_all_one_deltas(self):
        assert abs(_compute_trial_loss([1.0, 1.0, 1.0]) - 1.0) < 1e-9

    def test_single_zero(self):
        assert _compute_trial_loss([0.0]) == 0.0

    def test_returns_float(self):
        result = _compute_trial_loss([0.3, 0.7])
        assert isinstance(result, float)
