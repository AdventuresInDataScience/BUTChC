"""Unit tests for butchc._sampling."""

import random

import pytest

from butchc._sampling import sample_node, traverse_sample
from butchc._tree import initialize_prob_tree


@pytest.fixture
def rng():
    return random.Random(0)


class TestSampleNodeCategorical:
    def test_returns_a_listed_value(self, rng):
        node = initialize_prob_tree({"a": {"values": ["x", "y", "z"]}})["a"]
        for _ in range(50):
            external, internal = sample_node(node, 1.0, rng)
            assert external in {"x", "y", "z"}
            assert external == internal

    def test_respects_learned_probabilities(self, rng):
        node = initialize_prob_tree({"a": {"values": ["x", "y"]}})["a"]
        node["prob"] = {"x": 0.99, "y": 0.01}
        draws = [sample_node(node, 1.0, rng)[0] for _ in range(500)]
        assert draws.count("x") > 400

    def test_low_temperature_sharpens(self, rng):
        node = initialize_prob_tree({"a": {"values": ["x", "y"]}})["a"]
        node["prob"] = {"x": 0.7, "y": 0.3}
        greedy = [sample_node(node, 0.05, rng)[0] for _ in range(300)]
        assert greedy.count("x") > 290

    def test_high_temperature_flattens(self, rng):
        node = initialize_prob_tree({"a": {"values": ["x", "y"]}})["a"]
        node["prob"] = {"x": 0.95, "y": 0.05}
        draws = [sample_node(node, 20.0, rng)[0] for _ in range(400)]
        assert draws.count("y") > 100


class TestSampleNodeContinuous:
    def test_stays_within_bounds(self, rng):
        node = initialize_prob_tree({"x": {"min": -2.0, "max": 3.0}})["x"]
        for _ in range(500):
            external, _ = sample_node(node, 1.0, rng)
            assert -2.0 <= external <= 3.0

    def test_no_probability_mass_piles_on_bounds(self, rng):
        # Clipping jitter would deposit atoms at min and max. Reflection
        # must not: exact hits on either bound should be vanishingly rare.
        node = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})["x"]
        draws = [sample_node(node, 1.0, rng)[0] for _ in range(3000)]
        on_bound = sum(1 for v in draws if v in (0.0, 1.0))
        assert on_bound / len(draws) < 0.01

    def test_concentrated_reservoir_produces_local_samples(self, rng):
        node = initialize_prob_tree({"x": {"min": -10.0, "max": 10.0}})["x"]
        node["reservoir"] = [4.0] * 10
        node["weights"] = [0.1] * 10
        node["scores"] = [1.0] * 10
        draws = [sample_node(node, 1.0, rng)[0] for _ in range(300)]
        assert all(abs(v - 4.0) < 2.0 for v in draws)

    def test_log_node_returns_external_scale(self, rng):
        node = initialize_prob_tree({"lr": {"min": 1e-5, "max": 1e-1, "log": True}})["lr"]
        for _ in range(200):
            external, internal = sample_node(node, 1.0, rng)
            assert 1e-5 <= external <= 1e-1
            assert -5.0 <= internal <= -1.0

    def test_int_node_returns_integers(self, rng):
        node = initialize_prob_tree({"n": {"min": 1.0, "max": 6.0, "int": True}})["n"]
        for _ in range(200):
            external, _ = sample_node(node, 1.0, rng)
            assert isinstance(external, int)
            assert 1 <= external <= 6

    def test_int_internal_matches_emitted(self, rng):
        node = initialize_prob_tree({"n": {"min": 1.0, "max": 6.0, "int": True}})["n"]
        for _ in range(100):
            external, internal = sample_node(node, 1.0, rng)
            assert internal == pytest.approx(float(external))


class TestExploration:
    def test_explore_one_ignores_the_model(self, rng):
        node = initialize_prob_tree({"a": {"values": ["x", "y"]}})["a"]
        node["prob"] = {"x": 0.999, "y": 0.001}
        draws = [sample_node(node, 1.0, rng, explore=1.0)[0] for _ in range(600)]
        assert 200 < draws.count("y") < 400

    def test_explore_one_covers_full_continuous_range(self, rng):
        node = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})["x"]
        node["reservoir"] = [0.5] * 5
        node["weights"] = [0.2] * 5
        node["scores"] = [1.0] * 5
        draws = [sample_node(node, 1.0, rng, explore=1.0)[0] for _ in range(400)]
        assert min(draws) < 0.1 and max(draws) > 0.9

    def test_explore_zero_uses_the_model(self, rng):
        node = initialize_prob_tree({"a": {"values": ["x", "y"]}})["a"]
        node["prob"] = {"x": 0.999, "y": 0.001}
        draws = [sample_node(node, 1.0, rng, explore=0.0)[0] for _ in range(400)]
        assert draws.count("y") < 20


class TestMalformedNode:
    def test_raises_with_helpful_message(self, rng):
        with pytest.raises(ValueError, match="Malformed node"):
            sample_node({"nonsense": 1}, 1.0, rng)


class TestTraverseSample:
    SPACE = {
        "optimizer": {
            "values": ["adam", "sgd"],
            "next_level": {
                "adam": {"beta1": {"min": 0.8, "max": 0.99}},
                "sgd": {"momentum": {"min": 0.0, "max": 0.99}},
            },
        },
        "batch_size": {"values": [16, 32]},
    }

    def test_root_params_always_present(self, rng):
        tree = initialize_prob_tree(self.SPACE)
        for _ in range(50):
            config, _ = traverse_sample(tree, 1.0, rng)
            assert "optimizer" in config and "batch_size" in config

    def test_only_the_chosen_branch_appears(self, rng):
        tree = initialize_prob_tree(self.SPACE)
        for _ in range(100):
            config, _ = traverse_sample(tree, 1.0, rng)
            if config["optimizer"] == "adam":
                assert "beta1" in config and "momentum" not in config
            else:
                assert "momentum" in config and "beta1" not in config

    def test_trace_mirrors_the_chosen_branch(self, rng):
        tree = initialize_prob_tree(self.SPACE)
        for _ in range(50):
            config, trace = traverse_sample(tree, 1.0, rng)
            assert trace["optimizer"]["internal"] == config["optimizer"]
            assert set(trace["optimizer"]["sub"]) == (
                {"beta1"} if config["optimizer"] == "adam" else {"momentum"}
            )

    def test_trace_leaf_has_no_sub(self, rng):
        tree = initialize_prob_tree(self.SPACE)
        _, trace = traverse_sample(tree, 1.0, rng)
        assert trace["batch_size"]["sub"] is None

    def test_trace_stores_internal_coordinates(self, rng):
        tree = initialize_prob_tree({"lr": {"min": 1e-4, "max": 1e-1, "log": True}})
        config, trace = traverse_sample(tree, 1.0, rng)
        assert trace["lr"]["internal"] == pytest.approx(
            __import__("math").log10(config["lr"])
        )

    def test_deep_nesting_resolves(self, rng):
        space = {
            "model": {
                "values": ["nn"],
                "next_level": {
                    "nn": {
                        "arch": {
                            "values": ["cnn"],
                            "next_level": {"cnn": {"k": {"values": [3, 5]}}},
                        }
                    }
                },
            }
        }
        config, _ = traverse_sample(initialize_prob_tree(space), 1.0, rng)
        assert config["model"] == "nn" and config["arch"] == "cnn" and config["k"] in (3, 5)
