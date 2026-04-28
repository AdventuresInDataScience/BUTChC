"""Tests for butchc._sampling: _sample_node, _traverse_sample."""
import random
import pytest

from butchc._tree import _initialize_prob_tree
from butchc._sampling import _sample_node, _traverse_sample


# ---------------------------------------------------------------------------
# Search-space fixtures
# ---------------------------------------------------------------------------

SIMPLE_CAT = {'optimizer': {'values': ['adam', 'sgd', 'rmsprop']}}
SIMPLE_CONT = {'lr': {'min': 0.001, 'max': 0.1}}
NARROW_CONT = {'p': {'min': 0.49, 'max': 0.51}}

HIERARCHICAL = {
    'optimizer': {
        'values': ['adam', 'sgd'],
        'next_level': {
            'adam': {'lr': {'min': 1e-4, 'max': 1e-2}},
            'sgd':  {
                'lr':       {'min': 1e-3, 'max': 1e-1},
                'momentum': {'min': 0.0, 'max': 0.99},
            },
        },
    },
    'batch_size': {'values': [16, 32, 64]},
}


# ---------------------------------------------------------------------------
# _sample_node — categorical
# ---------------------------------------------------------------------------

class TestSampleNodeCategorical:
    def setup_method(self):
        random.seed(0)

    def test_returns_valid_choice(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        val  = _sample_node(tree['optimizer'], temp=1.0)
        assert val in {'adam', 'sgd', 'rmsprop'}

    def test_returns_string(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        val  = _sample_node(tree['optimizer'], temp=1.0)
        assert isinstance(val, str)

    def test_never_returns_invalid_value(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        node = tree['optimizer']
        for _ in range(200):
            assert _sample_node(node, temp=1.0) in {'adam', 'sgd', 'rmsprop'}

    def test_high_temp_samples_all_choices_eventually(self):
        random.seed(0)
        tree = _initialize_prob_tree(SIMPLE_CAT)
        node = tree['optimizer']
        seen = set()
        for _ in range(500):
            seen.add(_sample_node(node, temp=100.0))
        assert seen == {'adam', 'sgd', 'rmsprop'}

    def test_low_temp_concentrates_on_dominant_choice(self):
        random.seed(0)
        tree = _initialize_prob_tree(SIMPLE_CAT)
        node = tree['optimizer']
        # Bias heavily toward 'adam'
        node['prob'] = {'adam': 0.98, 'sgd': 0.01, 'rmsprop': 0.01}
        samples = [_sample_node(node, temp=0.01) for _ in range(200)]
        assert samples.count('adam') > 190

    def test_two_choice_node(self):
        tree = _initialize_prob_tree({'x': {'values': ['yes', 'no']}})
        for _ in range(100):
            assert _sample_node(tree['x'], temp=1.0) in {'yes', 'no'}

    def test_single_choice_always_returns_that_choice(self):
        tree = _initialize_prob_tree({'x': {'values': ['only']}})
        for _ in range(20):
            assert _sample_node(tree['x'], temp=1.0) == 'only'


# ---------------------------------------------------------------------------
# _sample_node — continuous
# ---------------------------------------------------------------------------

class TestSampleNodeContinuous:
    def setup_method(self):
        random.seed(0)

    def test_returns_float(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        val  = _sample_node(tree['lr'], temp=1.0)
        assert isinstance(val, float)

    def test_value_within_bounds(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        node = tree['lr']
        for _ in range(200):
            val = _sample_node(node, temp=1.0)
            assert node['min'] <= val <= node['max']

    def test_narrow_range_stays_within_bounds(self):
        tree = _initialize_prob_tree(NARROW_CONT)
        node = tree['p']
        for _ in range(200):
            val = _sample_node(node, temp=1.0)
            assert 0.49 <= val <= 0.51

    def test_samples_cover_range(self):
        # Over many draws, samples should spread across [min, max]
        random.seed(42)
        tree = _initialize_prob_tree({'x': {'min': 0.0, 'max': 1.0}})
        node = tree['x']
        samples = [_sample_node(node, temp=1.0) for _ in range(500)]
        assert min(samples) < 0.1
        assert max(samples) > 0.9


# ---------------------------------------------------------------------------
# _sample_node — error handling
# ---------------------------------------------------------------------------

class TestSampleNodeErrors:
    def test_malformed_node_raises_value_error(self):
        with pytest.raises(ValueError, match="Malformed node"):
            _sample_node({'unknown_key': 42}, temp=1.0)

    def test_empty_node_raises_value_error(self):
        with pytest.raises(ValueError, match="Malformed node"):
            _sample_node({}, temp=1.0)


# ---------------------------------------------------------------------------
# _traverse_sample
# ---------------------------------------------------------------------------

class TestTraverseSample:
    def setup_method(self):
        random.seed(42)

    def test_flat_space_returns_all_params(self):
        space = {
            'lr':         {'min': 1e-4, 'max': 1e-1},
            'batch_size': {'values': [16, 32, 64]},
        }
        tree   = _initialize_prob_tree(space)
        config = _traverse_sample(tree, temp=1.0)
        assert 'lr' in config
        assert 'batch_size' in config

    def test_flat_space_no_extra_keys(self):
        space  = {'lr': {'min': 0.0, 'max': 1.0}, 'opt': {'values': ['a', 'b']}}
        tree   = _initialize_prob_tree(space)
        config = _traverse_sample(tree, temp=1.0)
        assert set(config.keys()) == {'lr', 'opt'}

    def test_continuous_value_in_bounds(self):
        space  = {'x': {'min': 2.0, 'max': 5.0}}
        tree   = _initialize_prob_tree(space)
        for _ in range(100):
            config = _traverse_sample(tree, temp=1.0)
            assert 2.0 <= config['x'] <= 5.0

    def test_categorical_value_in_choices(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        for _ in range(50):
            config = _traverse_sample(tree, temp=1.0)
            assert config['optimizer'] in {'adam', 'sgd', 'rmsprop'}

    def test_hierarchical_adam_includes_lr(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        tree['optimizer']['prob'] = {'adam': 1.0, 'sgd': 0.0}
        config = _traverse_sample(tree, temp=0.01)
        assert config['optimizer'] == 'adam'
        assert 'lr' in config
        assert 'batch_size' in config

    def test_hierarchical_sgd_includes_momentum(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        tree['optimizer']['prob'] = {'adam': 0.0, 'sgd': 1.0}
        config = _traverse_sample(tree, temp=0.01)
        assert config['optimizer'] == 'sgd'
        assert 'lr' in config
        assert 'momentum' in config

    def test_hierarchical_adam_excludes_momentum(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        tree['optimizer']['prob'] = {'adam': 1.0, 'sgd': 0.0}
        config = _traverse_sample(tree, temp=0.01)
        assert 'momentum' not in config

    def test_hierarchical_sgd_excludes_adam_only_params(self):
        # adam has no extra params in this fixture, but sgd should not leak
        tree = _initialize_prob_tree(HIERARCHICAL)
        tree['optimizer']['prob'] = {'adam': 0.0, 'sgd': 1.0}
        config = _traverse_sample(tree, temp=0.01)
        # Only sgd sub-params should be present
        assert set(config.keys()) == {'optimizer', 'lr', 'momentum', 'batch_size'}

    def test_hierarchical_batch_size_always_present(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        for _ in range(30):
            config = _traverse_sample(tree, temp=1.0)
            assert 'batch_size' in config

    def test_hierarchical_batch_size_valid(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        for _ in range(30):
            config = _traverse_sample(tree, temp=1.0)
            assert config['batch_size'] in {16, 32, 64}

    def test_empty_tree_returns_empty_config(self):
        config = _traverse_sample({}, temp=1.0)
        assert config == {}

    def test_returns_dict(self):
        tree   = _initialize_prob_tree(SIMPLE_CAT)
        config = _traverse_sample(tree, temp=1.0)
        assert isinstance(config, dict)
