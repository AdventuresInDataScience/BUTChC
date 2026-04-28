"""Tests for butchc._tree: _initialize_prob_tree."""
import pytest

from butchc._tree import _initialize_prob_tree
from butchc._utils import KDE_RESERVOIR_SIZE


# ---------------------------------------------------------------------------
# Fixtures / shared search spaces
# ---------------------------------------------------------------------------

SIMPLE_CAT = {
    'optimizer': {'values': ['adam', 'sgd', 'rmsprop']},
}

SIMPLE_CONT = {
    'lr': {'min': 1e-5, 'max': 1e-1},
}

MIXED = {
    'optimizer':  {'values': ['adam', 'sgd']},
    'dropout':    {'min': 0.0, 'max': 0.5},
    'batch_size': {'values': [16, 32, 64]},
}

HIERARCHICAL = {
    'optimizer': {
        'values': ['adam', 'sgd'],
        'next_level': {
            'adam': {
                'lr':    {'min': 1e-4, 'max': 1e-2},
                'beta1': {'values': [0.9, 0.99]},
            },
            'sgd': {
                'lr':       {'min': 1e-3, 'max': 1e-1},
                'momentum': {'min': 0.0, 'max': 0.99},
            },
        },
    },
    'batch_size': {'values': [16, 32, 64, 128]},
}


# ---------------------------------------------------------------------------
# Categorical node tests
# ---------------------------------------------------------------------------

class TestCategoricalNode:
    def test_has_counts_key(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        assert 'counts' in tree['optimizer']

    def test_has_prob_key(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        assert 'prob' in tree['optimizer']

    def test_no_reservoir_key(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        assert 'reservoir' not in tree['optimizer']

    def test_probs_sum_to_one(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        total = sum(tree['optimizer']['prob'].values())
        assert abs(total - 1.0) < 1e-9

    def test_initial_probs_uniform(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        probs = list(tree['optimizer']['prob'].values())
        n = len(probs)
        for p in probs:
            assert abs(p - 1 / n) < 1e-9

    def test_initial_counts_equal(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        counts = list(tree['optimizer']['counts'].values())
        assert all(c == counts[0] for c in counts)

    def test_all_values_present_in_prob(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        assert set(tree['optimizer']['prob'].keys()) == {'adam', 'sgd', 'rmsprop'}

    def test_all_values_present_in_counts(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        assert set(tree['optimizer']['counts'].keys()) == {'adam', 'sgd', 'rmsprop'}

    def test_two_value_categorical(self):
        tree = _initialize_prob_tree({'choice': {'values': ['yes', 'no']}})
        probs = tree['choice']['prob']
        assert abs(probs['yes'] - 0.5) < 1e-9
        assert abs(probs['no'] - 0.5) < 1e-9

    def test_single_value_categorical(self):
        tree = _initialize_prob_tree({'only': {'values': ['x']}})
        assert abs(tree['only']['prob']['x'] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Continuous node tests
# ---------------------------------------------------------------------------

class TestContinuousNode:
    def test_has_reservoir_key(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert 'reservoir' in tree['lr']

    def test_has_weights_key(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert 'weights' in tree['lr']

    def test_has_min_max_keys(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert 'min' in tree['lr']
        assert 'max' in tree['lr']

    def test_no_prob_key(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert 'prob' not in tree['lr']

    def test_reservoir_size_equals_constant(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert len(tree['lr']['reservoir']) == KDE_RESERVOIR_SIZE

    def test_weights_size_matches_reservoir(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        node = tree['lr']
        assert len(node['weights']) == len(node['reservoir'])

    def test_reservoir_first_point_equals_min(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert abs(tree['lr']['reservoir'][0] - 1e-5) < 1e-12

    def test_reservoir_last_point_equals_max(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert abs(tree['lr']['reservoir'][-1] - 1e-1) < 1e-12

    def test_reservoir_all_points_within_bounds(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        lo, hi = tree['lr']['min'], tree['lr']['max']
        assert all(lo <= v <= hi for v in tree['lr']['reservoir'])

    def test_weights_sum_to_one(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert abs(sum(tree['lr']['weights']) - 1.0) < 1e-9

    def test_initial_weights_uniform(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        weights  = tree['lr']['weights']
        expected = 1.0 / len(weights)
        for w in weights:
            assert abs(w - expected) < 1e-9

    def test_min_max_preserved(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert tree['lr']['min'] == 1e-5
        assert tree['lr']['max'] == 1e-1

    def test_total_weight_initialised_to_one(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        assert tree['lr']['total_weight'] == 1.0

    def test_unit_range(self):
        tree = _initialize_prob_tree({'p': {'min': 0.0, 'max': 1.0}})
        node = tree['p']
        assert abs(node['reservoir'][0] - 0.0) < 1e-12
        assert abs(node['reservoir'][-1] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Mixed / hierarchical space tests
# ---------------------------------------------------------------------------

class TestMixedSpace:
    def test_all_top_level_keys_present(self):
        tree = _initialize_prob_tree(MIXED)
        assert set(tree.keys()) == {'optimizer', 'dropout', 'batch_size'}

    def test_categorical_and_continuous_coexist(self):
        tree = _initialize_prob_tree(MIXED)
        assert 'prob' in tree['optimizer']
        assert 'reservoir' in tree['dropout']

    def test_empty_searchspace_returns_empty_tree(self):
        assert _initialize_prob_tree({}) == {}


class TestHierarchicalSpace:
    def test_top_level_has_next_level_key(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        assert 'next_level' in tree['optimizer']

    def test_simple_cat_has_no_next_level(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        assert 'next_level' not in tree['optimizer']

    def test_both_choices_in_next_level(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        assert 'adam' in tree['optimizer']['next_level']
        assert 'sgd' in tree['optimizer']['next_level']

    def test_adam_subtree_has_lr_and_beta1(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        adam = tree['optimizer']['next_level']['adam']
        assert 'lr' in adam
        assert 'beta1' in adam

    def test_sgd_subtree_has_lr_and_momentum(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        sgd = tree['optimizer']['next_level']['sgd']
        assert 'lr' in sgd
        assert 'momentum' in sgd

    def test_adam_lr_is_continuous_node(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        lr_node = tree['optimizer']['next_level']['adam']['lr']
        assert 'reservoir' in lr_node
        assert 'min' in lr_node
        assert 'max' in lr_node

    def test_adam_beta1_is_categorical_node(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        beta1_node = tree['optimizer']['next_level']['adam']['beta1']
        assert 'prob' in beta1_node
        assert set(beta1_node['prob'].keys()) == {0.9, 0.99}

    def test_sub_reservoir_within_bounds(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        lr_node = tree['optimizer']['next_level']['sgd']['lr']
        lo, hi = lr_node['min'], lr_node['max']
        assert all(lo <= v <= hi for v in lr_node['reservoir'])

    def test_sub_weights_sum_to_one(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        weights = tree['optimizer']['next_level']['sgd']['momentum']['weights']
        assert abs(sum(weights) - 1.0) < 1e-9
