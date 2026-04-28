"""Tests for butchc._update: _update_tree."""
import pytest

from butchc._tree import _initialize_prob_tree
from butchc._update import _update_tree
from butchc._utils import KDE_RESERVOIR_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_CAT  = {'optimizer': {'values': ['adam', 'sgd', 'rmsprop']}}
SIMPLE_CONT = {'lr': {'min': 0.0, 'max': 1.0}}

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
# Categorical updates
# ---------------------------------------------------------------------------

class TestUpdateCategorical:
    def test_chosen_count_increases(self):
        tree   = _initialize_prob_tree(SIMPLE_CAT)
        before = tree['optimizer']['counts']['adam']
        _update_tree(tree, {'optimizer': 'adam'}, lambda_=1.0, alpha=1.0)
        assert tree['optimizer']['counts']['adam'] > before

    def test_unchosen_counts_unchanged(self):
        tree        = _initialize_prob_tree(SIMPLE_CAT)
        sgd_before  = tree['optimizer']['counts']['sgd']
        rmsp_before = tree['optimizer']['counts']['rmsprop']
        _update_tree(tree, {'optimizer': 'adam'}, lambda_=1.0, alpha=1.0)
        assert tree['optimizer']['counts']['sgd']     == sgd_before
        assert tree['optimizer']['counts']['rmsprop'] == rmsp_before

    def test_probs_sum_to_one_after_update(self):
        tree = _initialize_prob_tree(SIMPLE_CAT)
        _update_tree(tree, {'optimizer': 'adam'}, lambda_=1.0, alpha=1.0)
        assert abs(sum(tree['optimizer']['prob'].values()) - 1.0) < 1e-9

    def test_chosen_prob_increases(self):
        tree   = _initialize_prob_tree(SIMPLE_CAT)
        before = tree['optimizer']['prob']['adam']
        _update_tree(tree, {'optimizer': 'adam'}, lambda_=1.0, alpha=1.0)
        assert tree['optimizer']['prob']['adam'] > before

    def test_unchosen_probs_decrease(self):
        tree      = _initialize_prob_tree(SIMPLE_CAT)
        sgd_bef   = tree['optimizer']['prob']['sgd']
        rmsp_bef  = tree['optimizer']['prob']['rmsprop']
        _update_tree(tree, {'optimizer': 'adam'}, lambda_=1.0, alpha=1.0)
        assert tree['optimizer']['prob']['sgd']     < sgd_bef
        assert tree['optimizer']['prob']['rmsprop'] < rmsp_bef

    def test_returns_list(self):
        tree   = _initialize_prob_tree(SIMPLE_CAT)
        deltas = _update_tree(tree, {'optimizer': 'adam'}, lambda_=1.0, alpha=1.0)
        assert isinstance(deltas, list)

    def test_returns_one_delta_for_flat_cat(self):
        tree   = _initialize_prob_tree(SIMPLE_CAT)
        deltas = _update_tree(tree, {'optimizer': 'adam'}, lambda_=1.0, alpha=1.0)
        assert len(deltas) == 1

    def test_delta_in_valid_range(self):
        tree   = _initialize_prob_tree(SIMPLE_CAT)
        deltas = _update_tree(tree, {'optimizer': 'adam'}, lambda_=1.0, alpha=1.0)
        assert all(0.0 <= d <= 1.0 for d in deltas)

    def test_many_updates_converge_to_chosen(self):
        tree   = _initialize_prob_tree(SIMPLE_CAT)
        config = {'optimizer': 'adam'}
        for _ in range(100):
            _update_tree(tree, config, lambda_=1.0, alpha=0.1)
        assert tree['optimizer']['prob']['adam'] > 0.9

    def test_missing_param_returns_empty_deltas(self):
        tree   = _initialize_prob_tree(SIMPLE_CAT)
        deltas = _update_tree(tree, {}, lambda_=1.0, alpha=1.0)
        assert deltas == []

    def test_large_lambda_shifts_more(self):
        # With larger lambda_ the probability shift after one update is larger
        tree_small = _initialize_prob_tree(SIMPLE_CAT)
        tree_large = _initialize_prob_tree(SIMPLE_CAT)
        d_small = _update_tree(tree_small, {'optimizer': 'adam'}, lambda_=0.1, alpha=1.0)
        d_large = _update_tree(tree_large, {'optimizer': 'adam'}, lambda_=10.0, alpha=1.0)
        assert d_large[0] > d_small[0]

    def test_high_alpha_keeps_distribution_flatter(self):
        tree_low_a  = _initialize_prob_tree(SIMPLE_CAT)
        tree_high_a = _initialize_prob_tree(SIMPLE_CAT)
        for _ in range(20):
            _update_tree(tree_low_a,  {'optimizer': 'adam'}, lambda_=1.0, alpha=0.01)
            _update_tree(tree_high_a, {'optimizer': 'adam'}, lambda_=1.0, alpha=10.0)
        # High alpha keeps distribution flatter — adam prob should be lower
        assert tree_high_a['optimizer']['prob']['adam'] < tree_low_a['optimizer']['prob']['adam']


# ---------------------------------------------------------------------------
# Continuous updates
# ---------------------------------------------------------------------------

class TestUpdateContinuous:
    def test_reservoir_does_not_exceed_max_size(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        for i in range(200):
            _update_tree(tree, {'lr': i / 200}, lambda_=1.0, alpha=1.0)
        assert len(tree['lr']['reservoir']) <= KDE_RESERVOIR_SIZE

    def test_weights_sum_to_one_after_single_update(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        _update_tree(tree, {'lr': 0.5}, lambda_=1.0, alpha=1.0)
        assert abs(sum(tree['lr']['weights']) - 1.0) < 1e-9

    def test_weights_sum_to_one_after_many_updates(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        for v in [0.1, 0.2, 0.3, 0.7, 0.9]:
            _update_tree(tree, {'lr': v}, lambda_=1.0, alpha=1.0)
        assert abs(sum(tree['lr']['weights']) - 1.0) < 1e-9

    def test_returns_one_delta_for_single_cont_param(self):
        tree   = _initialize_prob_tree(SIMPLE_CONT)
        deltas = _update_tree(tree, {'lr': 0.5}, lambda_=1.0, alpha=1.0)
        assert len(deltas) == 1

    def test_delta_in_valid_range(self):
        tree   = _initialize_prob_tree(SIMPLE_CONT)
        deltas = _update_tree(tree, {'lr': 0.5}, lambda_=1.0, alpha=1.0)
        assert 0.0 <= deltas[0] <= 1.0

    def test_high_weight_point_survives_pruning(self):
        # A point inserted with a very high lambda_ should not be pruned
        tree = _initialize_prob_tree(SIMPLE_CONT)
        _update_tree(tree, {'lr': 0.99}, lambda_=1000.0, alpha=1.0)
        assert 0.99 in tree['lr']['reservoir']

    def test_weights_all_positive(self):
        tree = _initialize_prob_tree(SIMPLE_CONT)
        for v in [0.1, 0.5, 0.9]:
            _update_tree(tree, {'lr': v}, lambda_=1.0, alpha=1.0)
        assert all(w > 0 for w in tree['lr']['weights'])

    def test_zero_range_node_delta_is_zero(self):
        # A continuous node where min == max should produce delta = 0
        tree = {'p': {'min': 0.5, 'max': 0.5, 'reservoir': [0.5], 'weights': [1.0], 'total_weight': 1.0}}
        deltas = _update_tree(tree, {'p': 0.5}, lambda_=1.0, alpha=1.0)
        assert deltas[0] == 0.0


# ---------------------------------------------------------------------------
# Hierarchical updates
# ---------------------------------------------------------------------------

class TestUpdateHierarchical:
    def test_adam_updates_its_subtree(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        weights_before = list(tree['optimizer']['next_level']['adam']['lr']['weights'])
        _update_tree(
            tree,
            {'optimizer': 'adam', 'lr': 0.005, 'batch_size': 32},
            lambda_=1.0, alpha=1.0,
        )
        weights_after = tree['optimizer']['next_level']['adam']['lr']['weights']
        assert weights_after != weights_before

    def test_sgd_does_not_update_adam_subtree(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        weights_before = list(tree['optimizer']['next_level']['adam']['lr']['weights'])
        _update_tree(
            tree,
            {'optimizer': 'sgd', 'lr': 0.05, 'momentum': 0.9, 'batch_size': 32},
            lambda_=1.0, alpha=1.0,
        )
        assert tree['optimizer']['next_level']['adam']['lr']['weights'] == weights_before

    def test_adam_does_not_update_sgd_subtree(self):
        tree = _initialize_prob_tree(HIERARCHICAL)
        weights_before = list(tree['optimizer']['next_level']['sgd']['lr']['weights'])
        _update_tree(
            tree,
            {'optimizer': 'adam', 'lr': 0.005, 'batch_size': 32},
            lambda_=1.0, alpha=1.0,
        )
        assert tree['optimizer']['next_level']['sgd']['lr']['weights'] == weights_before

    def test_returns_deltas_for_all_touched_nodes(self):
        tree   = _initialize_prob_tree(HIERARCHICAL)
        # Touching optimizer (cat) + lr (cont) + batch_size (cat) = 3 nodes
        deltas = _update_tree(
            tree,
            {'optimizer': 'adam', 'lr': 0.005, 'batch_size': 32},
            lambda_=1.0, alpha=1.0,
        )
        assert len(deltas) == 3

    def test_sgd_update_touches_momentum_subtree(self):
        tree   = _initialize_prob_tree(HIERARCHICAL)
        # optimizer (cat) + lr (cont) + momentum (cont) + batch_size (cat) = 4
        deltas = _update_tree(
            tree,
            {'optimizer': 'sgd', 'lr': 0.05, 'momentum': 0.9, 'batch_size': 32},
            lambda_=1.0, alpha=1.0,
        )
        assert len(deltas) == 4

    def test_all_hierarchical_deltas_in_valid_range(self):
        tree   = _initialize_prob_tree(HIERARCHICAL)
        deltas = _update_tree(
            tree,
            {'optimizer': 'adam', 'lr': 0.005, 'batch_size': 32},
            lambda_=1.0, alpha=1.0,
        )
        for d in deltas:
            assert 0.0 <= d <= 1.0
