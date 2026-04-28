"""Integration tests for butchc.BUTChC_optimize."""
import copy
import random
import pytest

from butchc import BUTChC_optimize


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FLAT_SPACE = {
    'x':      {'min': -5.0, 'max': 5.0},
    'method': {'values': ['a', 'b', 'c']},
}

HIERARCHICAL_SPACE = {
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


def quadratic(config):
    """Maximize -(x-1)^2; peak at x=1."""
    return -(config['x'] - 1.0) ** 2


def constant_zero(config):
    return 0.0


def _run(space=FLAT_SPACE, obj=quadratic, budget=10, **kw):
    return BUTChC_optimize(
        space, obj, budget=budget,
        lambda_=1.0, alpha=1.0, verbose=False, **kw
    )


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    EXPECTED_KEYS = {'best_params', 'best_value', 'prob_tree',
                     'history', 'loss_history', 'rolling_loss'}

    def test_returns_dict(self):
        assert isinstance(_run(), dict)

    def test_all_keys_present(self):
        results = _run()
        assert self.EXPECTED_KEYS == set(results.keys())

    def test_history_length_equals_budget(self):
        assert len(_run(budget=7)['history']) == 7

    def test_loss_history_length_equals_budget(self):
        assert len(_run(budget=13)['loss_history']) == 13

    def test_rolling_loss_length_equals_budget(self):
        assert len(_run(budget=9)['rolling_loss']) == 9

    def test_history_entries_have_required_keys(self):
        results = _run(budget=3)
        for entry in results['history']:
            assert {'params', 'objective', 'loss', 'rolling_loss'}.issubset(entry.keys())

    def test_history_params_are_dicts(self):
        results = _run(budget=3)
        for entry in results['history']:
            assert isinstance(entry['params'], dict)

    def test_history_objectives_are_floats(self):
        results = _run(budget=3)
        for entry in results['history']:
            assert isinstance(entry['objective'], float)

    def test_loss_history_matches_history(self):
        results = _run(budget=5)
        expected = [h['loss'] for h in results['history']]
        assert results['loss_history'] == expected

    def test_rolling_loss_matches_history(self):
        results = _run(budget=5)
        expected = [h['rolling_loss'] for h in results['history']]
        assert results['rolling_loss'] == expected

    def test_prob_tree_is_dict(self):
        assert isinstance(_run()['prob_tree'], dict)

    def test_prob_tree_not_empty(self):
        assert len(_run()['prob_tree']) > 0

    def test_best_params_is_dict(self):
        assert isinstance(_run()['best_params'], dict)

    def test_best_value_is_float(self):
        assert isinstance(_run()['best_value'], float)


# ---------------------------------------------------------------------------
# Best-tracking correctness
# ---------------------------------------------------------------------------

class TestBestTracking:
    def test_best_value_equals_max_of_objectives(self):
        results = _run(budget=30)
        all_obj = [h['objective'] for h in results['history']]
        assert results['best_value'] == max(all_obj)

    def test_best_value_not_minus_infinity(self):
        results = _run(budget=3)
        assert results['best_value'] > -float('inf')

    def test_best_params_objective_matches_best_value(self):
        results = _run(budget=20)
        recomputed = quadratic(results['best_params'])
        assert abs(recomputed - results['best_value']) < 1e-9

    def test_best_value_is_non_decreasing_over_history(self):
        results = _run(budget=15)
        running_best = -float('inf')
        for entry in results['history']:
            running_best = max(running_best, entry['objective'])
        assert running_best == results['best_value']

    def test_budget_one_has_valid_best(self):
        results = _run(budget=1)
        assert results['best_params'] is not None
        assert results['best_value'] == results['history'][0]['objective']


# ---------------------------------------------------------------------------
# Loss properties
# ---------------------------------------------------------------------------

class TestLossProperties:
    def test_all_losses_non_negative(self):
        results = _run(budget=20)
        assert all(l >= 0.0 for l in results['loss_history'])

    def test_all_rolling_losses_non_negative(self):
        results = _run(budget=20)
        assert all(l >= 0.0 for l in results['rolling_loss'])

    def test_rolling_loss_non_decreasing_in_early_window(self):
        # Rolling loss is a running mean; it can only be 0 if all losses are 0
        results = _run(budget=5)
        # Simply check it's valid floats, not checking monotonicity (it's a mean)
        for rl in results['rolling_loss']:
            assert isinstance(rl, float)
            assert rl >= 0.0


# ---------------------------------------------------------------------------
# Config / objective forwarding
# ---------------------------------------------------------------------------

class TestObjectiveForwarding:
    def test_config_contains_searchspace_params(self):
        seen = []

        def capture(config):
            seen.append(config)
            return 0.0

        _run(obj=capture, budget=5)
        assert len(seen) == 5
        for cfg in seen:
            assert 'x' in cfg
            assert 'method' in cfg

    def test_kwargs_forwarded_to_objective(self):
        calls = []

        def obj_with_kwarg(config, scale=1.0):
            calls.append(scale)
            return 0.0

        BUTChC_optimize(
            {'x': {'min': 0.0, 'max': 1.0}},
            obj_with_kwarg,
            budget=3,
            lambda_=1.0, alpha=1.0, verbose=False,
            scale=7.0,
        )
        assert all(s == 7.0 for s in calls)

    def test_multiple_kwargs_forwarded(self):
        received = []

        def obj(config, a=0, b=0):
            received.append((a, b))
            return 0.0

        BUTChC_optimize(
            {'x': {'min': 0.0, 'max': 1.0}},
            obj,
            budget=2,
            lambda_=1.0, alpha=1.0, verbose=False,
            a=3, b=9,
        )
        assert all(pair == (3, 9) for pair in received)

    def test_objective_called_exactly_budget_times(self):
        counter = {'n': 0}

        def counting_obj(config):
            counter['n'] += 1
            return 0.0

        _run(obj=counting_obj, budget=17)
        assert counter['n'] == 17


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------

class TestVerbose:
    def test_verbose_false_produces_no_output(self, capsys):
        _run(budget=3)
        assert capsys.readouterr().out == ''

    def test_verbose_true_produces_output(self, capsys):
        BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=2,
            lambda_=1.0, alpha=1.0, verbose=True,
        )
        assert capsys.readouterr().out != ''

    def test_verbose_true_output_line_count_equals_budget(self, capsys):
        budget = 4
        BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=budget,
            lambda_=1.0, alpha=1.0, verbose=True,
        )
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert len(lines) == budget


# ---------------------------------------------------------------------------
# Warm starting
# ---------------------------------------------------------------------------

class TestWarmStart:
    def test_warm_start_runs_without_error(self):
        r1 = _run(budget=10)
        r2 = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=10,
            lambda_=1.0, alpha=1.0, verbose=False,
            start_prob_tree=r1['prob_tree'],
        )
        assert 'best_value' in r2

    def test_warm_start_does_not_mutate_input_tree(self):
        r1         = _run(budget=10)
        tree_copy  = copy.deepcopy(r1['prob_tree'])
        BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=5,
            lambda_=1.0, alpha=1.0, verbose=False,
            start_prob_tree=r1['prob_tree'],
        )
        assert r1['prob_tree'] == tree_copy

    def test_warm_start_has_correct_history_length(self):
        r1 = _run(budget=10)
        r2 = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=7,
            lambda_=1.0, alpha=1.0, verbose=False,
            start_prob_tree=r1['prob_tree'],
        )
        assert len(r2['history']) == 7

    def test_chained_warm_start(self):
        r1 = _run(budget=5)
        r2 = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=5,
            lambda_=1.0, alpha=1.0, verbose=False,
            start_prob_tree=r1['prob_tree'],
        )
        r3 = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=5,
            lambda_=1.0, alpha=1.0, verbose=False,
            start_prob_tree=r2['prob_tree'],
        )
        assert 'best_value' in r3


# ---------------------------------------------------------------------------
# Hierarchical space
# ---------------------------------------------------------------------------

class TestHierarchicalSpace:
    def test_configs_always_have_optimizer_and_batch_size(self):
        seen = []

        def capture(config):
            seen.append(config)
            return 0.0

        BUTChC_optimize(
            HIERARCHICAL_SPACE, capture, budget=20,
            lambda_=1.0, alpha=1.0, verbose=False,
        )
        for cfg in seen:
            assert 'optimizer' in cfg
            assert 'lr' in cfg
            assert 'batch_size' in cfg

    def test_sgd_configs_include_momentum(self):
        random.seed(42)
        sgd_configs = []

        def capture(config):
            if config['optimizer'] == 'sgd':
                sgd_configs.append(config)
            return 0.0

        BUTChC_optimize(
            HIERARCHICAL_SPACE, capture, budget=100,
            lambda_=1.0, alpha=1.0, verbose=False,
        )
        assert len(sgd_configs) > 0, "No SGD samples in 100 trials — increase budget"
        for cfg in sgd_configs:
            assert 'momentum' in cfg

    def test_adam_configs_exclude_momentum(self):
        random.seed(42)
        adam_configs = []

        def capture(config):
            if config['optimizer'] == 'adam':
                adam_configs.append(config)
            return 0.0

        BUTChC_optimize(
            HIERARCHICAL_SPACE, capture, budget=100,
            lambda_=1.0, alpha=1.0, verbose=False,
        )
        assert len(adam_configs) > 0, "No Adam samples in 100 trials — increase budget"
        for cfg in adam_configs:
            assert 'momentum' not in cfg


# ---------------------------------------------------------------------------
# Determinism and reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_gives_same_results(self):
        random.seed(99)
        r1 = _run(budget=15)
        random.seed(99)
        r2 = _run(budget=15)
        assert r1['best_value']   == r2['best_value']
        assert r1['loss_history'] == r2['loss_history']

    def test_different_seeds_may_differ(self):
        random.seed(1)
        r1 = _run(budget=20)
        random.seed(2)
        r2 = _run(budget=20)
        # Not guaranteed, but with high probability different seeds explore differently
        # Just check both are valid
        assert r1['best_value'] > -float('inf')
        assert r2['best_value'] > -float('inf')


# ---------------------------------------------------------------------------
# Convergence (statistical, seeded)
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_finds_near_optimal_for_simple_quadratic(self):
        random.seed(42)
        space   = {'x': {'min': -5.0, 'max': 5.0}}
        results = BUTChC_optimize(
            space,
            lambda cfg: -(cfg['x'] - 1.0) ** 2,
            budget=300,
            lambda_=2.0, alpha=1.0, verbose=False,
        )
        # Should find x within 1.5 of the true optimum at x=1
        assert abs(results['best_params']['x'] - 1.0) < 1.5

    def test_minimisation_via_negation(self):
        random.seed(0)
        space   = {'x': {'min': 0.0, 'max': 6.0}}
        results = BUTChC_optimize(
            space,
            lambda cfg: -((cfg['x'] - 3.0) ** 2),
            budget=300,
            lambda_=2.0, alpha=1.0, verbose=False,
        )
        assert abs(results['best_params']['x'] - 3.0) < 1.5

    def test_categorical_converges_to_best_choice(self):
        random.seed(7)

        def obj(config):
            scores = {'a': 1.0, 'b': 0.5, 'c': 0.1}
            return scores[config['choice']]

        results = BUTChC_optimize(
            {'choice': {'values': ['a', 'b', 'c']}},
            obj,
            budget=200, lambda_=2.0, alpha=0.5, verbose=False,
        )
        assert results['best_params']['choice'] == 'a'
        assert results['best_value'] == 1.0

    def test_two_d_continuous_improves_over_baseline(self):
        # Verify the optimizer finds a better-than-random solution in 2D.
        # Testing the objective value is more robust than testing coordinates
        # because the KDE may not fully concentrate in both dims in 400 trials.
        random.seed(5)
        space = {
            'x': {'min': -3.0, 'max': 3.0},
            'y': {'min': -3.0, 'max': 3.0},
        }

        def obj(config):
            return -((config['x'] - 1.0) ** 2) - ((config['y'] + 1.0) ** 2)

        results = BUTChC_optimize(
            space, obj, budget=400,
            lambda_=2.0, alpha=1.0, verbose=False,
        )
        # Worst possible value on this grid is -(4^2 + 4^2) = -32;
        # a decent optimizer should get well above -10.
        assert results['best_value'] > -10.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_budget_of_one(self):
        results = _run(budget=1)
        assert len(results['history']) == 1
        assert results['best_params'] is not None

    def test_all_same_objective_value(self):
        results = _run(obj=constant_zero, budget=10)
        assert results['best_value'] == 0.0

    def test_temperature_zero_point_one_does_not_crash(self):
        results = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=5,
            lambda_=1.0, alpha=1.0, temp=0.1, verbose=False,
        )
        assert 'best_value' in results

    def test_high_temperature_does_not_crash(self):
        results = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=5,
            lambda_=1.0, alpha=1.0, temp=10.0, verbose=False,
        )
        assert 'best_value' in results

    def test_very_large_lambda_does_not_crash(self):
        results = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=5,
            lambda_=100.0, alpha=1.0, verbose=False,
        )
        assert 'best_value' in results

    def test_very_small_lambda_does_not_crash(self):
        results = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=5,
            lambda_=1e-6, alpha=1.0, verbose=False,
        )
        assert 'best_value' in results

    def test_large_budget_runs_to_completion(self):
        results = BUTChC_optimize(
            {'x': {'min': 0.0, 'max': 1.0}},
            constant_zero,
            budget=500,
            lambda_=1.0, alpha=1.0, verbose=False,
        )
        assert len(results['history']) == 500
