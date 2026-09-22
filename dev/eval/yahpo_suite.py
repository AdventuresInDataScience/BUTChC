"""YAHPO Gym problems, exposed in the shape ``benchmarks/evaluate.py`` expects.

``rbv2_super`` is the reason this file exists. It is an AutoML pipeline space —
choose a learner, then tune that learner's own hyperparameters — which is
precisely the conditional structure BUTChC is built for, and it is a published
benchmark rather than a function invented alongside the optimiser being tested.
It is 38-dimensional with 7 conditional hyperparameters across roughly 100
OpenML instances, and it is surrogate-backed, so an evaluation is a neural
network forward pass rather than a model fit.

There is a specific result in the YAHPO paper worth checking against: on
``rbv2_super``, BOHB performed no better than Hyperband, which the authors
attribute to its kernel density estimator struggling with a high-dimensional
hierarchical space. BUTChC's continuous nodes are also KDEs. If the same
flattening shows up here, that is the finding, and it will not show up on any
synthetic problem in ``benchmarks/``.

Install:

    pip install yahpo-gym ConfigSpace
    # then download the surrogate data and point yahpo at it; see
    # https://github.com/slds-lmu/yahpo_gym

Status: the YAHPO side of this **works**. ``BenchmarkSet('rbv2_super')`` loads
against a 20 MB sparse checkout of the data repo (recipe in
``dev/eval/README.md``) and its ``ConfigurationSpace`` reads back 41
hyperparameters, 36 conditions and 103 instances.

What blocks it is ``butchc.interop``, not this file. ``from_configspace``
refuses ``rbv2_super`` because 11 of its conditions are ``AndConjunction``s —
all of them chain-shaped, e.g. ``(learner_id == svm) AND (svm.kernel ==
radial)``, which is the tree path ``learner_id=svm -> svm.kernel=radial`` and
is representable. Extending the converter to accept chains was considered and
declined, on the grounds that widening a documented refusal to unlock a
favourable benchmark result reads as fitting the benchmark to the tool. See
``dev/eval/README.md`` for the full reasoning and the chosen fallback.

So nothing below has been run end to end. Do not treat the numbers this module
would produce as measured until that changes.
"""

import os

from butchc.interop import from_configspace, wrap_objective

#: Instances used by the YAHPO single-objective benchmark suite for
#: ``rbv2_super``. These are OpenML *dataset* ids, not task ids.
RBV2_SUPER_INSTANCES = ["1053", "1457", "1063", "1479", "15", "1468"]

#: Parameters the benchmark fixes rather than optimises: the instance selector
#: and the multi-fidelity knobs. Searching them would be optimising the
#: benchmark's own budget.
NOT_SEARCHED = ("task_id", "OpenML_task_id", "trainsize", "repl", "epoch")

#: Objective evaluations per run. YAHPO's own single-objective protocol is
#: worth following rather than inventing a budget; adjust here if it differs
#: for the scenario you pick.
DEFAULT_BUDGET = 400


def _benchmark_set(scenario, instance):
    """Construct a YAHPO ``BenchmarkSet`` pinned to one instance."""
    from yahpo_gym import benchmark_set, local_config

    if os.environ.get("YAHPO_DATA_PATH"):
        local_config.init_config(data_path=os.environ["YAHPO_DATA_PATH"])

    bench = benchmark_set.BenchmarkSet(scenario)
    bench.set_instance(instance)
    return bench


def make_problem(scenario, instance, target="acc", budget=DEFAULT_BUDGET,
                 maximise=True):
    """Build one ``(searchspace, objective, budget)`` triple.

    The tuple shape matches ``benchmarks/problems.py``, so the same harness,
    the same baselines and the same sign tests apply without modification.

    Args:
        scenario: YAHPO scenario name, e.g. ``"rbv2_super"``.
        instance: Instance id within that scenario.
        target:   Which of the scenario's target metrics to optimise.
        budget:   Objective evaluations.
        maximise: BUTChC always maximises. Set False for a loss metric and the
                  objective is negated here, so every problem in the suite is
                  a maximisation and the tables stay comparable.

    Returns:
        ``(searchspace, objective, budget)``.
    """
    bench = _benchmark_set(scenario, instance)
    space = bench.get_opt_space(drop_fidelity_params=True)
    searchspace, fixed, casts = from_configspace(space, drop=NOT_SEARCHED)

    # The instance is part of the query, not part of the search.
    fixed = dict(fixed)
    fixed.setdefault("task_id", instance)

    def raw(config):
        result = bench.objective_function(config)
        record = result[0] if isinstance(result, list) else result
        value = float(record[target])
        return value if maximise else -value

    return searchspace, wrap_objective(raw, fixed, casts), budget


def rbv2_super(instances=None, target="acc", budget=DEFAULT_BUDGET):
    """The ``rbv2_super`` suite, keyed by name like ``problems.TUNING``.

    Each instance is a different OpenML dataset over the same hierarchical
    space, so this is one search-space design measured across many landscapes
    — a much better test of a structural claim than many designs measured once.
    """
    out = {}
    for instance in (instances or RBV2_SUPER_INSTANCES):
        out[f"rbv2_super/{instance}"] = make_problem(
            "rbv2_super", instance, target=target, budget=budget
        )
    return out
