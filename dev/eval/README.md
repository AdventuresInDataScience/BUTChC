# dev/eval — real-world benchmarks

The synthetic suite in `benchmarks/` is a regression loop: fast, dependency-free,
run it on every change. It cannot tell you whether BUTChC works on a real
hierarchical problem, because every problem in it was written alongside the
optimiser.

This tier answers that. It needs dependencies the library does not, which is
why it lives outside the package.

## Install

`yahpo-gym` 1.0.2 pins `ConfigSpace<=0.6.1` and `numpy<2`, so it **must** go in
its own virtualenv. Installing it into an environment that has anything modern
in it will downgrade numpy and break unrelated packages.

```bash
python -m venv .venv-yahpo
.venv-yahpo/bin/pip install -r dev/eval/requirements.txt
```

The surrogate data is a separate repo. The full clone is ~1.6 GB of git
history, but one scenario is ~10 MB, so sparse-checkout it:

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/slds-lmu/yahpo_data.git
cd yahpo_data
git sparse-checkout init --cone
git sparse-checkout set rbv2_super benchmark_suites global_statistics
git checkout                       # ~20 MB on disk
export YAHPO_DATA_PATH=$PWD
```

This recipe is verified working — `BenchmarkSet('rbv2_super')` loads and its
`ConfigurationSpace` is readable. What does not work is the conversion; see
below.

## Tools that need nothing

Two scripts here run on a bare Python, no install and no dependencies:

```bash
python dev/eval/fingerprint.py       # hash seeded runs; prove a refactor is behaviour-neutral
python dev/eval/check_parallel.py    # real thread/process pools; speedup + reproducibility
```

## Run

```bash
python dev/eval/run_real.py 10
python dev/eval/run_real.py 20 --instances 1053 --methods random,optuna,butchc
```

Output is the same format as `benchmarks/evaluate.py` — medians, then paired
win-loss records with sign-test p-values — because it imports that harness
rather than duplicating it.

## Why `rbv2_super` — and why it is not used

It is an AutoML pipeline space: choose a learner, then tune that learner's own
hyperparameters. Measured from the benchmark's own `ConfigurationSpace` rather
than quoted from the paper:

| | |
|---|---|
| Hyperparameters | 41 |
| Active in a valid configuration | median 10 (min 7, max 18) |
| **Inactive per configuration** | **75.6%** |
| Conditions | 25 `EqualsCondition`, 11 `AndConjunction` |
| Forbidden clauses | 0 |
| Instances | 103 OpenML datasets |

That 75.6% is the number BUTChC's founding claim rests on: it is the share of
the space a flat optimiser searches and a conditional one skips. It is the best
available real measurement of that waste, published by the mlr3 group, across a
hundred landscapes rather than one.

The specific thing to look for: the YAHPO paper reports that BOHB did no better
than Hyperband on `rbv2_super`, and attributes it to BOHB's kernel density
estimator struggling with a high-dimensional hierarchical space. BUTChC's
continuous nodes are also KDEs. If the same flattening appears, that is a real
finding about the method, and no synthetic problem in `benchmarks/` will
surface it.

### The blocker

`from_configspace` refuses the space:

```
['ranger.num.random.splits']: gated by a AndConjunction. A tree gives each
parameter one parent, so a conjunction has no single place to live.
```

All 11 conjunctions are **chain-shaped** — none has independent parents:

```
svm.gamma     <- (learner_id == svm)     AND (svm.kernel == radial)
xgboost.eta   <- (learner_id == xgboost) AND (xgboost.booster in {dart, gbtree})
```

and `svm.kernel` is itself gated on `learner_id == svm`. So each one is a
root-to-leaf path — `learner_id=svm -> svm.kernel=radial -> svm.gamma` — which
`next_level` expresses natively and arbitrarily deep. The refusal's stated
reason ("two independent parents") does not apply to any case in this space.

A converter that attached such a child to the *deepest* parent in its chain,
letting tree position imply the shallower conditions, would represent
`rbv2_super` exactly rather than approximately, while still refusing genuinely
independent parents.

**That change was considered and declined.** Extending the converter to fit the
one benchmark that would showcase the library invites the reading that the
benchmark was chosen to fit the tool. The refusal is documented in three places
as a principled limit, and relaxing it to unlock a favourable result is a worse
trade than not having the result. Recorded here so the reasoning is not
rediscovered: the limitation is in `butchc.interop`, not in the tree model, and
`rbv2_super` is representable whenever that is revisited on its own merits.

Consequence: use fallback 1 below. The conditional-waste claim therefore has no
published-benchmark evidence behind it yet, which is worth stating plainly
wherever the claim is made.

## The bridge now ships

The ConfigSpace conversion used to live here as `configspace_bridge.py`. It is
now `butchc.interop`, part of the package but never imported by it, so
ConfigSpace remains an optional extra rather than a dependency:

```bash
pip install butchc[configspace]
```

`from_configspace` is what this suite uses. The refusals, the ordinal
flattening and the log-integer cast are documented in
`butchc/interop/_configspace.py` and covered by `tests/test_interop.py`.

Most of those tests run against **stubs**, which is what keeps the suite
dependency-free: they cover the DAG-to-tree assembly, the refusals and the name
qualifying, which is the part that can actually be wrong about the *design*.
Since 0.5.1 a further seven exercise a real `ConfigurationSpace` — conditions
to branches, constants to `fixed`, forbidden-clause refusal, name qualifying,
round-tripping, sampling — and skip when ConfigSpace is absent. Install the
extra and they run:

```bash
pip install .[configspace] && pytest tests/test_interop.py
```

What neither tier covers is the surrogate stack: `run_real.py` against a live
YAHPO install has still not been done, and per the section above it cannot be
until either the converter is extended or a conjunction-free scenario is used.

## Fallbacks

In rough order of effort. **Fallback 1 is the chosen path.**

1. **A hand-built sklearn CASH space** on a few OpenML-CC18 datasets: scaler ×
   model × model-specific hyperparameters, scored by cross-validation. Real,
   genuinely hierarchical, credible to practitioners, and needs only
   `scikit-learn` + `openml`. Slower per evaluation, so budgets shrink. The
   conditional structure here is the *obvious* encoding of the problem — an
   SVM has no `n_estimators` — so ex-ante knowledge of it is a practitioner's
   normal starting point rather than an advantage handed to the optimiser.
   Its weakness against `rbv2_super` is that we author it, so the space is
   ours; mitigate by taking the model list and ranges from a published AutoML
   default (auto-sklearn's or mlr3's) rather than inventing them, and by
   citing which.
2. **Other YAHPO scenarios.** `lcbench` and `nb301` are in the same 20 MB-per
   -scenario data repo and may avoid conjunctions — worth probing with the
   recipe above before writing anything. Both are flatter than `rbv2_super`,
   so they demonstrate less of the conditional-waste effect.
3. **HPOBench**, which wraps YAHPO among others behind a container interface —
   more setup, but avoids managing surrogate data yourself.
