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

Three scripts here run on a bare Python, no install and no dependencies:

```bash
python dev/eval/fingerprint.py       # hash seeded runs; prove a refactor is behaviour-neutral
python dev/eval/check_parallel.py    # real thread/process pools; speedup + reproducibility
python dev/eval/pcs_stats.py --download   # inactive share of published config spaces
```

## How hierarchical are real spaces? (`pcs_stats.py`)

The founding claim needs two things: that real spaces waste a large share of a
flat optimiser's budget on inactive parameters, and that BUTChC exploits it.
The second needs a surrogate and a run. The first is a property of published
space definitions, and `pcs_stats.py` measures it against the `.pcs` files
vendored in ConfigSpace's test suite, which come from AClib and the
Configurable SAT Solver Competition. No solver, no surrogate, no dependency.

| Space | Params | Median active | Inactive | Depth | Multi-parent | Independent | Forbidden |
|---|---|---|---|---|---|---|---|
| autoweka_original | 786 | 14 | **98.2%** | 4 | 174 | **0** | **0** |
| auto-sklearn_2017_11_17 | 138 | 16 | 88.4% | 2 | 0 | 0 | 79 |
| SparrowToRiss-cssc14 | 222 | 67 | 69.8% | 4 | 13 | 11 | 21 |
| satenstein | 54 | 26 | 51.9% | 4 | 33 | 31 | 0 |
| clasp-3.1.4 | 98 | 59 | 39.8% | 3 | 5 | 2 | 2 |
| lpg | 67 | 51 | 23.9% | 2 | 3 | 3 | 12 |
| probSAT | 9 | 7 | 22.2% | 1 | 1 | 1 | 0 |
| spear-params | 26 | 24 | 7.7% | 2 | 0 | 0 | 0 |
| cplex12.6 | 74 | 71 | 4.1% | 1 | 0 | 0 | 0 |
| cryptominisat-params | 36 | 36 | 0.0% | 1 | 0 | 0 | 0 |
| lingeling-params | 323 | 323 | 0.0% | 0 | 0 | 0 | 0 |

Three findings, two of which were not what was expected:

- **The AutoML pipeline spaces are the hierarchical ones, not the SAT solvers.**
  `lingeling` is 323 parameters and completely flat; `cplex12.6` has four
  conditions. The intuition that algorithm-configuration spaces are the most
  conditional artefacts published is wrong — they are *wide*. AutoWEKA and
  auto-sklearn are the deep ones.
- **AutoWEKA beats `rbv2_super` on the metric that matters.** 98.2% inactive
  against 75.6%, on 786 parameters against 41.
- **AutoWEKA is exactly representable.** Zero forbidden clauses, and all 174 of
  its multi-parent children are chain-shaped — no child has two independent
  parents. The tree model can hold this space with no approximation; only the
  converter refuses it.

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

**That change was considered and declined, on grounds that have since
weakened.** The original objection was that extending the converter to fit the
one benchmark that would showcase the library invites the reading that the
benchmark was chosen to fit the tool. `pcs_stats.py` changes the arithmetic:
chain-shaped conjunctions are the common case across eleven published spaces,
not a quirk of `rbv2_super`, so the extension is now a general converter
improvement that happens to unlock a benchmark rather than the reverse. Worth
revisiting — with the caveat that the credible order is to ship and test the
converter change on its own merits *first*, and run the benchmark afterwards.

The limitation is in `butchc.interop`, not in the tree model.

Consequence for now: the *premise* of the conditional-waste claim is evidenced
by `pcs_stats.py`; the *head-to-head* is not, and that distinction is worth
preserving wherever the claim is made.

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

Reordered after `pcs_stats.py`. **Option 1 is the chosen path.**

1. **AutoWEKA's published space, evaluated through scikit-learn equivalents.**
   The `pcs_stats.py` table above makes this the best candidate available:
   786 parameters, 98.2% inactive, depth 4, zero forbidden clauses, zero
   independent-parent children. It is published (Thornton et al., 2013),
   predates this library by a decade, and answers the authorship objection
   that sinks a hand-built space. Two pieces of work: teach the converter
   chain-shaped conjunctions (see the blocker above — note that argument now
   cuts differently, since the extension is justified by eleven spaces rather
   than by the one it would unlock), and map WEKA learners onto sklearn
   equivalents, or drive WEKA directly.
2. **A hand-built sklearn CASH space** on a few OpenML-CC18 datasets: scaler ×
   model × model-specific hyperparameters, scored by cross-validation. Needs
   only `scikit-learn` + `openml` and no converter work, which is why it stays
   on the list. Its weakness is that we author it; mitigate by taking the model
   list and ranges from a published AutoML default and citing which.
3. **HPOBench**, which wraps YAHPO among others behind a container interface —
   more setup, but avoids managing surrogate data yourself.

**Ruled out.** `lcbench` is 7 flat parameters and `nb301` is a DARTS cell space
of per-edge operation choices; neither has the conditional structure this tier
exists to test. Live AClib runs are also out: the objective is mean PAR10 over
an instance set at a 300 s cutoff, which puts a single trial in the tens of
minutes and 30 paired seeds beyond any sane budget.
