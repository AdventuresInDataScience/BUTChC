"""Select BUTChC's defaults by measurement rather than by taste.

    python benchmarks/tune.py [n_seeds] [--rounds 2] [--confirm-seeds 30]

A coordinate-descent sweep over every constant that shapes the search, scored
on ``problems.TUNING``. Per round, one knob at a time is varied over its grid
with the others held at their current value, and the best-scoring value is
kept. Candidates are compared by their mean rank across problems: rank rather
than a normalised value, because min-max normalising lets one badly-off
candidate stretch the range until every other candidate compresses together
and most knobs read as exact ties.

Three things protect against reading noise as signal:

*Split seeds.* Selection runs on ``select`` seeds; the chosen configuration is
then confirmed on a disjoint set of ``confirm`` seeds it never saw. A knob
picked because it suited the selection seeds shows up here as a shrunken gap.

*A held-out problem suite.* The confirmation also runs ``problems.HELDOUT``,
which selection never touches, so a configuration fitted to the shape of the
tuning problems shows up as a gap that does not transfer.

*A paired sign test.* The confirmation reports win-loss records with p-values,
not just medians, so "wins by a lot once" is distinguishable from "wins
reliably".

The cache key includes a hash of the ``butchc`` package source. Edit the
library and stale measurements are discarded rather than silently reused.
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import statistics
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.abspath(os.path.join(_HERE, os.pardir))]

import butchc._tree as tree_mod                          # noqa: E402
import butchc._update as upd                             # noqa: E402
import butchc._utils as utils                            # noqa: E402
import butchc.optimizer as opt                           # noqa: E402
from butchc import BUTChC_optimize                       # noqa: E402
from evaluate import sign_test                           # noqa: E402
from problems import HELDOUT, NOISY, TUNING              # noqa: E402

#: Override with ``BUTCHC_TUNE_CACHE``. The default goes to the platform temp
#: directory rather than a hardcoded ``/tmp``, which on Windows Python resolves
#: to ``C:\tmp`` — a directory that does not normally exist, so a multi-hour
#: sweep would discard every measurement it took.
CACHE_PATH = os.environ.get(
    "BUTCHC_TUNE_CACHE",
    os.path.join(tempfile.gettempdir(), "butchc_tune_cache.json"),
)

#: Noise draws are seeded from this base plus the trial seed, so every
#: candidate configuration meets the identical noise sequence.
NOISE_SEED_BASE = 90000

#: The shipped defaults. Keep in step with ``BUTChC_optimize``'s signature and
#: the module constants, or the sweep measures a baseline nobody ships.
BASE = {
    "lambda_": 2.0,
    "alpha": 3.0,
    "temp": 1.0,
    "gamma": 0.85,
    "explore": 0.05,
    "warmup_frac": 0.0,
    "commitment": 8.0,
    "reservoir": 25,
    "discount": 0.98,
    "rank_sharpness": 3.0,
    "min_bandwidth": 0.0003,
    "neutral": 0.5,
}

GRID = {
    "alpha": [0.3, 1.0, 3.0, 10.0, 30.0],
    "gamma": [0.0, 0.25, 0.5, 0.7, 0.85],
    "explore": [0.0, 0.05, 0.1, 0.2, 0.35],
    "lambda_": [0.5, 1.0, 2.0, 4.0],
    "temp": [0.5, 1.0, 2.0],
    "warmup_frac": [0.0, 0.05, 0.10, 0.20],
    "commitment": [2.0, 4.0, 6.0, 8.0, 12.0],
    "reservoir": [25, 50, 100],
    "discount": [1.0, 0.98, 0.95, 0.9],
    "rank_sharpness": [1.5, 3.0, 6.0],
    # Extends below the previous floor of 0.003. Measured directly, 0.003 beat
    # the shipped 0.01 by 185-38 across 17 problems and 0.001 beat 0.003 again
    # on every continuous problem, so the old grid's lower edge was also its
    # best value — which is the signature of a grid that stops too early.
    # 0.0003 is included to bracket the reversal rather than assume it.
    "min_bandwidth": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
    # 0.5 is the expected midrank of a configuration drawn at random, which is
    # the principled value; the grid brackets it because "credit an untried
    # branch with better than average" is an optimism knob, not a mistake.
    "neutral": [0.3, 0.5, 0.7],
}


def source_fingerprint():
    """Hash every source file in the ``butchc`` package.

    Measurements are cached across runs, and a cache that survives an edit to
    the update rule is worse than no cache: it reports the old behaviour under
    the new code's name.
    """
    root = pathlib.Path(opt.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def apply_config(cfg):
    """Patch the module constants a candidate configuration overrides.

    ``KDE_RESERVOIR_SIZE`` is imported by name into two modules besides the
    one that defines it, and a ``from x import y`` binding is a separate name
    that patching ``x`` does not reach. All three are set here: patching only
    some of them leaves the initial grid at one size while the eviction cap
    uses another, and the knob then measures something nobody ships.
    """
    utils.KDE_RESERVOIR_SIZE = cfg["reservoir"]
    tree_mod.KDE_RESERVOIR_SIZE = cfg["reservoir"]
    upd.KDE_RESERVOIR_SIZE = cfg["reservoir"]
    utils.RANK_SHARPNESS = cfg["rank_sharpness"]
    utils.MIN_BANDWIDTH_FRACTION = cfg["min_bandwidth"]
    upd.DISCOUNT = cfg["discount"]
    upd.NEUTRAL_QUALITY = cfg["neutral"]
    opt.COMMITMENT = cfg["commitment"]


def run_one(space, objective, budget, cfg, seed, report=None):
    """One seeded run of one problem under one candidate configuration."""
    apply_config(cfg)
    result = BUTChC_optimize(
        space, objective, budget=budget, verbose=False, seed=seed,
        lambda_=cfg["lambda_"], alpha=cfg["alpha"], temp=cfg["temp"],
        gamma=cfg["gamma"], explore=cfg["explore"],
        n_warmup=int(round(cfg["warmup_frac"] * budget)),
    )
    if report is None:
        return result["best_value"]
    return report(result["best_params"]) if result["best_params"] else -float("inf")


def evaluate(cfg, seeds, suites=(TUNING,), include_noisy=True):
    """Per-seed results for one configuration.

    Returns:
        ``{problem: [result per seed]}``. Per-seed rather than a median, so
        the confirmation stage can pair runs and test them.
    """
    out = {}
    for suite in suites:
        for name, (space, objective, budget) in suite.items():
            out[name] = [run_one(space, objective, budget, cfg, s)
                         for s in seeds]
    if include_noisy:
        for name, (space, objective, budget, clean, state) in NOISY.items():
            values = []
            for s in seeds:
                state["rng"] = random.Random(NOISE_SEED_BASE + s)
                values.append(run_one(space, objective, budget, cfg, s,
                                      report=clean))
            out[name] = values
    return out


# ----------------------------------------------------------------- regimes
#
# One set of defaults cannot be right for every shape of problem. A space with
# twelve branches and one with a single smooth basin want different amounts of
# commitment, and averaging over both produces a compromise that suits neither.
# These subsets let the sweep answer "what would you set for a space like
# mine", which is a more useful thing to publish than one global winner.
#
# Membership is by the property being isolated, not by which suite a problem
# came from, so a regime deliberately mixes tuned-on and held-out problems.

REGIMES = {
    "branched": ["Branch trap", "Categorical mix", "Nested pipeline",
                 "Optimiser choice"],
    "highdim": ["10D sphere", "20D sphere", "Rastrigin 8D", "Griewank 6D"],
    "multimodal": ["Rastrigin 4D", "Ackley 5D", "Styblinski 4D",
                   "Rastrigin 8D"],
    "flat": ["Plateau (ties)", "Rosenbrock", "2D quadratic"],
    "noisy": ["Noisy 5D sphere"],
}


def regime_suites(regime):
    """Return ``(suites, include_noisy)`` for one regime name.

    Raises:
        KeyError: If the regime is not defined.
    """
    names = set(REGIMES[regime])
    pool = {}
    pool.update(TUNING)
    pool.update(HELDOUT)
    picked = {k: v for k, v in pool.items() if k in names}
    missing = names - set(picked) - set(NOISY)
    if missing:
        raise KeyError(f"regime {regime!r} names unknown problems: "
                       f"{sorted(missing)}")
    return (picked,), bool(names & set(NOISY))


def score_candidates(candidates):
    """Rank candidates within each problem, then average the ranks.

    Ties share the average of the ranks they span, so a knob that genuinely
    does nothing reads as a tie rather than as noise.

    Args:
        candidates: ``[(knob_value, {problem: [per-seed result]})]``.

    Returns:
        ``[(score, knob_value)]`` with score in ``[0, 1]``, higher is better.
    """
    if len(candidates) == 1:
        return [(1.0, candidates[0][0])]

    n = len(candidates)
    medians = [{p: statistics.median(v) for p, v in results.items()}
               for _, results in candidates]
    totals = [0.0] * n

    for problem in medians[0]:
        order = sorted(range(n), key=lambda i: medians[i][problem])
        start = 0
        while start < n:
            stop = start
            while (stop + 1 < n
                   and medians[order[stop + 1]][problem]
                   == medians[order[start]][problem]):
                stop += 1
            shared = (start + stop) / 2.0 / (n - 1)
            for position in range(start, stop + 1):
                totals[order[position]] += shared
            start = stop + 1

    n_problems = len(medians[0])
    return [(totals[i] / n_problems, candidates[i][0]) for i in range(n)]


class Measurer:
    """Caches problem results by configuration, seeds and library source."""

    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.cache = self._load()

    def _load(self):
        try:
            with open(CACHE_PATH) as fh:
                cache = json.load(fh)
        except (OSError, ValueError):
            return {}
        return cache if cache.get("_source") == self.fingerprint else {}

    def _save(self):
        self.cache["_source"] = self.fingerprint
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.cache, fh)
        os.replace(tmp, CACHE_PATH)

    def __call__(self, cfg, seeds, suites=(TUNING,), tag="tuning",
                 include_noisy=True):
        key = json.dumps({"cfg": cfg, "seeds": list(seeds), "tag": tag},
                         sort_keys=True)
        if key not in self.cache:
            self.cache[key] = evaluate(cfg, seeds, suites, include_noisy)
            self._save()
        return self.cache[key]


def sweep(measure, seeds, rounds, suites=(TUNING,), include_noisy=True,
          tag="tuning"):
    """Coordinate descent over ``GRID``, starting from ``BASE``."""
    current = dict(BASE)
    for round_no in range(1, rounds + 1):
        print(f"\n=== round {round_no} ===", flush=True)
        for knob, options in GRID.items():
            candidates = []
            for value in options:
                cfg = dict(current)
                cfg[knob] = value
                candidates.append((value, measure(
                    cfg, seeds, suites, tag=tag,
                    include_noisy=include_noisy)))
            scored = score_candidates(candidates)
            best_score = max(score for score, _ in scored)
            # Ties go to the incumbent. `max` over (score, value) pairs would
            # break them on the knob's magnitude, which is not evidence about
            # anything: with few seeds most knobs tie, and the sweep would
            # report every one of them as a change and move the defaults on
            # noise. Keeping the incumbent means "changed" always means the
            # alternative actually scored better.
            incumbent = next(score for score, value in scored
                             if value == current[knob])
            best_value = (current[knob] if incumbent >= best_score
                          else max(v for s_, v in scored if s_ == best_score))
            changed = best_value != current[knob]
            current[knob] = best_value
            by_option = {v: s for s, v in scored}
            trace = "  ".join(f"{v}:{by_option[v]:.3f}" for v in options)
            print(f"  {knob:<15}{trace}   best={best_value}"
                  + ("  <- changed" if changed else ""), flush=True)
    return current


def confirm(measure, chosen, seeds):
    """Compare the chosen configuration against ``BASE`` on unseen seeds.

    Both the tuning and held-out suites are reported. A gap that holds on the
    tuning problems but vanishes on the held-out ones means the sweep fitted
    the problem set, not the algorithm.
    """
    suites = (TUNING, HELDOUT)
    base = measure(BASE, seeds, suites, tag="confirm")
    new = measure(chosen, seeds, suites, tag="confirm")

    tuning_names = set(TUNING) | set(NOISY)
    print(f"\nconfirmation on {len(seeds)} unseen seeds "
          f"(paired; higher is better)\n")
    head = f"{'problem':<20}{'suite':>9}{'base':>12}{'tuned':>12}{'record':>18}"
    print(head)
    print("-" * len(head))
    for name in base:
        wins, losses, p = sign_test(new[name], base[name])
        suite = "tuning" if name in tuning_names else "held-out"
        print(f"{name:<20}{suite:>9}"
              f"{statistics.median(base[name]):>12.4f}"
              f"{statistics.median(new[name]):>12.4f}"
              f"{f'{wins}-{losses}  p={p:.3f}':>18}")

    held = [n for n in base if n not in tuning_names]
    wins = sum(1 for n in held
               if statistics.median(new[n]) > statistics.median(base[n]))
    print(f"\nheld-out problems improved: {wins} of {len(held)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_seeds", nargs="?", type=int, default=12,
                        help="seeds used for selection")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--confirm-seeds", type=int, default=30,
                        help="seeds used for confirmation, disjoint from selection")
    parser.add_argument("--regime", choices=sorted(REGIMES),
                        help="tune against one problem shape instead of the "
                             "whole suite; prints a recommendation rather "
                             "than a proposed default")
    args = parser.parse_args(argv)

    select_seeds = list(range(args.n_seeds))
    confirm_seeds = list(range(1000, 1000 + args.confirm_seeds))

    measure = Measurer(source_fingerprint())
    started = time.time()

    if args.regime:
        suites, include_noisy = regime_suites(args.regime)
        n_problems = len(suites[0]) + (len(NOISY) if include_noisy else 0)
        print(f"regime {args.regime!r}: {args.n_seeds} selection seeds, "
              f"{n_problems} problems, {args.rounds} rounds", flush=True)
        chosen = sweep(measure, select_seeds, args.rounds, suites,
                       include_noisy, tag=f"regime:{args.regime}")
        print(f"\nrecommended for {args.regime!r} spaces:")
        for knob, value in chosen.items():
            if value != BASE[knob]:
                print(f"  {knob:<15}{value}   (default {BASE[knob]})")
        if chosen == BASE:
            print("  no change from the defaults")
        print("\nThis is a recommendation for this problem shape, not a "
              "proposed default:\nit was selected on a handful of problems "
              "sharing one property, so it is\nfitted to that property by "
              "construction. Confirm on your own space.")
        print(f"\n{time.time() - started:.1f}s")
        return

    print(f"coordinate descent: {args.n_seeds} selection seeds, "
          f"{len(TUNING) + len(NOISY)} problems, {args.rounds} rounds",
          flush=True)

    chosen = sweep(measure, select_seeds, args.rounds)

    print("\nchosen:")
    for knob, value in chosen.items():
        print(f"  {knob:<15}{value}"
              + ("" if value == BASE[knob] else f"   (was {BASE[knob]})"))

    confirm(measure, chosen, confirm_seeds)
    print(f"\n{time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
