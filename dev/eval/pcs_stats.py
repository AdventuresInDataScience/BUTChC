"""How hierarchical are real configuration spaces, and how much of one is dead?

BUTChC rests on a claim about waste: in a space where most parameters are
inactive under most configurations, an optimiser that knows the conditional
structure skips what a flat one searches. The synthetic suite in `benchmarks/`
cannot evidence that claim, because its spaces are narrow and were written
here. `dev/eval/README.md` records why the obvious real candidate, YAHPO's
`rbv2_super`, is unavailable.

This script evidences the *premise* rather than the claim. It measures the
inactive share of configuration spaces that were published by other people,
for other purposes, years before this library existed. It does not run an
optimiser and it does not need a solver, a surrogate or a dependency: a PCS
file is a text description of a space, and the share of it that goes unused is
a property of the text.

Source: the `.pcs` files in ConfigSpace's own test suite, which vendors the
spaces from AClib and the Configurable SAT Solver Competition.

    python dev/eval/pcs_stats.py --download
    python dev/eval/pcs_stats.py --dir path/to/pcs --samples 20000

Columns, per space:

    params        declared parameters
    cond          parameters gated behind at least one condition
    active        median parameters live in a sampled valid configuration
    inactive      share of declared parameters that are dead, on median
    depth         longest root-to-leaf chain of conditions
    multi         children gated on more than one parent
    indep         those whose parents are *not* on one chain
    forbid        declared forbidden clauses

`indep` is the number that matters for representability. BUTChC gives each
parameter one parent, so a child gated on two parents that lie on a single
root-to-leaf path is expressible -- tree position implies the shallower
condition -- while a child gated on two genuinely independent parents is not.
`butchc.interop` currently refuses both; the distinction is what tells you
whether that refusal is a modelling limit or a converter limit.
"""

import argparse
import os
import random
import re
import statistics
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "pcs_cache")

BASE = ("https://raw.githubusercontent.com/automl/ConfigSpace/main"
        "/test/test_searchspaces")

# Chosen for breadth rather than favourability: the two AutoML pipeline spaces
# that predate this library, and the algorithm-configuration spaces that the
# field built conditional configuration for in the first place.
SPACES = [
    "auto-sklearn_2017_11_17.pcs",
    "autoweka_original.pcs",
    "clasp-3.1.4.pcs",
    "cplex12.6.pcs",
    "lingeling-params.pcs",
    "satenstein.pcs",
    "spear-params.pcs",
    "cryptominisat-params.pcs",
    "SparrowToRiss-cssc14.pcs",
    "probSAT.pcs",
    "lpg.pcs",
    "vampire.pcs",
]

# name {a, b} [default]           categorical
# name [lo, hi] [default]         real, optional trailing i / l / il
_CATEGORICAL = re.compile(r"^\s*([^\s|{}]+)\s*\{([^}]*)\}\s*\[([^\]]*)\]")
_NUMERIC = re.compile(r"^\s*([^\s|\[\]]+)\s*\[([^\]]*)\]\s*\[([^\]]*)\]\s*([il]*)")
# child | parent in {a, b} && other in {c}
_CONDITION = re.compile(r"^\s*([^\s|]+)\s*\|\s*(.+)$")
_CLAUSE = re.compile(r"([^\s|]+)\s+in\s*\{([^}]*)\}")


def strip_comment(line):
    return line.split("#", 1)[0].rstrip()


def parse(path):
    """Return (params, conditions, forbidden) from a PCS file.

    ``params`` maps name -> list of values for a categorical, or None for a
    numeric. ``conditions`` maps child -> list of clause dicts, one per
    condition line, each mapping parent -> set of permitted values.
    """
    params, conditions, forbidden = {}, {}, 0

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = strip_comment(raw)
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                forbidden += 1
                continue

            match = _CONDITION.match(line)
            if match and " in " in match.group(2):
                child, rest = match.group(1), match.group(2)
                clause = {p: {v.strip() for v in vals.split(",")}
                          for p, vals in _CLAUSE.findall(rest)}
                if clause:
                    conditions.setdefault(child, []).append(clause)
                continue

            match = _CATEGORICAL.match(line)
            if match:
                values = [v.strip() for v in match.group(2).split(",")
                          if v.strip()]
                params[match.group(1)] = values
                continue

            match = _NUMERIC.match(line)
            if match:
                params[match.group(1)] = None

    # A condition may name a parent the file never declares; drop those so the
    # activity sampler does not deadlock on a parameter that cannot be set.
    for child, clauses in list(conditions.items()):
        kept = [c for c in clauses if all(p in params for p in c)]
        if kept:
            conditions[child] = kept
        else:
            del conditions[child]
    conditions = {c: v for c, v in conditions.items() if c in params}
    return params, conditions, forbidden


def parents_of(clauses):
    return {p for clause in clauses for p in clause}


def depth_of(conditions):
    """Longest root-to-leaf chain, guarding against cycles in malformed files."""
    memo, visiting = {}, set()

    def walk(name):
        if name in memo:
            return memo[name]
        if name in visiting or name not in conditions:
            return 0
        visiting.add(name)
        best = 1 + max((walk(p) for p in parents_of(conditions[name])),
                       default=0)
        visiting.discard(name)
        memo[name] = best
        return best

    return max((walk(n) for n in conditions), default=0)


def ancestors(name, conditions, seen=None):
    seen = set() if seen is None else seen
    for parent in parents_of(conditions.get(name, [])):
        if parent not in seen:
            seen.add(parent)
            ancestors(parent, conditions, seen)
    return seen


def multi_parent_shapes(conditions):
    """Split multi-parent children into chain-shaped and genuinely independent.

    A child gated on ``learner == svm`` and ``svm.kernel == radial`` has two
    parents, but one is an ancestor of the other, so the pair is a path and a
    tree can hold it. A child gated on two parents from different subtrees
    cannot be placed.
    """
    multi, independent = 0, 0
    for child, clauses in conditions.items():
        parents = parents_of(clauses)
        if len(parents) < 2:
            continue
        multi += 1
        chain = any(
            all(other == p or other in ancestors(p, conditions)
                for other in parents)
            for p in parents
        )
        if not chain:
            independent += 1
    return multi, independent


def sample_active(params, conditions, rng):
    """Count parameters live in one uniformly sampled valid configuration.

    Values are drawn for categoricals only; a numeric parent is treated as
    never satisfying a set-membership condition, which is how these files use
    them. Activity is resolved to a fixpoint because a parent may itself be
    conditional.
    """
    drawn = {n: rng.choice(v) for n, v in params.items() if v}
    active = {n for n in params if n not in conditions}

    changed = True
    while changed:
        changed = False
        for child, clauses in conditions.items():
            if child in active:
                continue
            for clause in clauses:
                if all(p in active and drawn.get(p) in vals
                       for p, vals in clause.items()):
                    active.add(child)
                    changed = True
                    break
    return len(active)


def analyse(path, samples, seed):
    params, conditions, forbidden = parse(path)
    if not params:
        return None
    rng = random.Random(seed)
    counts = [sample_active(params, conditions, rng) for _ in range(samples)]
    active = statistics.median(counts)
    multi, independent = multi_parent_shapes(conditions)
    return {
        "name": os.path.basename(path).replace(".pcs", ""),
        "params": len(params),
        "cond": len(conditions),
        "active": active,
        "lo": min(counts),
        "hi": max(counts),
        "inactive": 100.0 * (1.0 - active / len(params)),
        "depth": depth_of(conditions),
        "multi": multi,
        "indep": independent,
        "forbidden": forbidden,
    }


def download(names, into):
    if not os.path.isdir(into):
        os.makedirs(into)
    for name in names:
        target = os.path.join(into, name)
        if os.path.exists(target):
            continue
        url = "{}/{}".format(BASE, name)
        print("fetching {} ...".format(name))
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        with open(target, "wb") as handle:
            handle.write(data)


HEADER = ("{:<28} {:>7} {:>6} {:>8} {:>10} {:>6} {:>6} {:>6} {:>7}"
          .format("space", "params", "cond", "active", "inactive",
                  "depth", "multi", "indep", "forbid"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", default=CACHE,
                        help="directory of .pcs files (default: pcs_cache)")
    parser.add_argument("--download", action="store_true",
                        help="fetch the published spaces into --dir first")
    parser.add_argument("--samples", type=int, default=10000,
                        help="configurations drawn per space (default: 10000)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.download:
        download(SPACES, args.dir)

    if not os.path.isdir(args.dir):
        print("no such directory: {}\nrun with --download first."
              .format(args.dir), file=sys.stderr)
        return 1

    paths = sorted(os.path.join(args.dir, f) for f in os.listdir(args.dir)
                   if f.endswith(".pcs"))
    if not paths:
        print("no .pcs files in {}\nrun with --download first."
              .format(args.dir), file=sys.stderr)
        return 1

    rows = [r for r in (analyse(p, args.samples, args.seed) for p in paths)
            if r]
    rows.sort(key=lambda r: r["inactive"], reverse=True)

    print("inactive share of published configuration spaces")
    print("({} configurations sampled uniformly per space)\n"
          .format(args.samples))
    print(HEADER)
    print("-" * len(HEADER))
    for r in rows:
        print("{name:<28} {params:>7} {cond:>6} {active:>8.0f} "
              "{inactive:>9.1f}% {depth:>6} {multi:>6} {indep:>6} "
              "{forbidden:>7}".format(**r))

    conditional = [r for r in rows if r["cond"]]
    if conditional:
        print("\nacross {} conditional spaces: median inactive share {:.1f}%"
              .format(len(conditional),
                      statistics.median(r["inactive"] for r in conditional)))
        exact = [r for r in conditional if not r["indep"] and not r["forbidden"]]
        print("{} of {} are expressible as a tree with no approximation: {}"
              .format(len(exact), len(conditional),
                      ", ".join(r["name"] for r in exact) or "none"))
        blocked = [r for r in conditional if r["indep"] or r["forbidden"]]
        for r in blocked:
            reasons = []
            if r["indep"]:
                reasons.append("{} independent-parent children".format(r["indep"]))
            if r["forbidden"]:
                reasons.append("{} forbidden clauses".format(r["forbidden"]))
            print("  {:<26} not expressible: {}"
                  .format(r["name"], "; ".join(reasons)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
