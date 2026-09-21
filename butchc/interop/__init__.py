"""Conversion between BUTChC search spaces and other libraries' formats.

Nothing in this package is imported unless you use it, and nothing in it is a
hard dependency. ``butchc`` itself remains standard-library only.

ConfigSpace is the format every standard HPO benchmark hands you — YAHPO Gym,
HPOBench, HPO-B, SMAC's own suites — so accepting and emitting it is what makes
BUTChC comparable against its peers on their home ground.

The two directions are not symmetric, and that asymmetry decides the design:

- **ConfigSpace to BUTChC is partial.** ConfigSpace expresses conditionality as
  a directed acyclic graph of conditions; BUTChC expresses it as a tree. The
  DAG is strictly more general, so conjunctions, multiple parents and forbidden
  clauses have no faithful tree representation. They are refused by name rather
  than approximated, because approximating them means silently optimising a
  different problem.
- **BUTChC to ConfigSpace never refuses.** Every tree is a DAG. The one
  reconciliation is naming: BUTChC allows a name to repeat across sibling
  branches, ConfigSpace does not, so ``to_configspace`` qualifies collisions
  and returns the mapping.
"""

from ._configspace import (
    UnsupportedSpace,
    from_configspace,
    to_configspace,
    wrap_objective,
)
from ._json import from_json, to_json

__all__ = [
    "UnsupportedSpace",
    "from_configspace",
    "to_configspace",
    "wrap_objective",
    "from_json",
    "to_json",
]
