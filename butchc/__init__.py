"""
BUTChC: Bayesian Update Tree Chained Conditionally

A dependency-free, probabilistic black-box hyperparameter optimiser with
hierarchical/conditional search spaces, KDE-modelled continuous parameters,
log and integer scales, warm starting, optional batch/parallel evaluation and
convergence tracking.

``butchc.interop`` converts to and from ConfigSpace and JSON. It is not
imported here, so ConfigSpace never becomes a dependency of the core package.
"""

from ._prune import prune, prune_report
from ._tree import reservoir_summary
from ._utils import KDE_RESERVOIR_SIZE
from ._validate import SearchSpaceError
from .optimizer import BUTChC_optimize

__all__ = [
    "BUTChC_optimize",
    "KDE_RESERVOIR_SIZE",
    "SearchSpaceError",
    "prune",
    "prune_report",
    "reservoir_summary",
]
__version__ = "0.6.0"
