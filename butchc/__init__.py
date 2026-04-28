"""
BUTChC: Bayesian Update Tree Chained Conditionally

A dependency-free, probabilistic black-box hyperparameter optimizer.
Supports hierarchical/conditional search spaces, KDE-based continuous
parameters, warm starting, and built-in convergence tracking.
"""

from .optimizer import BUTChC_optimize
from ._utils import KDE_RESERVOIR_SIZE

__all__ = ["BUTChC_optimize", "KDE_RESERVOIR_SIZE"]
__version__ = "0.1.0"
