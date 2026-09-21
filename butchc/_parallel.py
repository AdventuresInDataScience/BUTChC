"""Evaluating a batch of configurations, optionally in parallel.

BUTChC never creates a process or thread pool. It accepts an ``executor`` you
supply and dispatches through it. That keeps the package dependency-free, keeps
process lifecycle out of a library that has no business owning it, and leaves
the picklability decision with the person who wrote the objective — which is
the only person who can make it.

Any object satisfying one of these works:

- ``.map(fn, iterable)`` returning results **in input order** —
  ``concurrent.futures`` executors, ``multiprocessing.Pool``, ``joblib``
  wrappers, ``dask.distributed.Client``, ``mpi4py.futures``.
- ``.submit(fn, arg)`` returning futures with ``.result()`` — used when there
  is no ``map``.

Order preservation is load-bearing. Updates are applied in *sample* order so a
seeded run is reproducible regardless of which evaluation finishes first; an
executor that returned results as they completed would silently break that.
"""

import os
from functools import partial


def resolve_batch(batch):
    """Turn the user-facing ``batch`` argument into a concrete size.

    ``-1`` means one configuration per available CPU. ``os.cpu_count()`` can
    return ``None`` on exotic platforms, in which case the sequential default
    is the safe reading.
    """
    if batch == -1:
        return os.cpu_count() or 1
    return batch


def evaluate_batch(objective, configs, executor, kwargs):
    """Evaluate ``configs``, returning results in the same order.

    A single configuration, or no executor, is evaluated inline — there is
    nothing to gain from dispatching one call, and a batch without an executor
    is a deliberate mode (see ``batch`` in the API docs) rather than an error.
    """
    if executor is None or len(configs) == 1:
        return [objective(config, **kwargs) for config in configs]

    call = partial(objective, **kwargs) if kwargs else objective

    try:
        if hasattr(executor, "map"):
            return list(executor.map(call, configs))
        if hasattr(executor, "submit"):
            futures = [executor.submit(call, config) for config in configs]
            return [f.result() for f in futures]
    except Exception as exc:
        raise _explain(exc) from exc

    raise TypeError(
        f"executor must provide 'map' or 'submit'; "
        f"{type(executor).__name__} provides neither"
    )


def _explain(exc):
    """Attach the diagnosis that a raw pickling error does not carry.

    Process pools transmit the objective by pickle, and the objectives people
    write are very often closures over training data or functions defined in
    ``__main__`` at an interactive prompt. Neither pickles. The failure surfaces
    far from its cause, so name the cause.
    """
    text = str(exc).lower()
    picklish = (
        isinstance(exc, (AttributeError, TypeError))
        and ("pickle" in text or "local object" in text or "lambda" in text)
    ) or type(exc).__name__ in ("PicklingError", "MaybeEncodingError")

    if not picklish:
        return exc

    return RuntimeError(
        "the objective could not be sent to the executor. A process pool "
        "transmits the objective by pickle, so lambdas, closures and functions "
        "defined interactively cannot be used with one. Either define the "
        "objective at module level and pass its data via **kwargs, or use a "
        "ThreadPoolExecutor — threads share memory and pickle nothing, and an "
        "objective that spends its time in NumPy, scikit-learn, PyTorch or a "
        "subprocess releases the GIL and parallelises fine on threads. "
        f"Original error: {exc}"
    )
