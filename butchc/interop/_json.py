"""Serialising a search space to JSON and back.

A BUTChC search space is already plain data, so this is nearly ``json.dumps``.
Nearly, and not quite, for one reason: ``next_level`` is keyed by the parent's
*values*, and JSON object keys are always strings. A space branching on
``{'values': [1, 2, 3]}`` therefore comes back with ``next_level`` keyed by
``"1"``, ``"2"``, ``"3"``, which no longer match the choices — the branch would
be silently unreachable and every config would come back missing its
sub-parameters.

``from_json`` repairs that by re-keying each ``next_level`` against its own
node's ``values``, which are recoverable because JSON preserves them as a list.
"""

import json


def to_json(searchspace, **kwargs):
    """Serialise a search space to a JSON string.

    Args:
        searchspace: A BUTChC search-space dict.
        **kwargs:    Passed to ``json.dumps`` — ``indent=2`` is the usual one.

    Returns:
        A JSON string.

    Raises:
        TypeError: If the space contains values JSON cannot represent. Tuples
            are the common case: use lists.
    """
    kwargs.setdefault("sort_keys", False)
    return json.dumps(searchspace, **kwargs)


def from_json(text):
    """Parse a search space from a JSON string, restoring ``next_level`` keys.

    Args:
        text: JSON produced by ``to_json``, or written by hand.

    Returns:
        A BUTChC search-space dict.
    """
    return _restore(json.loads(text))


def _restore(subspace):
    for spec in subspace.values():
        levels = spec.get("next_level")
        if not levels:
            continue
        by_string = {str(v): v for v in spec.get("values", [])}
        spec["next_level"] = {
            by_string.get(key, key): _restore(sub) for key, sub in levels.items()
        }
    return subspace
