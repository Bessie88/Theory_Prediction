"""Freeze finalized condition and action features as executable rules."""

from dataclasses import dataclass

# A gate is a feature ID or an expression such as ("and", "f1", ("not", "f2")).
Gate = str | tuple["Gate", ...]


@dataclass(frozen=True)
class Rule:
    condition_id: str
    gate: Gate
    actions: frozenset[str]


def freeze(theory, fold_boundaries, condition_gate, action_feature):
    """Compile the finalized condition and action text into immutable rules."""
    return tuple(
        Rule(condition_id, condition_gate(condition),
             frozenset(action_feature(action) for action in actions))
        for condition_id, condition, actions in fold_boundaries(theory)
    )
