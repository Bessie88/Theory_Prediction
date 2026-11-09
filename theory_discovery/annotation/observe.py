"""Label actual changes from both texts using numbered diff evidence."""

from difflib import SequenceMatcher


def annotate_actual(before, after, known_actions, label_hunks):
    """The labeler returns Action IDs per hunk and any new definitions."""
    if before == after:
        return frozenset({"NO_CHANGE"}), {}

    hunks = {
        f"H{index}": (before[i:j], after[k:m])
        for index, (kind, i, j, k, m) in enumerate(
            SequenceMatcher(None, before, after, autojunk=False).get_opcodes(), 1
        )
        if kind != "equal"
    }
    assignments, new_actions = label_hunks(before, after, hunks, known_actions)
    allowed = (set(known_actions) | set(new_actions)) - {"NO_CHANGE", "OTHER"}
    if set(assignments) != set(hunks) or any(
        not actions or not set(actions) <= allowed for actions in assignments.values()
    ):
        return None, hunks
    return frozenset(action for actions in assignments.values() for action in actions), hunks
