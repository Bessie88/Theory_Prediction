"""Induce an evidence-backed theory from the earlier cohort."""

def induce(edits, identify_actions, code_before, group_actions,
           compare_conditions, find_boundaries, synthesize):
    """The supplied functions represent the model-backed discovery steps."""
    observations = []
    for case_id, before, after in edits:
        actions = identify_actions(before, after)
        observations.append({
            "case_id": case_id,
            "actions": actions,
            "conditions": code_before(before, actions),
        })
    groups = group_actions(observations)
    rules = compare_conditions(groups, observations)
    boundaries = find_boundaries(rules, groups, observations)
    return synthesize(rules, boundaries)
