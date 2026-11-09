"""Measure conditions from Before and apply the frozen rules."""


def features_in(gate):
    if isinstance(gate, str):
        return {gate}
    return {feature for child in gate[1:] for feature in features_in(child)}


def evaluate_gate(gate, values):
    if isinstance(gate, str):
        return values[gate]
    operator, *children = gate
    results = [evaluate_gate(child, values) for child in children]
    if operator == "not" and len(results) == 1:
        return None if results[0] is None else not results[0]
    if operator in {"and", "or"} and results:
        decisive = operator == "or"
        if decisive in results:
            return decisive
        return None if None in results else not decisive
    raise ValueError(f"Invalid gate: {gate!r}")


def measure_conditions(before, rules, measure):
    feature_ids = {feature for rule in rules for feature in features_in(rule.gate)}
    labels = {feature: measure(before, feature) for feature in sorted(feature_ids)}
    if any(value is not None and type(value) is not bool for value in labels.values()):
        raise ValueError("Feature measurements must be Boolean or unavailable")
    return {rule.condition_id: evaluate_gate(rule.gate, labels) for rule in rules}


def predict(before, rules, measure):
    rules = tuple(rules)
    conditions = measure_conditions(before, rules, measure)
    actions = frozenset(action for rule in rules if conditions[rule.condition_id] is True
                        for action in rule.actions)
    incomplete = None in conditions.values()
    status = "PREDICTED" if actions else "ERROR" if incomplete else "NOT_COVERED"
    return status, actions, conditions
