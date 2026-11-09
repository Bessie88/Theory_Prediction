"""Restricted JSON DSL with three-valued logic. Never eval/exec model code."""
import math
from theory_discovery.core.schemas import require

COMPARATORS = {'eq', 'ne', 'gt', 'gte', 'lt', 'lte'}


def validate_expr(expr, condition_ids, depth=0):
    require(depth <= 16 and isinstance(expr, dict), 'Invalid/deep rule expression')
    op = expr.get('op')
    if op in ('and', 'or'):
        require(set(expr) == {'op', 'args'} and isinstance(expr['args'], list) and 1 <= len(expr['args']) <= 64, 'Invalid logical operands')
        for arg in expr['args']:
            validate_expr(arg, condition_ids, depth + 1)
    elif op == 'not':
        require(set(expr) == {'op', 'arg'}, 'Invalid negation')
        validate_expr(expr['arg'], condition_ids, depth + 1)
    elif op == 'pred':
        require(set(expr) == {'op', 'condition_id'} and expr['condition_id'] in condition_ids, 'Unknown predicate')
    elif op in COMPARATORS:
        require(set(expr) == {'op', 'condition_id', 'value'} and expr['condition_id'] in condition_ids, 'Unknown comparison predicate')
        value = expr['value']
        require(type(value) in (bool, int, float, str), 'Non-scalar comparator')
        if type(value) in (int, float):
            require(math.isfinite(value), 'Non-finite threshold')
        if op not in ('eq', 'ne'):
            require(type(value) in (int, float), 'Ordered comparison requires numeric threshold')
    else:
        raise ValueError('Unsupported DSL operator: ' + str(op))


def evaluate(expr, values):
    op = expr['op']
    if op in ('and', 'or'):
        results = [evaluate(x, values) for x in expr['args']]
        if op == 'and':
            return False if False in results else (None if None in results else True)
        return True if True in results else (None if None in results else False)
    if op == 'not':
        value = evaluate(expr['arg'], values)
        return None if value is None else not value
    value = values.get(expr['condition_id'])
    if value is None:
        return None
    if op == 'pred':
        return value if type(value) is bool else None
    target = expr['value']
    numeric = type(value) in (int, float) and type(target) in (int, float)
    if op not in ('eq', 'ne') and not numeric:
        return None
    if not numeric and type(value) is not type(target):
        return None
    if numeric and not math.isfinite(value):
        return None
    return {'eq': lambda: value == target, 'ne': lambda: value != target,
            'gt': lambda: value > target, 'gte': lambda: value >= target,
            'lt': lambda: value < target, 'lte': lambda: value <= target}[op]()


def predicate(condition_id):
    return {'op': 'pred', 'condition_id': condition_id}


def conjunction(args):
    args = list(args)
    require(bool(args), 'Empty conditions are not a theory')
    return args[0] if len(args) == 1 else {'op': 'and', 'args': args}


def effective_expression(rule):
    args = [rule['conditions'], *rule['boundaries']]
    args.extend({'op': 'not', 'arg': exception} for exception in rule['exceptions'])
    return conjunction(args)


def condition_ids(expr):
    if expr['op'] in ('and', 'or'):
        return set().union(*(condition_ids(x) for x in expr['args']))
    if expr['op'] == 'not':
        return condition_ids(expr['arg'])
    return {expr['condition_id']}


def complexity(expr):
    if expr['op'] in ('and', 'or'):
        return 1 + sum(complexity(a) for a in expr['args'])
    return 1 + complexity(expr['arg']) if expr['op'] == 'not' else 1


class RuleExecutor:
    def __init__(self, theory):
        self.theory = theory
        ids = {c['condition_id'] for c in theory['conditions']}
        require(len(ids) == len(theory['conditions']), 'Duplicate theory conditions')
        for rule in theory['rules']:
            validate_expr(effective_expression(rule), ids)
            require(rule['status'] in ('candidate', 'ambiguous', 'insufficient_evidence'), 'Invalid rule status')
            require(rule['claim']['minimum_applicable_cases'] > 0 and 0 <= rule['claim']['expected_min_action_frequency'] <= 1, 'Invalid falsifiable claim')

    def predict_values(self, values):
        matches, unknown = [], []
        for rule in self.theory['rules']:
            if rule['status'] == 'insufficient_evidence':
                continue
            result = evaluate(effective_expression(rule), values)
            if result is True:
                matches.append({'rule_id': rule['rule_id'], 'action': rule['action'],
                                'conditional_action_frequency': rule['conditional_action_frequency'],
                                'status': rule['status']})
            elif result is None:
                unknown.append(rule['rule_id'])
        return {'predictions': matches, 'abstained': not bool(matches), 'unknown_rule_ids': unknown,
                'reason': 'applicable_candidate_rules' if matches else ('unknown_conditions' if unknown else 'no_supported_match')}

    def predict_before(self, context, detector):
        # BeforeContext is the only accepted prediction input; After cannot be passed.
        from theory_discovery.core.schemas import BeforeContext, condition_from_dict
        require(type(context) is BeforeContext, 'Prediction requires BeforeContext')
        conditions = [condition_from_dict(c) for c in self.theory['conditions']]
        return self.predict_values(detector.detect(context, conditions))


def check_claim(rule, observations, groups):
    """Future independent check; this function does not load or infer anything."""
    construction_ids = set(rule['construction_case_ids'])
    require(not construction_ids & {o.case_id for o in observations}, 'Claim check overlaps construction evidence')
    require(all(o.split in ('test', 'development') for o in observations), 'Claim check requires independent split')
    from .condition_comparison import evidence_statistics
    stats = evidence_statistics(effective_expression(rule), rule['action']['group_id'], observations, groups)
    claim = rule['claim']
    if stats['matched_cases'] < claim['minimum_applicable_cases']:
        outcome = 'insufficient_evidence'
    else:
        outcome = 'challenged' if stats['conditional_action_frequency'] < claim['expected_min_action_frequency'] else 'not_challenged'
    return {'outcome': outcome, 'statistics': stats, 'note': 'Descriptive frequency check; not a causal or statistical significance claim.'}
