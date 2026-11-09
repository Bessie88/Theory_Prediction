"""Stage 5: compact, executable, probabilistic candidate theories."""
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION

STAGE_INSTRUCTION = PLAIN_LANGUAGE_INSTRUCTION
from .rule_executor import RuleExecutor, complexity, condition_ids, effective_expression
from theory_discovery.core.schemas import stable_id, to_jsonable


def candidate_rule_pool(boundaries, groups, conditions, observations, config):
    group_index = {g.group_id: g for g in groups}
    candidates = []
    construction_ids = sorted(o.case_id for o in observations)
    for discovery in boundaries:
        group = group_index[discovery['group_id']]
        if not group.normalized_action['semantic'] or group.grouping_uncertainty:
            # Uninterpreted diff or uncertain grouping is evidence, not a behavioral theory.
            continue
        revision = discovery['selected_revision']
        stats = revision['statistics'] if revision else discovery['original_statistics']
        status = 'candidate'
        if stats['support_count'] < config.min_support or (stats['conditional_action_frequency'] or 0) < config.min_action_frequency:
            status = 'insufficient_evidence'
        elif stats['contradiction_count']:
            status = 'ambiguous'
        rule = {'rule_id': stable_id('rule', [group.group_id, discovery['original_conditions'], revision['boundary'] if revision else None]),
                'action': {'group_id': group.group_id, **group.normalized_action},
                'conditions': discovery['original_conditions'],
                'boundaries': [revision['boundary']] if revision else [], 'exceptions': [],
                'supporting_case_ids': stats['supporting_case_ids'],
                'contradicting_case_ids': stats['contradicting_case_ids'],
                'unknown_case_ids': stats['unknown_case_ids'],
                'construction_case_ids': construction_ids,
                'coverage': stats['coverage'], 'conditional_action_frequency': stats['conditional_action_frequency'],
                'evidence_statistics': stats, 'status': status,
                'claim': {'kind': 'conditional_action_frequency',
                          'expected_min_action_frequency': config.min_action_frequency,
                          'minimum_applicable_cases': config.min_test_cases,
                          'evaluation_population': 'independent observed neutrality-edit pairs satisfying the rule',
                          'construction_estimate_is_validation': False},
                'refinement_history': {'hypothesis_id': discovery['hypothesis_id'],
                                       'selected_revision_id': revision['revision_id'] if revision else None,
                                       'original_counterexample_ids': discovery['original_statistics']['contradicting_case_ids'],
                                       'unresolved_case_ids': discovery['unresolved_case_ids']}}
        rule['statistical_diagnostic_status'] = status
        if discovery.get('induction_mode') == 'qwen_open_code_induction':
            # Qwen decides whether the rule is useful from evidence in theory construction.
            rule['status'] = 'candidate'
        rule['natural_rule_text'] = discovery.get('natural_rule_text', '')
        rule['induction_mode'] = discovery.get('induction_mode', 'statistical')
        rule['induction_rationale'] = discovery.get('induction_rationale', '')
        rule['induction_evidence_citations'] = discovery.get('induction_evidence_citations', {})
        rule['boundary_rationale'] = revision.get('rationale', '') if revision else ''
        rule['complexity'] = complexity(effective_expression(rule))
        candidates.append(rule)
    candidates.sort(key=lambda r: (r['complexity'], -(r['conditional_action_frequency'] or 0), -len(r['supporting_case_ids']), r['rule_id']))
    rules, suppressed = [], []
    for rule in candidates:
        # Same action and identical empirical truth partition: keep the simpler rule.
        signature = (rule['action']['group_id'], tuple(rule['supporting_case_ids']),
                     tuple(rule['contradicting_case_ids']), tuple(rule['unknown_case_ids']))
        existing = next((r for r in rules if (r['action']['group_id'], tuple(r['supporting_case_ids']),
                        tuple(r['contradicting_case_ids']), tuple(r['unknown_case_ids'])) == signature), None)
        if existing and rule.get('induction_mode') != 'qwen_open_code_induction':
            suppressed.append({'rule_id': rule['rule_id'], 'retained_rule_id': existing['rule_id'],
                               'reason': 'equivalent_construction_evidence_prefer_simpler', 'rule': rule})
        else:
            rules.append(rule)
    used = set().union(*(condition_ids(effective_expression(r)) for r in rules)) if rules else set()
    theory = {'schema_version': '1', 'status': 'candidate_unvalidated', 'language_instruction': STAGE_INSTRUCTION,
              'scope': 'How selected observed neutrality edits are performed; observational associations only.',
              'conditions': [to_jsonable(c) for c in conditions if c.condition_id in used],
              'rules': rules, 'suppressed_rules': suppressed, 'construction_case_ids': construction_ids,
              'ambiguity': 'Multiple editing actions may be possible under identical measured conditions.',
              'termination': 'Configured processing/search budgets, not theory saturation.'}
    RuleExecutor(theory)
    return theory


def construct_theory(boundaries, groups, conditions, observations, config, backend=None, cache=None):
    pool = candidate_rule_pool(boundaries, groups, conditions, observations, config)
    if backend is None:
        # Explicit synthetic/no-model support; production orchestration requires a backend.
        pool['generation_mode'] = 'python_candidate_pool_only'
        return pool
    from .predictive_theory import build_predictive_theory
    return build_predictive_theory(pool, groups, conditions, observations, config, backend, cache)
