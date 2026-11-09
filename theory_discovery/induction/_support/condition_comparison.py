"""Stage 3: condition recurrence, alternatives, compact provisional hypotheses."""
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION

STAGE_INSTRUCTION = PLAIN_LANGUAGE_INSTRUCTION
from collections import Counter
from itertools import combinations
from .action_grouping import has_action
from .rule_executor import conjunction, evaluate, predicate
from theory_discovery.core.schemas import stable_id


def evidence_statistics(expression, group_id, observations, groups, comparison_case_ids=None):
    group = next(g for g in groups if g.group_id == group_id)
    all_ids={o.case_id for o in observations}
    compared=all_ids if comparison_case_ids is None else set(comparison_case_ids)
    if not compared<=all_ids:raise ValueError('Unknown comparison case ID')
    outside=sorted(all_ids-compared)
    observations=[o for o in observations if o.case_id in compared]
    support, contradictions, unknown, nonmatches = [], [], [], []
    target_ids = []
    for obs in observations:
        target = has_action(obs, group)
        if target:
            target_ids.append(obs.case_id)
        result = evaluate(expression, obs.condition_values)
        if result is None:
            unknown.append(obs.case_id)
        elif result is False:
            nonmatches.append(obs.case_id)
        elif target:
            support.append(obs.case_id)
        else:
            contradictions.append(obs.case_id)
    matched = len(support) + len(contradictions)
    return {'supporting_case_ids': sorted(support), 'contradicting_case_ids': sorted(contradictions),
            'unknown_case_ids': sorted(unknown), 'nonmatching_case_ids': sorted(nonmatches),
            'matched_cases': matched, 'support_count': len(support), 'contradiction_count': len(contradictions),
            'eligible_cases': len(observations), 'known_cases': len(observations) - len(unknown),
            'conditional_action_frequency': len(support) / matched if matched else None,
            'coverage': len(support) / len(target_ids) if target_ids else 0,
            'population_match_coverage': matched / len(observations) if observations else 0,
            'compared_case_ids':sorted(compared),'not_compared_case_ids':outside}


def candidate_atoms(conditions, observations, limit):
    """Observed boolean literals and bounded numeric thresholds; no closed taxonomy."""
    atoms = []
    for condition in conditions:
        cid = condition.condition_id
        values = [o.condition_values.get(cid) for o in observations]
        if condition.value_type == 'boolean':
            if any(v is True for v in values):
                atoms.append(predicate(cid))
            if any(v is False for v in values):
                atoms.append({'op': 'not', 'arg': predicate(cid)})
        elif condition.value_type == 'number':
            thresholds = sorted({v for v in values if type(v) in (int, float)})
            # Quantiles of observed values, never case IDs or arbitrary generated thresholds.
            if len(thresholds) > 8:
                thresholds = sorted({thresholds[round(i * (len(thresholds) - 1) / 7)] for i in range(8)})
            atoms.extend({'op': op, 'condition_id': cid, 'value': v} for v in thresholds for op in ('gte', 'lte'))
        else:
            categories = sorted({v for v in values if isinstance(v, str)})[:8]
            atoms.extend({'op': 'eq', 'condition_id': cid, 'value': v} for v in categories)
        if len(atoms) >= limit:
            break
    return atoms[:limit]


def compare_conditions_statistical(observations, groups, conditions, config):
    results = []
    for group in groups:
        members = [o for o in observations if o.case_id in group.member_case_ids]
        frequencies = []
        for condition in conditions:
            values = [o.condition_values.get(condition.condition_id) for o in members]
            counts = Counter(str(v) for v in values if v is not None)
            frequencies.append({'condition_id': condition.condition_id, 'value_counts': dict(counts),
                                'known_count': sum(v is not None for v in values),
                                'unknown_count': sum(v is None for v in values),
                                'observed_in_cases': [o.case_id for o in members if o.condition_values.get(condition.condition_id) is not None],
                                'necessary_condition_claimed': False})
        atoms = candidate_atoms(conditions, members, config.max_candidates_per_group)
        hypotheses, combinations_seen = [], 0
        budget_reached = False
        for size in range(1, min(config.max_condition_size, len(atoms)) + 1):
            for combo in combinations(atoms, size):
                if combinations_seen >= config.max_candidates_per_group:
                    budget_reached = True
                    break
                combinations_seen += 1
                expression = conjunction(combo)
                stats = evidence_statistics(expression, group.group_id, observations, groups)
                if stats['support_count'] < config.min_support:
                    continue
                hid = stable_id('hypothesis', [group.group_id, expression])
                hypotheses.append({'hypothesis_id': hid, 'group_id': group.group_id,
                                   'conditions': expression, 'statistics': stats, 'status': 'preliminary',
                                   'divergent_member_case_ids': sorted(set(group.member_case_ids) - set(stats['supporting_case_ids']))})
            if budget_reached:
                break
        results.append({'group_id': group.group_id, 'language_instruction': STAGE_INSTRUCTION, 'condition_frequencies': frequencies,
                        'alternative_condition_combinations': hypotheses,
                        'preliminary_hypotheses': hypotheses,
                        'combination_candidates_examined': combinations_seen,
                        'search_budget_reached': budget_reached,
                        'note': 'Recurrence is not necessity; frequency estimates are construction-set descriptions.'})
    return results


def compare_conditions(observations, groups, conditions, config, backend=None, cache=None):
    if backend is not None and config.rule_induction_mode == 'qwen_open_codes':
        from .semantic_rules import induce_rules
        return induce_rules(observations, groups, conditions, config, backend, cache)
    return compare_conditions_statistical(observations, groups, conditions, config)
