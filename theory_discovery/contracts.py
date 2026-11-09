"""Cross-stage consistency checks before artifacts are committed."""
from .rule_executor import effective_expression, validate_expr, condition_ids
from .schemas import require


def validate_artifacts(observations, groups, conditions, comparisons, boundaries, theory):
    ids = {o.case_id for o in observations}
    require(len(ids) == len(observations), 'Duplicate observation IDs')
    condition_set = {c.condition_id for c in conditions}
    require(len(condition_set) == len(conditions), 'Duplicate condition IDs')
    actions = {(o.case_id, a.action_id) for o in observations for a in o.actions}
    group_ids = {g.group_id for g in groups}
    require(len(group_ids) == len(groups), 'Duplicate action group IDs')
    for obs in observations:
        require(set(obs.condition_values) <= condition_set, 'Scoped condition values reference unknown columns')
        require(set(obs.candidate_conditions) <= condition_set, 'Unknown observation predicate')
    assigned = set()
    for group in groups:
        require(set(group.member_case_ids) <= ids and group.group_size == len(group.member_case_ids), 'Invalid group membership')
        for ref in group.action_refs:
            key = (ref['case_id'], ref['action_id'])
            require(key in actions and ref['case_id'] in group.member_case_ids, 'Invalid source action reference')
            assigned.add(key)
    require(actions == assigned, 'Action grouping dropped original actions')
    hypotheses = set()
    for comparison in comparisons:
        require(comparison['group_id'] in group_ids, 'Comparison references unknown group')
        for hypothesis in comparison['preliminary_hypotheses']:
            hypotheses.add(hypothesis['hypothesis_id'])
            validate_expr(hypothesis['conditions'], condition_set)
            _statistics(hypothesis['statistics'], ids)
            if hypothesis.get('generation_mode')=='qwen_open_code_induction':
                members=set(next(g for g in groups if g.group_id==hypothesis['group_id']).member_case_ids)
                used=condition_ids(hypothesis['conditions'])
                require(all(used<=set(o.condition_values) for o in observations if o.case_id in members),
                        'A group rule has unassessed required conditions within its own group')
    for boundary in boundaries:
        require(boundary['hypothesis_id'] in hypotheses, 'Unknown boundary hypothesis')
        _statistics(boundary['original_statistics'], ids)
        if boundary.get('selection_source')=='qwen_single_comparison':
            members=set(next(g for g in groups if g.group_id==boundary['group_id']).member_case_ids)
            members.update(boundary['context_retrieval']['selected_case_ids'])
            used=condition_ids(boundary['original_conditions'])
            if boundary['selected_revision']:used.update(condition_ids(boundary['selected_revision']['boundary']))
            require(all(used<=set(o.condition_values) for o in observations if o.case_id in members),
                    'Boundary comparison has unassessed required conditions on a relevant case')
        for revision in boundary['candidate_boundaries']:
            validate_expr(revision['revised_conditions'], condition_set)
            _statistics(revision['statistics'], ids)
    for rule in theory['rules']:
        require(rule['action']['group_id'] in group_ids, 'Unknown theory action')
        validate_expr(effective_expression(rule), condition_set)
        _statistics(rule['evidence_statistics'], ids)
        require(set(rule['construction_case_ids']) <= ids, 'Invented construction evidence')
        require(set(rule['supporting_case_ids']) == set(rule['evidence_statistics']['supporting_case_ids']), 'Inconsistent theory support')
    if theory.get('generation_mode') == 'rule_grounded_predictive_theory':
        from .predictive_theory import validate_predictive_theory
        from .condition_comparison import evidence_statistics
        validate_predictive_theory(theory)
        for branch in theory['branches']:
            actual = evidence_statistics(branch['when'], branch['action']['group_id'], observations, groups,branch['evidence_statistics'].get('compared_case_ids'))
            require(branch['evidence_statistics'] == actual, 'Theory branch evidence counts do not match actual discovery cases')
    return True


def _statistics(stats, known_ids):
    partitions = [set(stats[key]) for key in ('supporting_case_ids', 'contradicting_case_ids', 'unknown_case_ids', 'nonmatching_case_ids')]
    outside=set(stats.get('not_compared_case_ids',[]))
    require(set().union(*partitions,outside) == known_ids, 'Evidence accounting missing/invented cases')
    require(sum(map(len, partitions))+len(outside) == len(known_ids), 'Overlapping evidence partitions')
    require(stats['eligible_cases']==len(known_ids)-len(outside),'Comparison denominator includes unselected cases')
    require(stats['support_count'] == len(partitions[0]) and stats['contradiction_count'] == len(partitions[1]), 'Incorrect support counts')
