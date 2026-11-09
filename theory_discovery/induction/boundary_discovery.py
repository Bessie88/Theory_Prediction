"""Stage 4: counterexamples, evidence-checked boundaries and unresolved ambiguity."""
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION

STAGE_INSTRUCTION = PLAIN_LANGUAGE_INSTRUCTION
from .condition_comparison import candidate_atoms, evidence_statistics
from .rule_executor import complexity, condition_ids, conjunction, evaluate, validate_expr
from theory_discovery.core.schemas import stable_id, require

from .semantic_rules import BOUNDARY_TASK
from .scoped_conditions import assess_conditions, refresh_statistics


class SemanticBoundaryProposer:
    def __init__(self, backend, cache=None, workers=1):
        self.backend, self.cache, self.workers = backend, cache, workers
        self.history = []
        self.comparison_packets = []

    def retrieve(self, hypothesis, comparisons, observations, groups, conditions):
        from .semantic_rules import find_similar_contexts
        return find_similar_contexts(self.backend, self.cache, hypothesis, comparisons, observations, groups, conditions)

    def propose(self, hypothesis, supports, counters, conditions):
        from .semantic_rules import propose_boundaries
        packets=[]
        rows = propose_boundaries(self.backend, self.cache, hypothesis, supports, counters, conditions,packets)
        self.comparison_packets.extend(dict(packet,hypothesis_id=hypothesis['hypothesis_id']) for packet in packets)
        self.history.extend(dict(row, hypothesis_id=hypothesis['hypothesis_id']) for row in rows)
        return [row['expression'] for row in rows]


def discover_boundaries(comparisons, observations, groups, conditions, config, proposer=None):
    """Optional proposer.propose(hypothesis, supports, counters, condition_definitions).

    Returns DSL expressions over registered conditions only. New predicates require
    a local Stage 1 re-extraction and before-only evaluation before reconsideration.
    Qwen reviews proposed expressions against open codes and descriptive full-cohort statistics.
    """
    if isinstance(proposer, SemanticBoundaryProposer):
        return _discover_model_boundaries(comparisons, observations, groups, conditions, proposer)
    result = []
    known_ids = {c.condition_id for c in conditions}
    for comparison in comparisons:
        for hypothesis in comparison['preliminary_hypotheses']:
            original = hypothesis['conditions']
            original_stats = hypothesis['statistics']
            matched_ids = set(original_stats['supporting_case_ids'] + original_stats['contradicting_case_ids'])
            comparable = [o for o in observations if o.case_id in matched_ids]
            cross = [{'case_id': o.case_id,
                      'other_group_ids': [g.group_id for g in groups if o.case_id in g.member_case_ids and g.group_id != hypothesis['group_id']]}
                     for o in comparable if o.case_id in original_stats['contradicting_case_ids']]
            proposals = []
            if cross:
                used = condition_ids(original)
                if proposer is not None:
                    support_cases = [o for o in comparable if o.case_id in original_stats['supporting_case_ids']]
                    counter_cases = [o for o in comparable if o.case_id in original_stats['contradicting_case_ids']]
                    proposals = list(proposer.propose(hypothesis, support_cases, counter_cases, conditions))
                else:
                    # Explicit statistical/synthetic mode only; no fallback after model abstention.
                    proposals = [atom for atom in candidate_atoms(conditions, comparable, config.max_boundary_candidates * 2)
                                 if not condition_ids(atom) <= used]
            revisions = []
            seen = set()
            for boundary in proposals[:config.max_boundary_candidates]:
                validate_expr(boundary, known_ids)
                key = stable_id('boundary', boundary)
                if key in seen:
                    continue
                seen.add(key)
                revised = conjunction([original, boundary])
                stats = evidence_statistics(revised, hypothesis['group_id'], observations, groups)
                removed_support = sorted(set(original_stats['supporting_case_ids']) - set(stats['supporting_case_ids']))
                removed_counter = sorted(set(original_stats['contradicting_case_ids']) - set(stats['contradicting_case_ids']))
                # Unknown detection cannot count as explanatory boundary exclusion.
                explained_counter = [o.case_id for o in comparable if o.case_id in removed_counter and evaluate(boundary, o.condition_values) is False]
                gain = (stats['conditional_action_frequency'] or 0) - (original_stats['conditional_action_frequency'] or 0)
                reasons = []
                if stats['support_count'] < config.min_support:
                    reasons.append('insufficient_retained_support')
                if len(explained_counter) < config.min_support:
                    reasons.append('boundary_not_replicated_across_counterexamples')
                if stats['support_count'] / max(1, original_stats['support_count']) < config.min_retained_support:
                    reasons.append('excessive_coverage_loss')
                if gain < config.min_boundary_gain:
                    reasons.append('insufficient_conditional_frequency_gain')
                if len(condition_ids(revised)) > config.max_condition_size:
                    reasons.append('complexity_budget_exceeded')
                # A boundary must improve action separation in comparable contexts, not just drop unknowns.
                if len(explained_counter) / max(1, original_stats['contradiction_count']) <= len(removed_support) / max(1, original_stats['support_count']):
                    reasons.append('no_differential_action_separation')
                model_proposal = next((row for row in getattr(proposer, 'history', []) if row.get('hypothesis_id') == hypothesis['hypothesis_id'] and row['expression'] == boundary), {})
                revisions.append({'revision_id': key, 'boundary': boundary, 'revised_conditions': revised,
                                  'statistics': stats, 'frequency_gain': gain,
                                  'retained_support_fraction': stats['support_count'] / max(1, original_stats['support_count']),
                                  'removed_supporting_case_ids': removed_support,
                                  'excluded_counterexample_case_ids': explained_counter,
                                  'accepted_for_candidate': not reasons, 'rejection_reasons': reasons,
                                  'numerical_diagnostics': reasons,
                                  'validation_status': 'construction_evidence_only',
                                  'proposal_source': 'qwen_open_codes' if proposer is not None else 'statistical',
                                  'rationale': model_proposal.get('rationale', ''),
                                  'model_evidence_citations': {k: model_proposal.get(k, []) for k in ('supporting_case_ids', 'contrasting_case_ids')}})
            decision_history = []
            accepted = [r for r in revisions if r['accepted_for_candidate']]
            accepted.sort(key=lambda r: (-r['statistics']['conditional_action_frequency'], complexity(r['revised_conditions']), -r['statistics']['support_count']))
            selected = accepted[0] if accepted else None
            final_stats = selected['statistics'] if selected else original_stats
            unresolved = sorted(set(final_stats['contradicting_case_ids'] + final_stats['unknown_case_ids']))
            result.append({'hypothesis_id': hypothesis['hypothesis_id'], 'group_id': hypothesis['group_id'],
                           'original_conditions': original, 'original_statistics': original_stats,
                           'induction_mode': hypothesis.get('generation_mode', 'statistical'),
                           'induction_rationale': hypothesis.get('rationale', ''),
                           'induction_evidence_citations': hypothesis.get('model_evidence_citations', {}),
                           'cross_group_counterexamples': cross, 'candidate_boundaries': revisions,
                           'accepted_revision_ids': [r['revision_id'] for r in accepted],
                           'rejected_revision_ids': [r['revision_id'] for r in revisions if not r['accepted_for_candidate']],
                           'selected_revision': selected, 'unresolved_case_ids': unresolved,
                           'boundary_decision_history': decision_history,
                           'selection_source': 'statistical',
                           'ambiguity': bool(final_stats['contradicting_case_ids']),
                           'status': 'candidate_boundary' if selected else ('unresolved_ambiguity' if cross else 'no_observed_counterexample'),
                           'note': 'Boundary selection on construction evidence is provisional, not independent validation.'})
    return result


def _discover_model_boundaries(comparisons, observations, groups, conditions, proposer):
    """One selected-context comparison per rule, without a separate review pass."""
    result=[];group_index={g.group_id:g for g in groups};known={c.condition_id for c in conditions}
    obs_index={o.case_id:o for o in observations}
    for comparison in comparisons:
        for h in comparison['preliminary_hypotheses']:
            group=group_index[h['group_id']]
            ids=set(h.get('model_evidence_citations',{}).get('supporting_case_ids',[]))
            ids.update(h['statistics']['supporting_case_ids'])
            source_ids=sorted(cid for cid in ids if cid in group.member_case_ids and cid in obs_index)
            if not source_ids:source_ids=sorted(group.member_case_ids)
            supports=[obs_index[cid] for cid in source_ids]
            counters,retrieval=proposer.retrieve(h,comparisons,observations,groups,conditions)
            local_cases={o.case_id:o for o in observations if o.case_id in group.member_case_ids}
            local_cases.update({o.case_id:o for o in counters})
            assess_conditions(list(local_cases.values()),conditions,proposer.backend,proposer.cache,
                              condition_ids(h['conditions']),proposer.workers)
            h['statistics']=evidence_statistics(h['conditions'],h['group_id'],observations,groups,local_cases)
            comparison_hypothesis=dict(h,outside_group_ids_by_case=retrieval.get('outside_group_ids_by_case',{}))
            expressions=proposer.propose(comparison_hypothesis,supports,counters,conditions) if counters else []
            revisions=[]
            for boundary in expressions:
                validate_expr(boundary,known)
                revised=conjunction([h['conditions'],boundary])
                assess_conditions(list(local_cases.values()),conditions,proposer.backend,proposer.cache,
                                  condition_ids(boundary),proposer.workers)
                stats=evidence_statistics(revised,h['group_id'],observations,groups,local_cases)
                original=h['statistics']
                model=next(row for row in reversed(proposer.history) if row['hypothesis_id']==h['hypothesis_id'] and row['expression']==boundary)
                removed_support=sorted(set(original['supporting_case_ids'])-set(stats['supporting_case_ids']))
                excluded=[o.case_id for o in counters if evaluate(boundary,o.condition_values) is False]
                revisions.append({'revision_id':stable_id('boundary',boundary),'boundary':boundary,'revised_conditions':revised,
                    'statistics':stats,'frequency_gain':(stats['conditional_action_frequency'] or 0)-(original['conditional_action_frequency'] or 0),
                    'retained_support_fraction':stats['support_count']/max(1,original['support_count']),
                    'removed_supporting_case_ids':removed_support,'excluded_counterexample_case_ids':excluded,
                    'accepted_for_candidate':True,'rejection_reasons':[],
                    'validation_status':'construction_evidence_only','proposal_source':'qwen_similar_context_comparison',
                    'selection_source':'qwen_single_comparison','rationale':model['rationale'],
                    'model_evidence_citations':{k:model.get(k,[]) for k in ('supporting_case_ids','contrasting_case_ids')}})
            selected=revisions[0] if revisions else None
            stats=selected['statistics'] if selected else h['statistics']
            result.append({'hypothesis_id':h['hypothesis_id'],'group_id':h['group_id'],
                'original_conditions':h['conditions'],'original_statistics':h['statistics'],
                'induction_mode':h.get('generation_mode','statistical'),'induction_rationale':h.get('rationale',''),
                'induction_evidence_citations':h.get('model_evidence_citations',{}),
                'natural_rule_text':h.get('natural_rule_text',''),
                'cross_group_counterexamples':[{'case_id':o.case_id,
                    'other_group_ids':[g.group_id for g in groups if g.group_id!=h['group_id'] and o.case_id in g.member_case_ids]} for o in counters],
                'context_retrieval':retrieval,
                'available_support_case_ids':source_ids,
                'compared_support_case_ids':source_ids if counters else [],
                'boundary_comparison_packets':[p for p in proposer.comparison_packets if p['hypothesis_id']==h['hypothesis_id']],
                'boundary_support_sampling':False,
                'candidate_boundaries':revisions,'accepted_revision_ids':[r['revision_id'] for r in revisions],
                'rejected_revision_ids':[],'selected_revision':selected,
                'unresolved_case_ids':sorted(set(stats['contradicting_case_ids']+stats['unknown_case_ids'])),
                'boundary_decision_history':[],'selection_source':'qwen_single_comparison',
                'ambiguity':bool(stats['contradicting_case_ids']),
                'status':'candidate_boundary' if selected else ('unresolved_ambiguity' if counters else 'no_similar_outside_context'),
                'note':'All relevant own-group supports and all Qwen-selected approximately similar outside cases were compared across context-bounded batches. No independent or exhaustive validation.'})
    refresh_statistics(comparisons,result,observations,groups)
    return result


def relevant_observations(expression, observations, group):
    """Local refinement retrieval: matches/unknowns plus all target action cases."""
    return [o for o in observations if o.case_id in group.member_case_ids or evaluate(expression, o.condition_values) is not False]
