"""Before-only condition values evaluated within explicit local case scopes."""
from concurrent.futures import ThreadPoolExecutor
import logging
from .conditions import ConditionDetector
from .schemas import BeforeContext,require
VERSION='group-first-scoped-values-v1'

def assess_conditions(cases,conditions,backend,cache,requested_ids=None,workers=1):
    """None selects each case's own candidates; explicit IDs apply only to supplied cases."""
    index={c.condition_id:c for c in conditions}
    unique={o.case_id:o for o in cases}
    detector=ConditionDetector(backend)
    if backend is not None and hasattr(backend,'reserve_work_calls'):
        batch_size=backend.config.condition_batch_size
        batches=0
        for obs in unique.values():
            wanted=set(obs.candidate_conditions if requested_ids is None else requested_ids)
            missing=[cid for cid in wanted-set(obs.condition_values) if cid in index and index[cid].detector['kind']=='semantic']
            batches+=(len(missing)+batch_size-1)//batch_size
        backend.reserve_work_calls(batches*(backend.config.retries+1)**2,'local_condition_assessments')
    def assess(obs):
        wanted=set(obs.candidate_conditions if requested_ids is None else requested_ids)
        require(wanted<=set(index),'Unknown scoped condition reference')
        missing=sorted(wanted-set(obs.condition_values))
        if not missing:return obs,{},{}
        selected=[index[cid] for cid in missing]
        context=BeforeContext(obs.before_text,obs.pre_context)
        view=[{'condition_id':c.condition_id,'definition':c.definition,'value_type':c.value_type,'detector':c.detector} for c in selected]
        def compute():return detector.detect_with_evidence(context,selected,obs.case_id)
        if cache is None:values,evidence=compute()
        else:values,evidence=cache.get_or_compute('scoped_condition_values',
            {'version':VERSION,'case_id':obs.case_id,'context':context.to_dict(),'conditions':view,
             'backend':backend.fingerprint if backend else 'mechanical-only-v1'},compute)
        return obs,values,evidence
    with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        results=list(pool.map(assess,unique.values()))
    for obs,values,evidence in results:
        obs.condition_values.update(values);obs.condition_evidence.update(evidence)
    logging.getLogger(__name__).info('Scoped condition assessment: %d cases, %d newly assessed values',
        len(unique),sum(len(values) for obs,values,evidence in results))
    return sum(len(values) for obs,values,evidence in results)

def refresh_statistics(comparisons,boundaries,observations,groups):
    """Refresh descriptive counts after local assessments; no new model calls."""
    from .condition_comparison import evidence_statistics
    from .rule_executor import evaluate
    hypotheses={h['hypothesis_id']:h for c in comparisons for h in c['preliminary_hypotheses']}
    for h in hypotheses.values():
        members=next(g for g in groups if g.group_id==h['group_id']).member_case_ids
        relevant=set(members)
        boundary=next((b for b in boundaries if b['hypothesis_id']==h['hypothesis_id']),None)
        if boundary:relevant.update(boundary.get('context_retrieval',{}).get('selected_case_ids',[]))
        h['comparison_case_ids']=sorted(relevant)
        h['statistics']=evidence_statistics(h['conditions'],h['group_id'],observations,groups,relevant)
        h['divergent_member_case_ids']=sorted(set(next(g for g in groups if g.group_id==h['group_id']).member_case_ids)-set(h['statistics']['supporting_case_ids']))
        citations=h.get('model_evidence_citations',{})
        h['citation_accounting_disagreements']={k:sorted(set(citations.get(k,[]))-set(h['statistics'][k])) for k in ('supporting_case_ids','contradicting_case_ids')}
    for row in boundaries:
        original=hypotheses[row['hypothesis_id']]['statistics']
        row['original_statistics']=original
        for rev in row['candidate_boundaries']:
            stats=evidence_statistics(rev['revised_conditions'],row['group_id'],observations,groups,original['compared_case_ids'])
            rev['statistics']=stats
            rev['frequency_gain']=(stats['conditional_action_frequency'] or 0)-(original['conditional_action_frequency'] or 0)
            rev['retained_support_fraction']=stats['support_count']/max(1,original['support_count'])
            rev['removed_supporting_case_ids']=sorted(set(original['supporting_case_ids'])-set(stats['supporting_case_ids']))
            compared={x['case_id'] for x in row['cross_group_counterexamples']}
            rev['excluded_counterexample_case_ids']=[o.case_id for o in observations if o.case_id in compared and evaluate(rev['boundary'],o.condition_values) is False]
        selected=row['selected_revision']
        if selected:
            selected=next(r for r in row['candidate_boundaries'] if r['revision_id']==selected['revision_id'])
            row['selected_revision']=selected
        final=selected['statistics'] if selected else original
        row['unresolved_case_ids']=sorted(set(final['contradicting_case_ids']+final['unknown_case_ids']))
        row['ambiguity']=bool(final['contradicting_case_ids'])
