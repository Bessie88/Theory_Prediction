"""Preserve a valid global draft and repair only unassigned source-rule accounting."""
from copy import deepcopy
import logging
from pathlib import Path

from theory_discovery.induction.hierarchical_theory import GLOBAL_TASK,VERSION,_decode,_source_view
from theory_discovery.core.llm import checked_infer,PLAIN_LANGUAGE_INSTRUCTION
from theory_discovery.core.schemas import require
from theory_discovery.core.storage import read_json,write_json
from theory_discovery.core.response_recovery import recoverable_cached_call

LOGGER=logging.getLogger(__name__)
SOURCE_REPAIR_TASK=(
    'Read the whole accepted draft and all missing rules. Place each missing rule into an existing '
    'proposition, add a distinct proposition, or exclude it with a reason. Preserve accepted sources, '
    'conditions, boundaries and meaningful differences. Do not regenerate the draft, invent evidence '
    'or force one action or a fixed theory count. Return only this patch: '
    '{"updates":[{"branch_index":int,"source_rule_ids":[str],"situation":str,'
    '"interpretation":str,"boundary_text":str}],'
    '"new_branches":[{"situation":str,"interpretation":str,"boundary_text":str,'
    '"source_rule_ids":[str],"contrast_rule_ids":[str],"uncertainty":[str]}],'
    '"excluded_rules":[{"rule_id":str,"reason":str}],"open_questions":[str],'
    '"account_addendum":str}. '
    'branch_index is zero-based. Cite the missing rule IDs being placed; code retains existing sources. '
    'Omit unchanged optional prose. Account for every missing rule. '+PLAIN_LANGUAGE_INSTRUCTION)


def _refs(value):
    if isinstance(value,str):return [value] if value.strip() else []
    require(isinstance(value,list) and all(isinstance(v,str) for v in value),'Invalid patch sources')
    return list(dict.fromkeys(value))


def apply_source_patch(raw,draft,allowed):
    require(isinstance(raw,dict),'Missing source-accounting patch')
    missing=set(draft['unassigned_rule_ids'])
    result=deepcopy(draft);result.pop('unassigned_rule_ids',None)
    updates=raw.get('updates') or [];require(isinstance(updates,list),'Invalid theory patch updates')
    touched=set()
    for row in updates:
        require(isinstance(row,dict),'Invalid patch update')
        index=row.get('branch_index')
        if isinstance(index,str) and index.isdigit():index=int(index)
        require(type(index) is int and 0<=index<len(draft['branches'])  ,
                'Unknown patch branch index')
        refs=_refs(row.get('source_rule_ids'))
        require(refs and set(refs)<=allowed,'Patch cites an unknown source')
        target=result['branches'][index]
        target['source_rule_ids']=list(dict.fromkeys(target['source_rule_ids']+refs))
        for key in ('situation','interpretation','boundary_text'):
            value=row.get(key)
            if isinstance(value,str) and value.strip():target[key]=value
        touched.add(index)
    new=raw.get('new_branches') or []
    require(isinstance(new,list),'Invalid new theory propositions')
    for row in new:
        require(isinstance(row,dict),'Invalid new proposition')
        refs=_refs(row.get('source_rule_ids'))
        require(refs and set(refs)<=allowed,'New proposition cites an unknown source')
    result['branches'].extend(new)
    excluded=raw.get('excluded_rules') or []
    require(isinstance(excluded,list),'Invalid patched exclusions')
    for row in excluded:
        require(isinstance(row,dict) and row.get('rule_id') in missing,
                'Patch excludes an already assigned or unknown source')
    result['excluded_rules'].extend(excluded)
    questions=raw.get('open_questions') or []
    if isinstance(questions,str):questions=[questions]
    require(isinstance(questions,list) and all(isinstance(q,str) for q in questions),'Invalid patch questions')
    result['open_questions']=list(dict.fromkeys(result['open_questions']+questions))
    addendum=raw.get('account_addendum')
    if isinstance(addendum,str) and addendum.strip():
        result['account']+='\n\n'+addendum
    # Final strict check still requires ALL source IDs. No missing rule is silently dropped.
    return _decode(result,allowed)


def _historical_draft(cache,payload,allowed):
    best=None;path=None
    for candidate in (cache.directory/'unresolved').glob('*.json'):
        item=read_json(candidate)
        if not str(item.get('task','')).startswith(GLOBAL_TASK):continue
        original=deepcopy(item.get('payload',{}))
        # These retry instructions changed formatting only; source inputs are identical.
        original.pop('validation_feedback',None)
        original.pop('output_repair',None)
        if original!=payload:continue
        try:draft=_decode(item['response'],allowed,allow_unassigned=True)
        except (ValueError,TypeError,KeyError,IndexError):continue
        if not draft['branches']:continue
        if best is None or len(draft['unassigned_rule_ids'])<len(best['unassigned_rule_ids']):
            best,path=draft,str(candidate)
    return best,path


def complete_global_theory(backend,cache,payload,allowed,source,*,output=None):
    """One full global draft, followed only by an explicit bounded missing-source patch."""
    provenance={}
    def initial():
        draft,path=_historical_draft(cache,payload,allowed)
        if draft is not None:
            provenance['recovered_from_unresolved']=path
            LOGGER.info('Recovered global draft: %d propositions, %d/%d sources assigned',
                        len(draft['branches']),len(allowed)-len(draft['unassigned_rule_ids']),len(allowed))
            return draft
        return checked_infer(backend,GLOBAL_TASK,payload,
                             lambda raw:_decode(raw,allowed,allow_unassigned=True))
    draft=cache.get_or_compute('hierarchical_global_draft_recovery',
        {'version':VERSION,'task':GLOBAL_TASK,'payload':payload,'backend':backend.fingerprint},
        initial)
    draft=_decode(draft,allowed,allow_unassigned=True)
    missing=draft['unassigned_rule_ids']
    if output is not None:
        write_json(Path(output)/'global_draft_recovered.json',draft)
        write_json(Path(output)/'global_recovery_progress.json',
            {'stage':'source_accounting_repair' if missing else 'global_complete',
             'draft_propositions':len(draft['branches']),'assigned_sources':len(allowed)-len(missing),
             'unassigned_sources':len(missing),**provenance})
    if not missing:return _decode(draft,allowed)
    patch_payload={'accepted_global_theory':{k:deepcopy(draft[k]) for k in
                   ('account','branches','excluded_rules','open_questions')},
                   'unassigned_rules':[_source_view(source[rid]) for rid in sorted(missing)]}
    checker=getattr(backend,'prompt_fits',None)
    require(checker is None or checker(SOURCE_REPAIR_TASK,patch_payload,repair_reserve=True),
            'Complete draft plus unassigned sources exceeds repair context; nothing was dropped')
    result=recoverable_cached_call(backend,cache,'hierarchical_global_source_patch',
        'global-source-patch-v1',SOURCE_REPAIR_TASK,patch_payload,
        lambda raw:apply_source_patch(raw,draft,allowed))
    result=_decode(result,allowed)
    if output is not None:
        write_json(Path(output)/'global_recovery_report.json',{
            'status':'complete','original_draft_propositions':len(draft['branches']),
            'original_assigned_sources':len(allowed)-len(missing),'patched_source_count':len(missing),
            'final_propositions':len(result['branches']),'total_source_rules_accounted':len(allowed),
            'full_global_draft_supplied_to_repair':True,'original_cases_or_new_evidence_supplied':False,
            **provenance})
        write_json(Path(output)/'global_recovery_progress.json',
                   {'stage':'global_complete','unassigned_sources':0,'total_source_rules':len(allowed)})
    return result
