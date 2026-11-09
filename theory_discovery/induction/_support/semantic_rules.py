"""Qwen proposes within-group rules and cross-group boundaries from open codes."""
from collections import Counter
import logging
from .action_grouping import has_action
from .condition_comparison import evidence_statistics
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION, checked_infer
from .rule_executor import condition_ids, evaluate, validate_expr
from theory_discovery.core.schemas import require, stable_id, to_jsonable
from .scoped_conditions import assess_conditions, refresh_statistics

VERSION = 'qwen-group-first-local-conditions-v6'
BOUNDARY_VERSION = 'all-own-supports-all-matched-outside-v1'
INDUCTION_TASK = """Use only this group's open codes to write reusable natural-language condition-action rules. Explain shared conditions, meaningful differences and alternative situations. Infer no motives or unseen facts; return no rule when unsupported.
Return {"rules":[{"rule_text":str,"rationale":str,"supporting_case_ids":[str],"contradicting_case_ids":[str]}],"comparison_memo":str,"unresolved_questions":[str]}. Cite only supplied group cases.""" + '\n' + PLAIN_LANGUAGE_INSTRUCTION
RULE_FORMAT_TASK = """Convert these already-written natural-language rules into the supplied condition format. Preserve their meaning; do not induce new rules or add requirements. Use the supplied condition definitions and approximate equivalent wording.
Return {"rules":[{"rule_id":str,"conditions":DSL}]}. Return every supplied rule ID once. DSL: pred(condition_id), and/or(args), not(arg), eq/ne/gt/gte/lt/lte(condition_id,value). Use registered condition IDs only. Use pred for boolean features; use value comparisons for numeric or string features.""" + '\n' + PLAIN_LANGUAGE_INSTRUCTION
SIMILAR_CONTEXT_TASK = """Find every supplied other-group case with approximately the same condition TYPE as the target rule. Wording, entities, values and full combinations may differ; such differences can reveal boundaries. A shared topic alone is insufficient. Do not sample matches.
Return {"similar_case_ids":[str],"reason":str}.""" + '\n' + PLAIN_LANGUAGE_INSTRUCTION
BOUNDARY_TASK = """Use all selected other-group open codes with similar condition TYPES to find the boundaries of the rule already formed within this group. Compare the editing methods and add or revise a concise boundary if supported; do not re-induce or replace the original group rule. Conditions need not match exactly. Consider the previous boundary, if any, together with this next batch; keep uncertainty when a distinction is unsupported.
Return {"proposals":[{"expression":DSL,"supporting_case_ids":[str],"contrasting_case_ids":[str],"rationale":str}],"unresolved_questions":[str]}. Return at most one combined boundary, or none to keep the original rule without a boundary. It may combine supplied conditions with pred/and/or/not/eq/ne/gt/gte/lt/lte. Cite only supplied case IDs; no new facts or predicates. A case can belong to both roles when groups overlap; compare its actions and do not treat group membership alone as a contradiction. Read all supplied own-group supports and outside cases; context batches do not sample either side. No separate review follows this comparison.""" + '\n' + PLAIN_LANGUAGE_INSTRUCTION


def code_packet(cases, conditions, extra_ids=()):
    known={c.condition_id for c in conditions}
    selected=(set(extra_ids)|{cid for o in cases for cid in o.candidate_conditions})&known
    return {'conditions':[{'condition_id':c.condition_id,'name':c.name,'definition':c.definition,'value_type':c.value_type} for c in conditions if c.condition_id in selected],
            'open_codes':[{'case_id':o.case_id,'actions':[{k:getattr(a,k) for k in ('family','method','target_feature','transformation','scope')} for a in o.actions],
                'condition_values':{cid:o.condition_values[cid] for cid in sorted(selected) if cid in o.condition_values},
                'candidate_condition_ids':[cid for cid in o.candidate_conditions if cid in selected]} for o in cases]}


def _strings(value,message):
    require(isinstance(value,list) and all(isinstance(x,str) for x in value),message)
    return value


def _call(backend,cache,stage,task,payload,decode,analysis_context=None):
    def compute():
        if hasattr(backend,'reserve_work_calls'):backend.reserve_work_calls((backend.config.retries+1)**2,stage)
        return checked_infer(backend,task,payload,decode)
    if cache is None:return compute()
    return cache.get_or_compute(stage,{'version':BOUNDARY_VERSION if stage=='semantic_boundary_packets' else VERSION,'task':task,'payload':payload,'backend':backend.fingerprint,'analysis_context':analysis_context},compute)


def decode_induction(raw,payload,group,observations,groups,config):
    require(isinstance(raw,dict) and isinstance(raw.get('rules'),list),'Invalid semantic rule induction')
    require(isinstance(raw.get('comparison_memo'),str) and bool(raw['comparison_memo'].strip()),'Missing within-group comparison memo')
    _strings(raw['unresolved_questions'],'Invalid induction unresolved questions')
    known={c['condition_id'] for c in payload['conditions']};shown={c['case_id'] for c in payload['open_codes']}
    hypotheses=[]
    for row in raw['rules']:
        validate_expr(row['conditions'],known)
        require(isinstance(row['rationale'],str) and bool(row['rationale'].strip()),'Induced rule needs comparison rationale')
        stats=evidence_statistics(row['conditions'],group.group_id,observations,groups,group.member_case_ids)
        for key in ('supporting_case_ids','contradicting_case_ids'):
            ids=_strings(row[key],'Invalid induction evidence IDs')
            require(set(ids)<=shown,'Induced rule cites unknown evidence')
        require(bool(row['supporting_case_ids']),'Induced rule needs a supporting open-code case')
        hypotheses.append({'hypothesis_id':stable_id('hypothesis',[VERSION,group.group_id,row['conditions']]),'group_id':group.group_id,'action':group.normalized_action,
            'conditions':row['conditions'],'statistics':stats,'comparison_case_ids':list(group.member_case_ids),'status':'preliminary','rationale':row['rationale'],
            'natural_rule_text':row.get('rule_text',''), 'rule_formatting':'natural_language_then_qwen_DSL',
            'generation_mode':'qwen_open_code_induction',
            'citation_accounting_disagreements':{k:sorted(set(row[k])-set(stats[k])) for k in ('supporting_case_ids','contradicting_case_ids')},
            'model_evidence_citations':{k:row[k] for k in ('supporting_case_ids','contradicting_case_ids')},
            'divergent_member_case_ids':sorted(set(group.member_case_ids)-set(stats['supporting_case_ids']))})
    return {'hypotheses':hypotheses,'memo':raw['comparison_memo'],'unresolved_questions':raw['unresolved_questions']}


def induce_rules(observations,groups,conditions,config,backend,cache=None):
    results=[]
    analysis_context=stable_id('cohort',{'observations':to_jsonable(observations),'groups':to_jsonable(groups)})
    for group in groups:
        members=[o for o in observations if o.case_id in group.member_case_ids]
        frequencies=[]
        own_ids={cid for o in members for cid in o.candidate_conditions}
        for c in conditions:
            if c.condition_id not in own_ids:continue
            values=[o.condition_values.get(c.condition_id) for o in members]
            frequencies.append({'condition_id':c.condition_id,'value_counts':dict(Counter(str(v) for v in values if v is not None)),
                'known_count':sum(v is not None for v in values),'unknown_count':sum(v is None for v in values),
                'observed_in_cases':[o.case_id for o in members if o.condition_values.get(c.condition_id) is not None],'necessary_condition_claimed':False})
        hypotheses,memos,questions,seen_cases={},[],[],[]
        offset=0
        while offset<len(members):
            size=len(members)-offset
            while True:
                chunk=members[offset:offset+size]
                payload={**code_packet(chunk,conditions),'group_id':group.group_id,'action':group.normalized_action,
                    'member_case_ids':[o.case_id for o in chunk],
                    'full_group_size':len(members),
                    'condition_scope':'Per-case observed features and locally assessed values; omitted conditions were not assessed on that case.'}
                fits=getattr(backend,'prompt_fits',None)
                if fits is None or fits(INDUCTION_TASK,payload,repair_reserve=True):break
                require(size>1,'One group open code exceeds context; no evidence was truncated')
                size=max(1,size//2)
            logging.getLogger(__name__).info('Qwen within-group induction: %s, members %d-%d/%d; no outside cases',group.group_id,offset+1,offset+size,len(members))
            def decode_natural(raw):
                require(isinstance(raw,dict) and isinstance(raw.get('rules'),list),'Invalid natural-language rules')
                require(isinstance(raw.get('comparison_memo'),str) and raw['comparison_memo'].strip(),'Missing group comparison memo')
                shown={o.case_id for o in chunk}
                for row in raw['rules']:
                    require(isinstance(row.get('rule_text'),str) and row['rule_text'].strip(),'Missing natural rule text')
                    require(isinstance(row.get('rationale'),str) and row['rationale'].strip(),'Missing rule rationale')
                    for key in ('supporting_case_ids','contradicting_case_ids'):
                        require(set(_strings(row[key],'Invalid rule evidence IDs'))<=shown,'Unknown within-group case')
                    require(bool(row['supporting_case_ids']),'Rule needs a cited group case')
                _strings(raw['unresolved_questions'],'Invalid unresolved questions')
                return raw
            natural=_call(backend,cache,'natural_rule_induction',INDUCTION_TASK,payload,decode_natural,analysis_context)
            natural_rows=[dict(row,rule_id=stable_id('natural_rule',[group.group_id,row['rule_text']])) for row in natural['rules']]
            format_payload={'natural_rules':natural_rows,'conditions':payload['conditions']}
            def decode_format(raw):
                require(isinstance(raw,dict) and isinstance(raw.get('rules'),list),'Invalid rule formatting')
                index={row['rule_id']:row for row in natural_rows}
                returned=[row['rule_id'] for row in raw['rules']]
                require(len(returned)==len(set(returned)) and set(returned)==set(index),'Formatting lost/invented rules')
                combined={**natural,'rules':[dict(index[row['rule_id']],conditions=row['conditions']) for row in raw['rules']]}
                return decode_induction(combined,payload,group,observations,groups,config)
            result=(_call(backend,cache,'rule_format_conversion',RULE_FORMAT_TASK,format_payload,decode_format,analysis_context)
                    if natural_rows else {'hypotheses':[],'memo':natural['comparison_memo'],'unresolved_questions':natural['unresolved_questions']})
            for h in result['hypotheses']:hypotheses.setdefault(h['hypothesis_id'],h)
            memos.append({'member_case_ids':[o.case_id for o in chunk],'comparison_memo':result['memo']})
            questions.extend(result['unresolved_questions']);seen_cases.extend(o.case_id for o in chunk);offset+=size
        used=set().union(*(condition_ids(h['conditions']) for h in hypotheses.values())) if hypotheses else set()
        if used:
            assess_conditions(members,conditions,backend,cache,used,config.workers)
        all_hypotheses=list(hypotheses.values())
        ordered=all_hypotheses
        retained=ordered
        results.append({'group_id':group.group_id,'language_instruction':PLAIN_LANGUAGE_INSTRUCTION,'generation_mode':'qwen_open_code_induction',
            'statistical_enumeration_used':False,'condition_frequencies':frequencies,'alternative_condition_combinations':retained,
            'preliminary_hypotheses':retained,'combination_candidates_examined':len(all_hypotheses),'search_budget_reached':len(ordered)>len(retained),
            'unretained_model_hypotheses':ordered[len(retained):],'comparison_memos':memos,'open_code_case_ids_compared':sorted(set(seen_cases)),
            'unresolved_questions':questions,'note':'Qwen proposed relationships from open codes; Python evaluated full-discovery statistics. Budget-unretained model candidates remain recorded.'})
    refresh_statistics(results,[],observations,groups)
    return results


def decode_boundaries(raw,payload,supports,counters):
    require(isinstance(raw,dict) and isinstance(raw.get('proposals'),list) and len(raw['proposals'])<=1,'Invalid semantic boundary proposals')
    known={c['condition_id'] for c in payload['conditions']};shown={c['case_id'] for c in payload['open_codes']}|set(payload.get('previously_compared_case_ids',[]))
    support_index={o.case_id:o for o in supports};counter_index={o.case_id:o for o in counters};output=[]
    for row in raw['proposals']:
        validate_expr(row['expression'],known)
        a=_strings(row['supporting_case_ids'],'Invalid boundary support');b=_strings(row['contrasting_case_ids'],'Invalid boundary contrasts')
        require(bool(a) and bool(b) and set(a)<=shown&set(support_index) and set(b)<=shown&set(counter_index),'Invented boundary support/contrast')
        require(isinstance(row['rationale'],str) and bool(row['rationale'].strip()),'Boundary needs comparison rationale')
        output.append(row)
    return output


def _condition_description(expr,index):
    op=expr['op']
    if op in ('and','or'):
        return '('+(' '+op.upper()+' ').join(_condition_description(a,index) for a in expr['args'])+')'
    if op=='not':return 'NOT ('+_condition_description(expr['arg'],index)+')'
    definition=index[expr['condition_id']].definition
    return definition if op=='pred' else definition+' '+op+' '+str(expr.get('value'))


def find_similar_contexts(backend,cache,hypothesis,comparisons,observations,groups,conditions):
    """Scan condition profiles only; keep every semantic match without a top-k cap."""
    index={c.condition_id:c for c in conditions}
    target=next(g for g in groups if g.group_id==hypothesis['group_id'])
    other_memberships={}
    for group in groups:
        if group.group_id==target.group_id:continue
        for cid in group.member_case_ids:
            other_memberships.setdefault(cid,[]).append(group.group_id)
    outside=[o for o in observations if o.case_id in other_memberships]
    target_ids=condition_ids(hypothesis['conditions'])
    target_view={'rule_text':hypothesis.get('natural_rule_text',''),
                 'condition_description':_condition_description(hypothesis['conditions'],index)}
    selected=[];decisions=[];offset=0
    while offset<len(outside):
        size=len(outside)-offset
        while True:
            chunk=outside[offset:offset+size]
            used=set(target_ids)|{cid for o in chunk for cid in o.candidate_conditions}
            payload={'target_context':target_view,
                'condition_definitions':[{'condition_id':cid,'definition':index[cid].definition,'value_type':index[cid].value_type} for cid in sorted(used)],
                'outside_condition_profiles':[{'case_id':o.case_id,'other_group_ids':other_memberships[o.case_id],
                    'also_in_target_group':o.case_id in target.member_case_ids,
                    'candidate_condition_ids':list(o.candidate_conditions),
                    'observed_conditions':{cid:o.condition_values[cid] for cid in sorted(set(o.candidate_conditions)|target_ids) if cid in o.condition_values}}
                    for o in chunk]}
            fits=getattr(backend,'prompt_fits',None)
            if fits is None or fits(SIMILAR_CONTEXT_TASK,payload,repair_reserve=True):break
            require(size>1,'One outside condition profile exceeds context; no profile was truncated')
            size=max(1,size//2)
        def decode(raw):
            ids=_strings(raw['similar_case_ids'],'Invalid similar-context IDs')
            require(len(ids)==len(set(ids)) and set(ids)<={o.case_id for o in chunk},'Unknown/duplicate similar-context case')
            require(isinstance(raw.get('reason'),str) and bool(raw['reason'].strip()),'Context match needs a reason')
            return raw
        result=_call(backend,cache,'similar_outside_condition_profiles',SIMILAR_CONTEXT_TASK,payload,decode)
        wanted=set(result['similar_case_ids'])
        selected.extend(o for o in chunk if o.case_id in wanted)
        decisions.append({'considered_case_ids':[o.case_id for o in chunk],**result});offset+=size
    return selected,{'selected_case_ids':[o.case_id for o in selected],
        'outside_condition_profiles_considered':len(outside),'context_match_decisions':decisions,
        'fixed_case_limit':None,'match_policy':'qwen_all_approximately_similar_condition_profiles',
        'outside_group_ids_by_case':{o.case_id:other_memberships[o.case_id] for o in selected},
        'shared_with_target_case_ids':[o.case_id for o in selected if o.case_id in target.member_case_ids],
        'candidate_group_policy':'any_other_group_including_overlapping_membership'}


def propose_boundaries(backend,cache,hypothesis,supports,counters,conditions,packet_history=None):
    if not supports or not counters:return []
    # Visit every case on both sides; cycle the shorter side only to keep each
    # context window comparative. This is not a Cartesian case-pair enumeration.
    pairs=[(supports[i%len(supports)],counters[i%len(counters)])
           for i in range(max(len(supports),len(counters)))]
    offset=0;current=[];seen_supports=[];seen_outside=[];all_rows=[]
    while offset<len(pairs):
        size=len(pairs)-offset
        while True:
            window=pairs[offset:offset+size]
            support_chunk=list({o.case_id:o for o,c in window}.values())
            outside_chunk=list({c.case_id:c for o,c in window}.values())
            inherited=set(condition_ids(hypothesis['conditions']))
            for row in current:inherited.update(condition_ids(row['expression']))
            previously=list(dict.fromkeys(seen_supports+seen_outside))
            payload={**code_packet(list({o.case_id:o for o in support_chunk+outside_chunk}.values()),conditions,inherited),
                'hypothesis':{'hypothesis_id':hypothesis['hypothesis_id'],'group_id':hypothesis['group_id'],
                    'natural_rule_text':hypothesis.get('natural_rule_text',''),
                    'conditions':hypothesis['conditions'],'action':hypothesis.get('action'),'rationale':hypothesis.get('rationale','')},
                'previous_boundary':current[0] if current else None,
                'previously_compared_case_ids':previously,
                'outside_group_ids_by_case':hypothesis.get('outside_group_ids_by_case',{}),
                'supporting_case_ids':[o.case_id for o in support_chunk],
                'counterexample_case_ids':[o.case_id for o in outside_chunk],
                'comparison_basis':'All relevant own-group supports and all selected similar-condition-type outside cases are compared across batches, without sampling; exact logical matches are not required.'}
            fits=getattr(backend,'prompt_fits',None)
            if fits is None or fits(BOUNDARY_TASK,payload,repair_reserve=True):break
            require(size>1,'One support/outside comparison exceeds context; no case evidence was truncated')
            size=max(1,size//2)
        logging.getLogger(__name__).info('Qwen boundary comparison: %s, window %d-%d/%d; %d own supports and %d outside cases',
            hypothesis['hypothesis_id'],offset+1,offset+size,len(pairs),len(support_chunk),len(outside_chunk))
        rows=_call(backend,cache,'semantic_boundary_packets',BOUNDARY_TASK,payload,
            lambda raw:decode_boundaries(raw,payload,supports,counters))
        record={'supporting_case_ids':payload['supporting_case_ids'],
                'outside_case_ids':payload['counterexample_case_ids'],
                'previous_boundary_present':bool(current)}
        if packet_history is not None:packet_history.append(record)
        current=rows;all_rows.extend(rows)
        seen_supports.extend(record['supporting_case_ids']);seen_outside.extend(record['outside_case_ids'])
        offset+=size
    require(set(seen_supports)=={o.case_id for o in supports},'Boundary comparison missed own-group support evidence')
    require(set(seen_outside)=={o.case_id for o in counters},'Boundary comparison missed selected outside evidence')
    if current:
        current=[dict(current[0],comparison_support_case_ids=list(dict.fromkeys(seen_supports)),
            comparison_case_ids=list(dict.fromkeys(seen_outside)),earlier_boundary_proposals=all_rows[:-1])]
    return current
