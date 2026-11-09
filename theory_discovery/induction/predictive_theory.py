"""Evidence-grounded predictive theory assembled from immutable discovered rules."""
from collections import defaultdict, deque
import logging
from .condition_comparison import evidence_statistics
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION, checked_infer
from .rule_executor import condition_ids, effective_expression, validate_expr
from theory_discovery.core.schemas import require, stable_id, to_jsonable
from .scoped_conditions import assess_conditions

VERSION = 'rule-open-code-predictive-theory-v5-local-condition-evidence'
THEORY_TASK = """Read these rules, their boundaries and selected supporting/contrasting open codes. Combine related rules when that gives a useful account of which editing action may occur in different situations. Do not force unrelated rules together. When no faithful fusion is possible, keep each useful rule as its own predictive statement. Retain uncertainty and do not invent new facts, intentions, conditions or actions.
Return {"branches":[{"situation":str,"source_rule_ids":[str],"interpretation":str,"contrast_rule_ids":[str],"supporting_case_ids":[str],"contradicting_case_ids":[str],"uncertainty":[str]}],"excluded_rules":[{"rule_id":str,"reason":str}],"open_questions":[str]}. Be concise. A branch may combine related rules for the same action, or retain one rule by itself. Keep each supplied useful rule standalone or in a faithful synthesis; explicitly exclude only an unsupported rule, with a reason. Cite source rule IDs and shown case IDs. Source conditions, boundaries and actions stay traceable.""" + '\n' + PLAIN_LANGUAGE_INSTRUCTION


def _sample(ids, maximum=3):
    if len(ids) <= maximum:
        return list(ids)
    return [ids[round(i * (len(ids) - 1) / (maximum - 1))] for i in range(maximum)]


def packet(rules, conditions, observations, groups):
    used = set().union(*(condition_ids(effective_expression(r)) for r in rules)) if rules else set()
    views = [{k: r[k] for k in ('rule_id', 'action', 'conditions', 'boundaries', 'exceptions', 'status')} | {
        'statistics': {k: r['evidence_statistics'][k] for k in ('support_count', 'contradiction_count', 'coverage', 'conditional_action_frequency', 'eligible_cases')},
        'natural_rule_text': r.get('natural_rule_text', ''),
        'induction_rationale': r.get('induction_rationale', ''), 'boundary_rationale': r.get('boundary_rationale', ''),
        'supporting_case_ids': _sample(r['supporting_case_ids'] or r.get('induction_evidence_citations',{}).get('supporting_case_ids',[])),
        'contradicting_case_ids': _sample(r['contradicting_case_ids'])} for r in rules]
    sample_ids = {cid for row in views for key in ('supporting_case_ids', 'contradicting_case_ids') for cid in row[key]}
    examples = [o for o in observations if o.case_id in sample_ids]
    relevant_groups = {g.group_id for g in groups if any(o.case_id in g.member_case_ids for o in examples)}
    return {'rules': views,
            'conditions': [{'condition_id': c.condition_id, 'definition': c.definition, 'value_type': c.value_type} for c in conditions if c.condition_id in used],
            'open_codes': [{'case_id': o.case_id,
                       'actions': [{k:getattr(a,k) for k in ('family','method','target_feature','transformation','scope','target_expression')} for a in o.actions],
                       'condition_values': {cid: o.condition_values[cid] for cid in sorted(used) if cid in o.condition_values},
                       'candidate_condition_ids': [cid for cid in o.candidate_conditions if cid in used],
                       'observed_group_ids': [g.group_id for g in groups if o.case_id in g.member_case_ids]} for o in examples],
            'groups': [{'group_id': g.group_id, 'action': g.normalized_action} for g in groups if g.group_id in relevant_groups],
            'evidence_scope': 'Only actual local comparisons are included. Selected supporting and contradicting case-linked open codes are shown; statistics describe only the source group and the selected similar outside comparisons. Unselected cases are excluded. Original sentence pairs remain in the ledger and are not supplied here. Held-out outcomes are never used.'}


def branch_expression(refs, index):
    expressions = [effective_expression(index[rid]) for rid in refs]
    return expressions[0] if len(expressions) == 1 else {'op': 'or', 'args': expressions}


def decode_theory(raw, rules, payload, observations, groups, config):
    require(isinstance(raw, dict) and {'branches', 'excluded_rules', 'open_questions'} <= set(raw), 'Invalid predictive theory response')
    require(isinstance(raw['branches'], list) and isinstance(raw['excluded_rules'], list), 'Invalid theory branches')
    require(isinstance(raw['open_questions'], list) and all(isinstance(x, str) for x in raw['open_questions']), 'Invalid open questions')
    index = {r['rule_id']: r for r in rules};supplied_cases = {c['case_id'] for c in payload['open_codes']}
    covered, branches = set(), []
    for row in raw['branches']:
        require(set(row)=={'situation','source_rule_ids','interpretation','contrast_rule_ids','supporting_case_ids','contradicting_case_ids','uncertainty'}, 'Theory must not supply new executable conditions or actions')
        refs = row['source_rule_ids']
        require(isinstance(refs, list) and bool(refs) and len(refs) == len(set(refs)) and set(refs) <= set(index), 'Unknown/duplicate source rule IDs')
        group_ids = {index[rid]['action']['group_id'] for rid in refs}
        require(len(group_ids) == 1, 'A theory branch cannot invent or mix different actions')
        for key in ('situation', 'interpretation'):
            require(isinstance(row[key], str) and bool(row[key].strip()), 'Theory branch needs a situation and evidence interpretation')
        contrasts = row['contrast_rule_ids']
        require(isinstance(contrasts, list) and set(contrasts) <= set(index), 'Unknown theory contrast rules')
        require(isinstance(row['uncertainty'], list) and all(isinstance(x, str) for x in row['uncertainty']), 'Invalid branch uncertainty')
        expression = branch_expression(refs, index);gid = next(iter(group_ids))
        stats = evidence_statistics(expression, gid, observations, groups)
        for key in ('supporting_case_ids', 'contradicting_case_ids'):
            ids = row[key]
            require(isinstance(ids, list) and all(isinstance(cid, str) for cid in ids)
                    and set(ids) <= supplied_cases, 'Theory cites unknown evidence IDs')
        require(bool(row['supporting_case_ids']), 'Theory branch needs actual supporting edit evidence')
        # The model does not supply executable conditions/actions: they are inherited verbatim.
        branch = {k: row[k] for k in ('situation', 'source_rule_ids', 'interpretation', 'contrast_rule_ids', 'uncertainty')}
        branch.update(branch_id=stable_id('branch', {'rules': sorted(refs), 'version': VERSION}),
            when=expression, action=index[refs[0]]['action'], evidence_statistics=stats,
            evidence_citations={k: row[k] for k in ('supporting_case_ids', 'contradicting_case_ids')},
            status='candidate',
            statistical_diagnostic_status=('sufficient_construction_counts' if stats['support_count'] >= config.min_support and
                    (stats['conditional_action_frequency'] or 0) >= config.min_action_frequency else 'limited_construction_counts'),
            citation_accounting_disagreements={k:sorted(set(row[k])-set(stats[k])) for k in ('supporting_case_ids','contradicting_case_ids')})
        branches.append(branch);covered.update(refs)
    excluded = set()
    for row in raw['excluded_rules']:
        require(row['rule_id'] in index and row['rule_id'] not in excluded and isinstance(row['reason'], str) and bool(row['reason'].strip()), 'Invalid excluded theory rule')
        excluded.add(row['rule_id'])
    require(not covered & excluded and covered | excluded == set(index), 'Every source rule needs a standalone/synthesized statement or an explicit exclusion')
    return {'branches': branches, 'excluded_rules': raw['excluded_rules'], 'open_questions': raw['open_questions']}


def build_predictive_theory(pool, groups, conditions, observations, config, backend, cache=None):
    analysis_context = stable_id('cohort', {'observations':to_jsonable(observations),'groups':to_jsonable(groups)})
    queues = defaultdict(deque)
    for rule in pool['rules']:
        if rule['status'] != 'insufficient_evidence':
            queues[rule['action']['group_id']].append(rule)
    ordered = []
    while any(queues.values()):
        for gid in sorted(queues):
            if queues[gid]:ordered.append(queues[gid].popleft())
    branches, exclusions, questions = [], [], []
    offset = 0
    while offset < len(ordered):
        size = len(ordered) - offset
        while True:
            chunk = ordered[offset:offset + size];payload = packet(chunk, conditions, observations, groups)
            fits = getattr(backend, 'prompt_fits', None)
            if fits is None or fits(THEORY_TASK, payload, repair_reserve=True):break
            require(size > 1, 'One theory evidence packet exceeds 32K; no evidence was truncated')
            size = max(1, size // 2)
        logging.getLogger(__name__).info('Predictive theory: interpreting %d rules with %d case-linked open codes',len(chunk),len(payload['open_codes']))
        def compute():
            if hasattr(backend, 'reserve_work_calls'):
                backend.reserve_work_calls((backend.config.retries + 1) ** 2, 'simple_rule_theory_synthesis')
            return checked_infer(backend, THEORY_TASK, payload,
                lambda raw: decode_theory(raw, chunk, payload, observations, groups, config))
        result = (compute() if cache is None else cache.get_or_compute('predictive_theory_packets',
            {'version':VERSION,'payload':payload,'backend':backend.fingerprint,'analysis_context':analysis_context},compute))
        for branch in result['branches']:
            refs=branch['source_rule_ids'];source_index={r['rule_id']:r for r in chunk}
            relevant=set().union(*(set(source_index[r]['evidence_statistics'].get('compared_case_ids',source_index[r]['construction_case_ids'])) for r in refs))
            cases=[o for o in observations if o.case_id in relevant]
            assess_conditions(cases,conditions,backend,cache,condition_ids(branch['when']),config.workers)
            branch['evidence_statistics']=evidence_statistics(branch['when'],branch['action']['group_id'],observations,groups,relevant)
        branches.extend(result['branches']);exclusions.extend(result['excluded_rules']);questions.extend(result['open_questions'])
        offset += size
    unique = {}
    for branch in branches:
        unique.setdefault(branch['branch_id'],branch)
    theory = {**pool,'generation_mode':'rule_grounded_predictive_theory','predictive_theory_version':VERSION,
              'branches':list(unique.values()),'excluded_theory_rules':exclusions,'open_questions':questions,'evidence_reviews':[],
              'construction_method':'direct_rule_synthesis_with_selected_open_codes',
              'separate_model_audit':False,
              'prediction_basis':'Qwen interprets the frozen situation branches for a new Before; Python records condition diagnostics, statistics and provenance; Qwen decides semantic applicability.',
              'theory_model':getattr(getattr(backend,'config',None),'model','synthetic_mock'),
              'evidence_scope':'Built from source rules AND selected case-linked open codes, with counts from actual group and similar-context comparisons; unselected cases are not counted as unknown or counterexamples. Original Before/After pairs are not supplied to theory construction; source pairs remain traceable in the ledger.',
              'unsupported_new_conditions_or_actions_allowed':False}
    validate_predictive_theory(theory)
    return theory


def validate_predictive_theory(theory):
    require(theory.get('generation_mode')=='rule_grounded_predictive_theory','Predictive theory construction required')
    index = {r['rule_id']:r for r in theory['rules']};known = {c['condition_id'] for c in theory['conditions']}
    ids = [b['branch_id'] for b in theory['branches']]
    require(len(ids)==len(set(ids)),'Duplicate theory branches')
    for branch in theory['branches']:
        refs = branch['source_rule_ids']
        require(bool(refs) and set(refs)<=set(index),'Unknown frozen theory rule')
        require(all(index[r]['status']!='insufficient_evidence' for r in refs),'Unsupported source rule in predictive theory')
        require(branch['when']==branch_expression(refs,index),'Theory changed source rule conditions or boundaries')
        require(all(branch['action']==index[r]['action'] for r in refs),'Theory invented an action')
        validate_expr(branch['when'],known)
        require(set(branch['evidence_statistics']['supporting_case_ids']) <= set(theory['construction_case_ids']),'Unknown theory evidence')
    return True
