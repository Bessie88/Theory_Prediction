"""Qwen situation-based prediction with frozen source-rule provenance."""
from theory_discovery.induction.conditions import ConditionDetector
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION, checked_infer
from theory_discovery.induction.predictive_theory import validate_predictive_theory, VERSION
from theory_discovery.induction.rule_executor import evaluate, effective_expression, condition_ids
from theory_discovery.core.schemas import BeforeContext, require

THEORY_PREDICTION_TASK = '''Using only Before and the frozen evidence-grounded predictive theory, choose the editing behavior that best fits this situation. Read the situation descriptions, interpretations, boundaries, alternatives and uncertainty. Do not select mechanically by frequency alone, invent a new branch/action/condition, infer unseen sources, or claim knowledge of the actual edit. Several actions may remain plausible; abstain when the theory does not justify a choice.
Return {"abstain":bool,"branch_id":str|null,"source_rule_ids":[str],"reason":str}. Select a supplied branch and at least one of its source rules, or abstain with null and an empty list. Judge semantic applicability from Before, allowing approximate meanings; condition diagnostics are advisory, not eligibility gates. Respect the theory boundaries and abstain if the evidence does not support transfer. Python supplies the action from the chosen frozen branch. Explain briefly using Before and the theory.''' + '\n' + PLAIN_LANGUAGE_INSTRUCTION


def decode_choice(raw, choices):
    require(isinstance(raw,dict) and set(raw)=={'abstain','branch_id','source_rule_ids','reason'},'Invalid theory prediction response')
    require(type(raw['abstain']) is bool and isinstance(raw['reason'],str) and bool(raw['reason'].strip()),'Prediction needs abstention flag and reason')
    refs = raw['source_rule_ids']
    require(isinstance(refs,list) and all(isinstance(r,str) for r in refs) and len(refs)==len(set(refs)),'Invalid prediction source rules')
    index = {c['branch_id']:c for c in choices}
    if raw['abstain']:
        require(raw['branch_id'] is None and not refs,'Abstention cannot claim a theory branch')
    else:
        require(raw['branch_id'] in index,'Prediction invented a theory branch')
        require(bool(refs) and set(refs)<=set(index[raw['branch_id']]['source_rule_ids']), 'Prediction cites an unknown source rule')
    return raw


def predict_with_theory(before, theory, backend, cache, theory_hash):
    require(isinstance(before,str),'Theory prediction accepts Before text only')
    if theory.get('generation_mode') == 'natural_language_predictive_theory':
        from theory_discovery.induction.natural_pipeline import predict_natural
        return predict_natural(before, theory, backend, cache, theory_hash)
    if theory.get('generation_mode')!='rule_grounded_predictive_theory':
        require(not theory.get('rules'),'Legacy rule matching cannot replace predictive theory')
        return {'strategy':'','abstained':True,'branch_id':None,'source_rule_ids':[], 'reason':'No supported predictive theory branches.', 'applicable_branch_count':0,'unknown_branch_ids':[]}
    validate_predictive_theory(theory)
    if not theory['branches']:
        return {'strategy':'','abstained':True,'branch_id':None,'source_rule_ids':[], 'reason':'No supported predictive theory branches.', 'applicable_branch_count':0,'unknown_branch_ids':[]}
    from theory_discovery.core.schemas import condition_from_dict
    used = set().union(*(condition_ids(b['when']) for b in theory['branches']))
    definitions = [c for c in theory['conditions'] if c['condition_id'] in used]
    values = cache.get_or_compute('heldout_theory_condition_values',
        {'before':before,'theory_hash':theory_hash,'version':VERSION,'backend':backend.fingerprint},
        lambda:ConditionDetector(backend).detect(BeforeContext(before),[condition_from_dict(c) for c in definitions]))
    rules = {r['rule_id']:r for r in theory['rules']}
    choices, unknown = [], []
    for branch in theory['branches']:
        result = evaluate(branch['when'],values)
        if result is None:unknown.append(branch['branch_id'])
        refs = list(branch['source_rule_ids'])
        stats = branch['evidence_statistics']
        choices.append({k:branch[k] for k in ('branch_id','situation','interpretation','when','action','contrast_rule_ids','uncertainty')} | {
            'source_rule_ids':refs,
            'condition_diagnostics':{'branch_expression_result':result, 'advisory_only':True},
            'discovery_statistics':{k:stats[k] for k in ('support_count','contradiction_count','conditional_action_frequency','coverage')}})
    if not choices:
        return {'strategy':'','abstained':True,'branch_id':None,'source_rule_ids':[],
                'reason':'Unknown conditions or no supported applicable theory situation.',
                'applicable_branch_count':0,'unknown_branch_ids':unknown}
    finalists = choices
    selected = None
    # Bounded contextual comparisons, then compare the winners; every frozen branch enters; Qwen decides applicability.
    while True:
        winners = []
        offset = 0
        while offset < len(finalists):
            size = min(4,len(finalists)-offset)
            while True:
                chunk = finalists[offset:offset+size]
                ids = set().union(*(condition_ids(c['when']) for c in chunk))
                payload = {'before':before,'theory_situations':chunk,
                    'condition_definitions':[{k:c[k] for k in ('condition_id','definition','value_type')} for c in definitions if c['condition_id'] in ids],
                    'before_condition_values':{cid:values[cid] for cid in sorted(ids)},
                    'open_questions':[]}
                fits = getattr(backend,'prompt_fits',None)
                if fits is None or fits(THEORY_PREDICTION_TASK,payload,repair_reserve=True):break
                require(size>1,'One predictive theory situation exceeds context; no Before text was truncated')
                size=max(1,size//2)
            def compute():
                if hasattr(backend,'reserve_work_calls'):
                    backend.reserve_work_calls((backend.config.retries+1)**2,'theory_situation_prediction')
                return checked_infer(backend,THEORY_PREDICTION_TASK,payload,lambda raw:decode_choice(raw,chunk))
            decision = cache.get_or_compute('heldout_theory_choices',
                {'version':VERSION,'payload':payload,'backend':backend.fingerprint},compute)
            if not decision['abstain']:
                winner = next(c for c in chunk if c['branch_id']==decision['branch_id'])
                winners.append(winner);selected=decision
            offset+=size
        if not winners:
            return {'strategy':'','abstained':True,'branch_id':None,'source_rule_ids':[],
                    'reason':'The frozen theory did not justify one concrete choice.',
                    'applicable_branch_count':len(choices),'unknown_branch_ids':unknown}
        if len(winners)==1:break
        require(len(winners)<len(finalists),'Theory comparison cannot reduce candidate choices within context')
        finalists=winners
    branch = next(c for c in choices if c['branch_id']==selected['branch_id'])
    action = branch['action']
    return {'strategy':action['method']+' (target: '+action['target_feature']+').',
            'abstained':False,'branch_id':selected['branch_id'],'source_rule_ids':selected['source_rule_ids'],
            'reason':selected['reason'],'applicable_branch_count':len(choices),'unknown_branch_ids':unknown}
