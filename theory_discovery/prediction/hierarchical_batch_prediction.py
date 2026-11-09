"""Batch the Before-only DATA; include the complete frozen theory in every request."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
from pathlib import Path

from .heldout_prediction import load_before_only,text_hash,NPOV_POLICY,NPOV_SOURCE
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION
from theory_discovery.induction.natural_pipeline import validate_theory,fits
from theory_discovery.induction.natural_theory_validation import action_options_for_refs
from theory_discovery.core.response_recovery import RecoverableQwenJSONClient,recoverable_cached_call,ModelOutputError
from theory_discovery.core.schemas import require,stable_id
from theory_discovery.core.storage import Cache,read_json,write_json

VERSION='whole-theory-data-batch-v1'
THEORY_BATCH_TASK=(
    'Predict a concrete neutralizing editing strategy for EACH supplied Before-only case. '
    'The COMPLETE frozen integrated theory is supplied once and shared by this DATA batch. '
    'Read all its propositions together. Do not partition or tournament-select theory branches. '
    'Treat each case independently; do not transfer facts or proposed edits between cases. '
    'Use supported source action directions or a coherent supported combination; do not force '
    'one source action or one theory branch. Do not invent facts/sources or infer the human After. '
    'Return {"predictions":[{"case_id":str,"strategy":str,"abstain":bool,'
    '"source_rule_ids":[str],"reason":str}]}. Provide exactly one prediction for every supplied '
    'case ID; no unrequested cases. Each strategy should be concise, at most 100 words. '
    'Abstain only when no concrete source-backed prediction is supported. Source rule IDs are '
    'sufficient evidence. '+PLAIN_LANGUAGE_INSTRUCTION)
NPOV_BATCH_TASK=(
    'Using ONLY each Before text and the fixed NPOV guidance, independently predict one '
    'concrete strategy for making EACH supplied sentence more neutral. Describe the operation '
    'and textual target, not a complete rewrite or an unrelated list of alternatives. '
    'Do not transfer facts between cases, invent sources/facts or infer the actual edit. '
    'Abstain if no concrete prediction is supported. Return {"predictions":['
    '{"case_id":str,"strategy":str,"abstain":bool}]}. Exactly one result per supplied case ID; '
    'at most 100 words per strategy. '+PLAIN_LANGUAGE_INSTRUCTION)
LOGGER=logging.getLogger(__name__)


def full_theory_payload(theory):
    """All propositions, relationships and source action directions, without raw case data."""
    validate_theory(theory)
    require(theory.get('integration_method')=='within_method_group_then_all_groups',
            'Integrated theory required before batch prediction')
    source={r['rule_id']:r for r in theory['rules']}
    covered=list(dict.fromkeys(rid for branch in theory['branches']
                               for rid in branch['source_rule_ids']))
    return {
        'global_explanation':theory['global_explanation'],
        'propositions':[{key:branch.get(key,'') for key in
            ('branch_id','situation','interpretation','boundary_text','uncertainty','source_rule_ids')}
            for branch in theory['branches']],
        'source_actions':action_options_for_refs(covered,source),
        'open_questions':theory.get('open_questions',[])}


def _refs(value):
    if value is None:return []
    if isinstance(value,str):return [value] if value.strip() else []
    require(isinstance(value,list) and all(isinstance(x,str) for x in value),'Invalid source IDs')
    return list(dict.fromkeys(value))


def _bool(value):
    if type(value) is bool:return value
    if isinstance(value,str) and value.strip().lower() in ('true','false'):
        return value.strip().lower()=='true'
    if type(value) is int and value in (0,1):return bool(value)
    raise ValueError('Missing interpretable abstention decision')


def _decode(raw,cases,allowed_sources=None):
    require(isinstance(raw,dict),'Missing batch response')
    rows=raw.get('predictions')
    if rows is None and len(cases)==1 and 'case_id' in raw:rows=[raw]
    require(isinstance(rows,list),'Missing complete batch predictions list')
    expected={r['case_id'] for r in cases};seen=set();result=[]
    for row in rows:
        require(isinstance(row,dict),'Invalid case prediction')
        cid=row.get('case_id')
        require(cid in expected and cid not in seen,'Unknown or duplicate predicted case')
        abstain=_bool(row.get('abstain',row.get('abstained')))
        strategy=row.get('strategy','' if abstain else None)
        require(isinstance(strategy,str) and (abstain or strategy.strip()),
                'Missing non-abstaining strategy')
        normalized={'case_id':cid,'abstain':abstain,'strategy':strategy}
        if allowed_sources is not None:
            refs=_refs(row.get('source_rule_ids'))
            require(set(refs)<=allowed_sources and (abstain or refs),
                    'Prediction needs valid frozen source rules')
            normalized['source_rule_ids']=refs
            reason=row.get('reason')
            normalized['reason']=reason if isinstance(reason,str) else 'Supported by the frozen theory.'
        result.append(normalized);seen.add(cid)
    # Missing rows are retried as DATA only; valid rows are retained, never counted as absent predictions.
    return {'predictions':result}


def _request(backend,cache,method,shared,cases,allowed_sources):
    task=THEORY_BATCH_TASK if method=='theory' else NPOV_BATCH_TASK
    payload={**shared,'cases':cases}
    require(fits(backend,task,payload),'Data batch exceeds context; complete theory was retained')
    try:
        result=recoverable_cached_call(backend,cache,'whole_theory_'+method+'_data_batches',
            VERSION,task,payload,lambda raw:_decode(raw,cases,allowed_sources))
    except (ModelOutputError,ValueError):
        if len(cases)==1:raise
        middle=len(cases)//2
        return (_request(backend,cache,method,shared,cases[:middle],allowed_sources)
                +_request(backend,cache,method,shared,cases[middle:],allowed_sources))
    rows=result['predictions']
    found={r['case_id'] for r in rows};missing=[r for r in cases if r['case_id'] not in found]
    if missing:
        if not found:
            require(len(cases)>1,'Singleton prediction remained missing after recovery')
            middle=len(cases)//2
            return (_request(backend,cache,method,shared,cases[:middle],allowed_sources)
                    +_request(backend,cache,method,shared,cases[middle:],allowed_sources))
        rows+=_request(backend,cache,method,shared,missing,allowed_sources)
    by_id={r['case_id']:r for r in rows}
    require(set(by_id)=={r['case_id'] for r in cases},'Batch recovery missed a data case')
    return [by_id[r['case_id']] for r in cases]


def plan_data_batches(rows,complete_theory,backend,maximum=12):
    """Only reduce case count for context; NEVER divide the theory."""
    require(type(maximum) is int and maximum>=1,'Positive data batch size required')
    batches=[];offset=0
    while offset<len(rows):
        size=min(maximum,len(rows)-offset)
        while True:
            cases=rows[offset:offset+size]
            if (fits(backend,THEORY_BATCH_TASK,{'frozen_theory':complete_theory,'cases':cases})
                and fits(backend,NPOV_BATCH_TASK,{'npov_guidance':NPOV_POLICY,'cases':cases})):break
            require(size>1,'Complete theory plus one Before exceeds context; no theory was dropped')
            size=max(1,size//2)
        batches.append(cases);offset+=size
    return batches


def predict(before_path,frozen_directory,config,output,expected_count=200,backend=None,
            *,batch_size=None,workers=4):
    require(type(workers) is int and 1<=workers<=4,'Use one to four batch workers')
    output=Path(output);frozen_directory=Path(frozen_directory)
    frozen_bytes=(frozen_directory/'theory.json').read_bytes()
    freeze=read_json(frozen_directory/'freeze_manifest.json')
    frozen_hash=hashlib.sha256(frozen_bytes).hexdigest()
    require(frozen_hash==freeze['source_theory_sha256'],'Frozen theory integrity failure')
    theory=json.loads(frozen_bytes);complete=full_theory_payload(theory)
    rows=load_before_only(before_path,expected_count)
    require(not {text_hash(r['before']) for r in rows}&set(freeze['construction_before_sha256']),
            'Held-out Before overlaps discovery')
    cache=Cache(config.cache_dir);backend=backend or RecoverableQwenJSONClient(config.llm,cache)
    maximum=batch_size if batch_size is not None else config.llm.batch_size
    batch_plan=plan_data_batches(rows,complete,backend,maximum)
    source={r['rule_id']:r for r in theory['rules']}
    allowed={rid for p in complete['propositions'] for rid in p['source_rule_ids']}
    input_hash=hashlib.sha256(Path(before_path).read_bytes()).hexdigest()
    contract={'version':VERSION,'frozen_theory_sha256':frozen_hash,'input_sha256':input_hash,
              'backend':backend.fingerprint,'maximum_data_batch_size':maximum}
    contract_path=output/'batch_run_contract.json'
    if contract_path.exists():require(read_json(contract_path)==contract,'Batch resume contract changed')
    else:write_json(contract_path,contract)
    expected_ids={r['case_id'] for r in rows}
    results={'theory':{},'npov':{}}
    checkpoint=output/'paired_predictions_checkpoint.json'
    if checkpoint.exists():
        saved=read_json(checkpoint)
        require(saved.get('contract_hash')==stable_id('batch_contract',contract),
                'Partial prediction checkpoint contract changed')
        for method in results:
            values=saved.get(method)
            require(isinstance(values,dict) and set(values)<=expected_ids,
                    'Invalid partial batch checkpoint')
            results[method]=values
        require(set(results['theory'])==set(results['npov']),'Paired checkpoint is incomplete')
    def persist():
        # Commit both methods together. Interrupted display-file writes can be regenerated.
        write_json(checkpoint,{'contract_hash':stable_id('batch_contract',contract),**results})
        for method,values in results.items():
            write_json(output/(method+'_predictions.json'),
                       [values[r['case_id']] for r in rows if r['case_id'] in values])
        write_json(output/'batch_progress.json',{'stage':'batch_prediction',
            'completed_cases':len(results['theory']),'total_cases':len(rows),
            'data_batch_maximum':maximum,'planned_data_batches':len(batch_plan),
            'workers':workers,'full_theory_in_every_theory_request':True,'theory_partitioned':False})
    persist()
    def work(cases):
        ids={r['case_id'] for r in cases}
        theory_rows=_request(backend,cache,'theory',{'frozen_theory':complete},cases,allowed)
        npov_rows=_request(backend,cache,'npov',{'npov_guidance':NPOV_POLICY},cases,None)
        t={}
        for row in theory_rows:
            refs=row['source_rule_ids'];branch_ids=[b['branch_id'] for b in theory['branches']
                if set(refs)&set(b['source_rule_ids'])]
            t[row['case_id']]={'case_id':row['case_id'],'strategy':row['strategy'],
                'abstained':row['abstain'],'source_rule_ids':refs,
                'source_action_options':action_options_for_refs(refs,source),
                'branch_ids':branch_ids,'branch_id':branch_ids[0] if len(branch_ids)==1 else None,
                'reason':row['reason'],'rule_representation':'natural_language'}
        n={row['case_id']:{'case_id':row['case_id'],'strategy':row['strategy'],
                          'abstained':row['abstain']} for row in npov_rows}
        require(set(t)==set(n)==ids,'Data batch case accounting mismatch')
        return t,n
    remaining=[[row for row in batch if row['case_id'] not in results['theory']]
               for batch in batch_plan]
    remaining=[batch for batch in remaining if batch]
    LOGGER.info('Data batches: %d batches, maximum %d Before cases, %d workers; entire theory in every request',
                len(remaining),maximum,workers)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='prediction-data-batch') as pool:
        pending={pool.submit(work,batch):batch for batch in remaining}
        for future in as_completed(pending):
            t,n=future.result();results['theory'].update(t);results['npov'].update(n);persist()
            LOGGER.info('Batch predictions complete: %d/%d Before cases, both methods',
                        len(results['theory']),len(rows))
    require(all(set(v)==expected_ids for v in results.values()),'Incomplete batched predictions')
    require(hashlib.sha256((frozen_directory/'theory.json').read_bytes()).hexdigest()==frozen_hash,
            'Theory changed during prediction')
    manifest={'status':'complete','count_per_method':expected_count,'methods':['theory','npov'],
        'model':config.llm.model,'context_length':config.llm.context_length,
        'thinking':config.llm.enable_thinking,'temperature':config.llm.temperature,
        'input_allowlist':['before'],'heldout_after_loaded':False,
        'prediction_basis':'whole_frozen_theory_shared_by_data_batch','theory_used':True,
        'rule_representation':'natural_language','frozen_theory_sha256':frozen_hash,
        'npov_source':NPOV_SOURCE,'input_sha256':input_hash,
        'method_display_names':{'theory':'Theory-guided','npov':'NPOV baseline'},
        'data_batch_maximum':maximum,'batch_workers':workers,'theory_partitioned':False,
        'prediction_files_sha256':{m:hashlib.sha256((output/(m+'_predictions.json')).read_bytes()).hexdigest()
                                   for m in results}}
    write_json(output/'prediction_complete.json',manifest)
    persist()
    return {method:[values[r['case_id']] for r in rows] for method,values in results.items()}
