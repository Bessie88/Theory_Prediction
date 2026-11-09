"""Before-only prediction from integrated propositions, without expanding source rules."""
import hashlib
import json
import logging
from pathlib import Path

from .hierarchical_theory import VERSION
from .heldout_prediction import (load_before_only, text_hash, BASELINE_TASK, NPOV_POLICY,
                                  NPOV_SOURCE, baseline_decode)
from .llm import checked_infer, PLAIN_LANGUAGE_INSTRUCTION
from .natural_pipeline import validate_theory, fits
from .natural_theory_validation import selected_frozen_actions
from .response_recovery import (RecoverableQwenJSONClient, recoverable_cached_call,
                                 ModelOutputError, output_size_hint, remember_smaller_output)
from .schemas import require
from .storage import Cache, read_json, write_json
from .theory_prediction import decode_choice

PREDICTION_TASK = (
    'Using ONLY Before and the supplied frozen integrated theory propositions, predict '
    'a brief concrete editing strategy or source-backed combination of strategies. '
    'Read the situation, explanation, boundaries and preserved action alternatives. '
    'Use the source rule IDs to identify the supported directions; the original source '
    'rules remain frozen but are not expanded here. Do not invent facts or sources, '
    'claim knowledge of the actual edit, or force one action when a supported combination '
    'fits. Abstain when no concrete strategy is supported. Return '
    '{"abstain":bool,"branch_id":str|null,"source_rule_ids":[str],'
    '"strategy":str,"reason":str}; at most 100 words for strategy. '
    + PLAIN_LANGUAGE_INSTRUCTION
)


def compact_choices(theory):
    """Use integrated prose/boundaries and action metadata, never full source_rules."""
    return [{key: branch.get(key, '') for key in
             ('branch_id', 'situation', 'interpretation', 'boundary_text', 'uncertainty',
              'source_rule_ids', 'action_options')}
            for branch in theory['branches']]


def predict_with_hierarchical(before, theory, backend, cache, theory_hash):
    require(isinstance(before,str) and bool(before.strip()), 'Before-only text required')
    require(theory.get('integration_method')=='within_method_group_then_all_groups',
            'Hierarchical theory required')
    validate_theory(theory)
    original={b['branch_id']:b for b in theory['branches']}
    choices=compact_choices(theory)
    def decode(raw, chunk):
        require(isinstance(raw,dict), 'Missing integrated theory prediction')
        clean={key:raw.get(key) for key in ('abstain','branch_id','source_rule_ids','reason')}
        if not isinstance(clean['reason'],str) or not clean['reason'].strip():
            clean['reason']='The frozen integrated account supports this decision.'
        decision=decode_choice(clean,chunk)
        if not decision['abstain']:
            selected_frozen_actions(original[decision['branch_id']],decision['source_rule_ids'])
        strategy=raw.get('strategy')
        decision['strategy']=strategy if isinstance(strategy,str) and strategy.strip() else None
        return decision
    finalists=choices
    while finalists:
        winners=[];offset=0;last=None
        while offset<len(finalists):
            size=min(4,len(finalists)-offset)
            while True:
                chunk=finalists[offset:offset+size]
                payload={'before':before,'theory_situations':chunk,'frozen_theory_sha256':theory_hash}
                hint=output_size_hint(backend,'hierarchical_heldout_choices',PREDICTION_TASK,payload)
                if hint is not None and hint<size:
                    size=hint;continue
                if fits(backend,PREDICTION_TASK,payload):break
                require(size>1,'One integrated proposition exceeds context; Before was not truncated')
                size=max(1,size//2)
            try:
                decision=recoverable_cached_call(backend,cache,'hierarchical_heldout_choices',
                    VERSION,PREDICTION_TASK,payload,lambda raw:decode(raw,chunk))
            except ModelOutputError as exc:
                if size<=1:raise
                remember_smaller_output(backend,'hierarchical_heldout_choices',PREDICTION_TASK,
                                       payload,size,exc)
                continue
            if not decision['abstain']:
                winners.append(next(b for b in chunk if b['branch_id']==decision['branch_id']))
                last=decision
            offset+=size
        if not winners:break
        if len(winners)==1:
            branch=original[last['branch_id']]
            return {'strategy':last.get('strategy') or branch['interpretation'],
                'source_action_options':selected_frozen_actions(branch,last['source_rule_ids']),
                'abstained':False,'branch_id':last['branch_id'],'source_rule_ids':last['source_rule_ids'],
                'reason':last['reason'],'applicable_branch_count':len(choices),
                'unknown_branch_ids':[],'rule_representation':'natural_language'}
        require(len(winners)<len(finalists),'Integrated theory comparisons cannot reduce choices')
        finalists=winners
    return {'strategy':'','abstained':True,'branch_id':None,'source_rule_ids':[],
        'reason':'The frozen integrated theory did not justify a concrete choice.',
        'applicable_branch_count':len(choices),'unknown_branch_ids':[],
        'rule_representation':'natural_language'}


def predict(before_path, frozen_directory, config, output, expected_count=200, backend=None):
    """Same frozen-theory/NPOV protocol; only the integrated predictor differs."""
    output=Path(output);frozen_directory=Path(frozen_directory)
    frozen_bytes=(frozen_directory/'theory.json').read_bytes()
    freeze=read_json(frozen_directory/'freeze_manifest.json')
    require(hashlib.sha256(frozen_bytes).hexdigest()==freeze['source_theory_sha256'],
            'Frozen theory integrity failure')
    theory=json.loads(frozen_bytes);validate_theory(theory)
    require(theory.get('integration_method')=='within_method_group_then_all_groups',
            'Hierarchical theory required')
    rows=load_before_only(before_path,expected_count)
    require(not {text_hash(row['before']) for row in rows} &
            set(freeze['construction_before_sha256']), 'Held-out Before overlaps discovery')
    cache=Cache(config.cache_dir)
    backend=backend or RecoverableQwenJSONClient(config.llm,cache)
    results={'theory':[],'npov':[]}
    for index,row in enumerate(rows):
        logging.getLogger(__name__).info('Integrated Before-only predictions %d/%d: %s',
                                        index+1,len(rows),row['case_id'])
        prediction=predict_with_hierarchical(row['before'],theory,backend,cache,
                                            freeze['source_theory_sha256'])
        results['theory'].append({'case_id':row['case_id'],**prediction})
        baseline=checked_infer(backend,BASELINE_TASK,
            {'before':row['before'],'npov_guidance':NPOV_POLICY},baseline_decode)
        results['npov'].append({'case_id':row['case_id'],'strategy':baseline['strategy'],
                                'abstained':baseline['abstain']})
        for method,predictions in results.items():
            write_json(output/(method+'_predictions.json'),predictions)
    require(all(len(v)==expected_count for v in results.values()),'Incomplete predictions')
    final_hash=hashlib.sha256((frozen_directory/'theory.json').read_bytes()).hexdigest()
    require(final_hash==freeze['source_theory_sha256'],'Theory changed during prediction')
    manifest={'status':'complete','count_per_method':expected_count,'methods':['theory','npov'],
        'model':config.llm.model,'context_length':config.llm.context_length,
        'thinking':config.llm.enable_thinking,'temperature':config.llm.temperature,
        'input_allowlist':['before'],'heldout_after_loaded':False,
        'prediction_basis':'qwen_frozen_group_integrated_natural_language_theory',
        'theory_used':True,'rule_representation':'natural_language',
        'source_rules_expanded_in_prediction_prompts':False,
        'method_display_names':{'theory':'Theory-guided','npov':'NPOV baseline'},
        'frozen_theory_sha256':final_hash,'npov_source':NPOV_SOURCE,
        'prediction_files_sha256':{m:hashlib.sha256((output/(m+'_predictions.json')).read_bytes()).hexdigest()
                                   for m in results},
        'input_sha256':hashlib.sha256(Path(before_path).read_bytes()).hexdigest()}
    write_json(output/'prediction_complete.json',manifest)
    return results
