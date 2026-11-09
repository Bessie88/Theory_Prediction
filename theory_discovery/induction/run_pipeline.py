"""Five-stage orchestration. Default CLI is configuration/code validation only."""
import argparse
import json
import logging
from pathlib import Path
import time
from .action_grouping import group_actions
from .boundary_discovery import SemanticBoundaryProposer, discover_boundaries, relevant_observations
from .condition_comparison import compare_conditions
from theory_discovery.core.config import load_config
from .contracts import validate_artifacts
from theory_discovery.core.data_loader import load_pairs
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION, QwenJSONClient
from .rule_executor import effective_expression
from theory_discovery.core.schemas import ActionGroup, observation_from_dict, require, to_jsonable
from .semantic_alignment import SemanticActionResolver
from theory_discovery.core.storage import Cache, EvidenceLedger, write_json
from .structured_observation import extract_observations
from .theory_construction import construct_theory
from .scoped_conditions import assess_conditions, refresh_statistics

LOGGER = logging.getLogger(__name__)


def run_framework(pairs, config, backend=None, *, synthetic=False):
    config.validate()
    if not config.allow_real_data:
        require(synthetic and all(p.metadata.get('synthetic') is True for p in pairs), 'Construction mode only accepts explicitly synthetic inputs')
    require(len(pairs) <= config.max_observations, 'Observation processing budget exceeded')
    require(len({p.case_id for p in pairs}) == len(pairs), 'Duplicate observation IDs')
    require(all(p.split == 'discovery' for p in pairs), 'Discovery split required')
    cache = Cache(config.cache_dir)
    output = Path(config.output_dir)
    started = time.monotonic()
    ledger = EvidenceLedger(p.case_id for p in pairs)
    cfg = config.to_dict()
    # Local directory locations do not affect analytical cache identity.
    cfg.pop('output_dir')
    cfg.pop('cache_dir')
    fingerprint = backend.fingerprint if backend else 'mechanical-only-v1'
    LOGGER.info('Stage 1: extract actions and original per-case condition candidates; no global condition preprocessing')
    observations, catalog = extract_observations(pairs, config, cache, backend)
    conditions = list(catalog.conditions.values())
    alignment = {'original_conditions': getattr(catalog, 'original_conditions', to_jsonable(conditions)),
                 'condition_alignment': [], 'global_condition_alignment_enabled':False,
                 'condition_assessment_policy':'group_first_local_and_selected_outside_cases'}
    resolver = SemanticActionResolver(backend, config.llm.batch_size) if backend else None

    def grouping():
        groups = group_actions(observations, resolver)
        return {'groups': to_jsonable(groups), 'semantic_alignment': resolver.history if resolver else []}

    LOGGER.info('Stage 2: action methods aligned by meaning')
    grouping_result = cache.get_or_compute('action_groups', {'observations': to_jsonable(observations), 'backend': fingerprint,
        'grouping_policy':'incremental_approximate_method_type_v3_inherited_provenance'}, grouping)
    groups = [ActionGroup(**g) for g in grouping_result['groups']]
    alignment['action_alignment'] = grouping_result['semantic_alignment']
    for group in groups:
        ledger.record('action_grouping', 'group', group.member_case_ids, to_jsonable(group))
    write_json(output / 'action_groups.json',to_jsonable(groups))
    write_json(output / 'stage_progress.json',{'stage':'within_group_conditions','source_cases':len(observations),
        'action_groups':len(groups),'global_condition_alignment':False,'global_condition_matrix':False})
    LOGGER.info('Action grouping complete: %d cases, %d groups; assessing only original per-case conditions',len(observations),len(groups))
    assess_conditions(observations,conditions,backend,cache,workers=config.workers)
    inputs = {'observations': to_jsonable(observations), 'groups': to_jsonable(groups), 'conditions': to_jsonable(conditions), 'config': cfg}
    LOGGER.info('Stage 3: Qwen compares within-group open codes and proposes condition-action rules')
    def comparison_step():
        rows=compare_conditions(observations, groups, conditions, config, backend, cache)
        return {'comparisons':rows,'observations':to_jsonable(observations)}
    comparison_artifact = cache.get_or_compute('condition_comparisons',
        dict(inputs,backend=fingerprint,induction_version='qwen-group-first-local-conditions-v6'),comparison_step)
    comparisons=comparison_artifact['comparisons']
    observations=[observation_from_dict(o) for o in comparison_artifact['observations']]
    write_json(output / 'stage_progress.json',{'stage':'cross_group_boundaries','source_cases':len(observations),
        'action_groups':len(groups),'preliminary_rules':sum(len(c['preliminary_hypotheses']) for c in comparisons)})
    LOGGER.info('Stage 4: Qwen compares selected approximately similar outside contexts and adds boundaries')
    proposer = SemanticBoundaryProposer(backend, cache, config.workers) if backend and config.boundary_proposal_mode == 'qwen_open_codes' else None
    def boundary_step():
        results = discover_boundaries(comparisons, observations, groups, conditions, config, proposer)
        return {'results': results, 'semantic_proposals': proposer.history if proposer else [],
                'observations':to_jsonable(observations)}
    boundary_artifact = cache.get_or_compute('boundary_revisions', dict(inputs, comparisons=comparisons, backend=fingerprint, boundary_version='all-own-supports-all-matched-outside-v1'), boundary_step)
    boundaries = boundary_artifact['results']
    observations=[observation_from_dict(o) for o in boundary_artifact['observations']]
    refresh_statistics(comparisons,boundaries,observations,groups)
    inputs['observations']=to_jsonable(observations)
    alignment['boundary_proposals'] = boundary_artifact['semantic_proposals']
    for comparison in comparisons:
        for hypothesis in comparison['preliminary_hypotheses']:
            stats = hypothesis['statistics']
            ledger.record('condition_comparison', 'hypothesis', stats['supporting_case_ids'] + stats['contradicting_case_ids'], hypothesis)
    for boundary in boundaries:
        stats = boundary['original_statistics']
        ledger.record('boundary_discovery', 'boundary_decisions', stats['supporting_case_ids'] + stats['contradicting_case_ids'] + stats['unknown_case_ids'], boundary)
    LOGGER.info('Stage 5: direct synthesis of rules with selected open-code evidence')
    require(synthetic or backend is not None, 'Real predictive theory construction requires model evidence interpretation')
    def theory_step():
        result=construct_theory(boundaries, groups, conditions, observations, config, backend, cache)
        return {'theory':result,'observations':to_jsonable(observations)}
    theory_artifact = cache.get_or_compute('candidate_theories', dict(inputs, boundaries=boundaries,
        backend=fingerprint, theory_construction_version='rule-open-code-predictive-theory-v5-local-condition-evidence'),theory_step)
    theory=theory_artifact['theory']
    observations=[observation_from_dict(o) for o in theory_artifact['observations']]
    for obs in observations:
        ledger.record('structured_observation','source_observation',[obs.case_id],to_jsonable(obs))
    for condition in conditions:
        ledger.record('structured_observation','condition_definition',condition.evidence_by_case,to_jsonable(condition))
    validate_artifacts(observations, groups, conditions, comparisons, boundaries, theory)
    for rule in theory['rules']:
        ledger.record('theory_construction', 'rule', rule['construction_case_ids'], rule)
    for branch in theory.get('branches', []):
        citations = branch['evidence_citations']
        ledger.record('predictive_theory', 'evidence_grounded_branch',
            citations['supporting_case_ids'] + citations['contradicting_case_ids'], branch)
    artifacts = {'observations': to_jsonable(observations), 'conditions': to_jsonable(conditions),
                 'semantic_alignment': alignment, 'action_groups': to_jsonable(groups),
                 'condition_comparisons': comparisons, 'boundary_discovery': boundaries,
                 'theory': theory, 'evidence_ledger': ledger.to_dict()}
    for name, value in artifacts.items():
        write_json(output / (name + '.json'), value)
    stats = {'mode': 'synthetic_validation' if synthetic else 'discovery', 'processed_observations': len(observations),
             'observations_with_interpreted_actions': sum(any(a.semantic for a in o.actions) for o in observations),
             'discovered_methods': len(groups), 'candidate_rules': len(theory['rules']),
             'rules_with_sufficient_construction_evidence': sum(r['status'] != 'insufficient_evidence' for r in theory['rules']),
             'runtime_seconds': time.monotonic() - started, 'cache_hits': cache.hits, 'cache_misses': cache.misses,
             'llm_usage': dict(getattr(backend, 'stats', {})),
             'search_budgets_reached': sum(c['search_budget_reached'] for c in comparisons),
             'unresolved_case_ids': sorted({cid for b in boundaries for cid in b['unresolved_case_ids']}),
             'termination_means_saturation': False, 'independent_validation_performed': False,
             'condition_assessment_policy':'group_first_local_and_selected_outside_cases',
             'assessed_case_condition_pairs':sum(len(o.condition_values) for o in observations),
             'global_condition_alignment_performed':False,'global_condition_matrix_performed':False}
    write_json(output / 'run_statistics.json', stats)
    write_json(output / 'run_manifest.json', {'schema_version': '1', 'status': 'complete', 'config': config.to_dict(),
               'case_ids': [p.case_id for p in pairs], 'synthetic': synthetic, 'stage_instructions': {str(i): PLAIN_LANGUAGE_INSTRUCTION for i in range(1, 6)}, 'artifacts': list(artifacts), 'statistics': stats})
    write_json(output / 'stage_progress.json',{'stage':'complete','source_cases':len(observations),
        'action_groups':len(groups),'source_rules':len(theory['rules']),'theory_statements':len(theory.get('branches',[]))})
    return artifacts


def refine_rule(rule, observations, groups, conditions, config, proposer=None):
    """Targeted refinement of cached observations; no data loading or extraction."""
    group = next(g for g in groups if g.group_id == rule['action']['group_id'])
    subset = relevant_observations(effective_expression(rule), observations, group)
    # Only reconsider this rule's condition path. Other hypotheses/rules remain untouched.
    from .condition_comparison import evidence_statistics
    hypothesis = {'hypothesis_id': rule['refinement_history']['hypothesis_id'], 'group_id': group.group_id,
                  'conditions': effective_expression(rule),
                  'statistics': evidence_statistics(effective_expression(rule), group.group_id, subset, groups)}
    comparisons = [{'group_id': group.group_id, 'preliminary_hypotheses': [hypothesis]}]
    return {'retrieved_case_ids': [o.case_id for o in subset], 'cohort': 'relevant_cached_evidence_only',
            'boundaries': discover_boundaries(comparisons, subset, groups, conditions, config, proposer)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--execute', action='store_true', help='Future phase only; also requires allow_real_data=true in config')
    parser.add_argument('--verify-model', action='store_true', help='Explicit future endpoint check; no inference')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    cfg = load_config(args.config)
    if not args.execute and not args.verify_model:
        print(json.dumps({'status': 'configuration_valid', 'real_data_loaded': False, 'model_called': False}))
        return 0
    backend = QwenJSONClient(cfg.llm, Cache(cfg.cache_dir))
    if args.verify_model:
        backend.verify_model()
        print(json.dumps({'status': 'model_identifier_verified', 'model': cfg.llm.model}))
        if not args.execute:
            return 0
    require(cfg.allow_real_data and cfg.llm.enabled, 'Execution requires explicit real-data AND inference config gates')
    backend.verify_model()
    try:
        run_framework(list(load_pairs(cfg)), cfg, backend)
    except Exception as exc:
        LOGGER.error('Pipeline stopped: %s', type(exc).__name__)
        write_json(Path(cfg.output_dir) / 'run_manifest.json', {'status': 'failed', 'error_type': type(exc).__name__})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
