"""Four-rule scheduling for the next resume; prompts and analytical cache identities stay unchanged."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from pathlib import Path
import os
import threading
import time
from ._support import natural_pipeline as serial
from ._support.natural_pipeline import (require, Cache, LOGGER, SemanticActionResolver, to_jsonable,
    group_actions, ActionGroup, GROUPING_POLICY, write_json, assess_conditions, induce,
    synthesize, EvidenceLedger, VERSION)
from theory_discovery.core.schemas import stable_id

SCHEDULING_VERSION = 'four-independent-rules-v1'


def validate_capacity(server_info, workers=4, context=32768):
    args = server_info.get('server_args', server_info)
    def value(name):
        return args.get(name, server_info.get(name))
    configured_context = value('context_length')
    running = value('max_running_requests')
    total = value('max_total_num_tokens')
    require(type(workers) is int and 1 <= workers <= 4, 'Use between one and four independent rule workers')
    require(configured_context == context, 'Unexpected per-request context')
    require(type(running) is int and workers <= running, 'Worker count exceeds server concurrency')
    require(type(total) is int and workers * context <= total, 'Insufficient aggregate KV token capacity')
    return {'status': 'passed', 'workers': workers, 'per_request_context': context,
        'conservative_aggregate_context_tokens': workers * context, 'server_KV_token_capacity': total,
        'server_max_running_requests': running, 'single_request_context_changed': False,
        'note': 'Capacity bounds, not a measured speedup or live parallel GPU result.'}


class KeyedCache:
    """Avoid duplicate stage computation when independent rules share the same request."""
    def __init__(self, base):
        self.base = base
        self.directory = base.directory
        self._guard = threading.Lock()
        self._locks = {}
    def get_or_compute(self, stage, inputs, compute):
        key = stable_id('parallel_lock', {'stage': stage, 'inputs': inputs})
        with self._guard:
            lock = self._locks.setdefault(key, threading.RLock())
        with lock:
            return self.base.get_or_compute(stage, inputs, compute)


def parallel_boundaries(rules, observations, groups, conditions, backend, cache,
                        workers=4, progress_path=None):
    require(type(workers) is int and 1 <= workers <= 4, 'Invalid independent-rule worker count')
    shared = KeyedCache(cache)
    cohort = stable_id('boundary_cohort', {'observations': to_jsonable(observations),
        'groups': to_jsonable(groups), 'conditions': to_jsonable(conditions),
        'backend': backend.fingerprint, 'method_version': VERSION, 'cross_group_policy': serial.CROSS_GROUP_VERSION})
    results = [None] * len(rules)
    completed = set()

    def work(index):
        # A worker exclusively owns its rule. Cohort observations/conditions are read-only.
        original = deepcopy(rules[index])
        def compute():
            local = deepcopy(original)
            result = serial.add_boundaries([local], observations, groups, conditions, backend, shared)
            require(len(result) == 1, 'A rule worker returned incomplete boundary output')
            return {'rule': local, 'boundary': result[0]}
        result = shared.get_or_compute('natural_boundary_rules', {'scheduling_version': SCHEDULING_VERSION,
            'method_version': VERSION, 'cross_group_policy': serial.CROSS_GROUP_VERSION, 'cohort': cohort, 'original_rule': original}, compute)
        return index, result

    def record():
        if progress_path is not None:
            write_json(progress_path, {'stage': 'natural_boundaries', 'execution': 'parallel_independent_rules',
                'workers': workers, 'source_rules': len(rules), 'completed_rules': len(completed),
                'percent': round(100 * len(completed) / len(rules), 1) if rules else 100.0,
                'completed_rule_ids': [r['rule_id'] for i, r in enumerate(rules) if i in completed]})

    record()
    if not rules:
        return []
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='boundary-rule')
    pending = {}
    remaining = iter(range(len(rules)))
    try:
        for _ in range(min(workers, len(rules))):
            i = next(remaining)
            pending[executor.submit(work, i)] = i
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            successful = []
            # Observe every finished future before submitting more work.
            for future in done:
                i = pending.pop(future)
                index, result = future.result()
                require(index == i, 'Rule worker index mismatch')
                rules[i].clear()
                rules[i].update(deepcopy(result['rule']))
                results[i] = result['boundary']
                completed.add(i)
                successful.append(i)
            record()
            LOGGER.info('Parallel boundaries: %d/%d rules complete; up to %d independent rules in flight',
                        len(completed), len(rules), workers)
            for _ in successful:
                try:
                    i = next(remaining)
                except StopIteration:
                    break
                pending[executor.submit(work, i)] = i
        require(all(r is not None for r in results), 'Parallel scheduling omitted a source rule')
        return results
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def add_boundaries(rules, observations, groups, conditions, backend, cache):
    workers = int(os.environ.get('SIMULATION_BOUNDARY_WORKERS', '4'))
    output = getattr(cache, 'parallel_progress_path', None)
    return parallel_boundaries(rules, observations, groups, conditions, backend, cache,
                               workers=workers, progress_path=output)



def run_framework(pairs, config, backend, *, synthetic=False, retain_unchanged=False):
    from ._support.structured_observation import extract_observations
    config.validate()
    require(backend is not None, 'Natural-language discovery requires a model backend')
    require(config.allow_real_data or synthetic and all(p.metadata.get('synthetic') for p in pairs), 'Unapproved real-data execution')
    require(len(pairs) <= config.max_observations and len({p.case_id for p in pairs}) == len(pairs)
            and all(p.split == 'discovery' for p in pairs), 'Invalid discovery cohort')
    started = time.monotonic();cache = Cache(config.cache_dir);output = __import__('pathlib').Path(config.output_dir)
    cache.parallel_progress_path = output / 'stage_progress.json'
    LOGGER.info('Stage 1: reuse concrete actions and original Before-only candidates')
    observations, catalog = extract_observations(pairs, config, cache, backend,
                                                  retain_unchanged=retain_unchanged)
    conditions = list(catalog.conditions.values());fingerprint = backend.fingerprint
    resolver = SemanticActionResolver(backend, config.llm.batch_size)
    def grouping():
        return {'groups': to_jsonable(group_actions(observations, resolver)), 'semantic_alignment': resolver.history}
    LOGGER.info('Stage 2: reuse editing-method groups before condition assessment')
    grouped = cache.get_or_compute('action_groups', {'observations': to_jsonable(observations),
        'backend': fingerprint, 'grouping_policy': GROUPING_POLICY}, grouping)
    groups = [ActionGroup(**g) for g in grouped['groups']]
    write_json(output / 'action_groups.json', to_jsonable(groups))
    LOGGER.info('Action grouping complete: %d cases, %d groups', len(observations), len(groups))
    assess_conditions(observations, conditions, backend, cache, workers=config.workers)
    write_json(output / 'stage_progress.json', {'stage': 'natural_rules', 'source_cases': len(observations), 'action_groups': len(groups)})
    LOGGER.info('Stage 3: natural-language rules; no executable condition conversion')
    comparisons, rules = induce(observations, groups, conditions, config, backend, cache)
    write_json(output / 'natural_rules_before_boundaries.json', rules)
    write_json(output / 'stage_progress.json', {'stage': 'natural_boundaries', 'source_rules': len(rules)})
    LOGGER.info('Stage 4: natural-language boundaries from all matched outside cases and all own supports')
    boundaries = add_boundaries(rules, observations, groups, conditions, backend, cache)
    write_json(output / 'natural_rules.json', rules);write_json(output / 'boundary_discovery.json', boundaries)
    write_json(output / 'stage_progress.json', {'stage': 'natural_theory',
        'completed_boundary_rules': len(boundaries), 'source_rules': len(rules)})
    LOGGER.info('Stage 5: synthesize natural-language rules and boundaries into predictive theory')
    theory = synthesize(rules, observations, groups, conditions, backend, cache)
    ledger = EvidenceLedger(o.case_id for o in observations)
    for obs in observations: ledger.record('structured_observation', 'source_observation', [obs.case_id], to_jsonable(obs))
    for rule in rules: ledger.record('natural_rules', 'rule_and_boundary', rule['construction_case_ids'], rule)
    artifacts = {'observations': to_jsonable(observations), 'conditions': to_jsonable(conditions),
        'action_groups': to_jsonable(groups), 'condition_comparisons': comparisons, 'boundary_discovery': boundaries,
        'theory': theory, 'evidence_ledger': ledger.to_dict(), 'semantic_alignment': {
            'action_alignment': grouped['semantic_alignment'], 'condition_alignment': [],
            'global_condition_alignment_enabled': False, 'rule_format_conversion_performed': False}}
    for name, value in artifacts.items(): write_json(output / (name + '.json'), value)
    stats = {'processed_observations': len(observations), 'discovered_methods': len(groups), 'candidate_rules': len(rules),
        'runtime_seconds': time.monotonic() - started, 'cache_hits': cache.hits, 'cache_misses': cache.misses,
        'rule_representation': 'natural_language', 'rule_format_conversion_performed': False,
        'global_condition_alignment_performed': False, 'global_condition_matrix_performed': False,
        'conditional_action_frequency_computed': False, 'citation_counts_only': True, 'boundary_workers': int(os.environ.get('SIMULATION_BOUNDARY_WORKERS', '4')), 'boundary_execution': 'parallel_independent_rules'}
    write_json(output / 'run_statistics.json', stats)
    write_json(output / 'run_manifest.json', {'status': 'complete', 'version': VERSION, 'config': config.to_dict(),
        'synthetic': synthetic, 'case_ids': [p.case_id for p in pairs], 'statistics': stats, 'artifacts': list(artifacts)})
    write_json(output / 'stage_progress.json', {'stage': 'complete', 'source_rules': len(rules),
        'theory_statements': len(theory['branches']), 'rule_representation': 'natural_language'})
    return artifacts
