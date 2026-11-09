"""Record model-output failures and repartition exhaustive retrieval without losing cases."""
import json
from copy import deepcopy
import logging
import os
from .llm import QwenJSONClient, checked_infer
from .schemas import stable_id, require
from .storage import write_json, read_json
from .relaxed_json import recover_content

LOGGER = logging.getLogger(__name__)


class ModelOutputError(RuntimeError):
    """A completed HTTP request returned truncated or malformed structured output."""


class RecoverableQwenJSONClient(QwenJSONClient):
    # Inherit fingerprint, request construction, short-ID transport and cache keys.
    def _request(self, suffix, payload=None):
        response = super()._request(suffix, payload)
        if suffix != '/chat/completions':
            return response
        failure = None
        try:
            choice = response['choices'][0]
            finish = choice['finish_reason']
            content = choice['message']['content']
            wire_payload = json.loads(payload['messages'][1]['content'])
            try:
                result, repairs = recover_content(content, wire_payload)
            except (ValueError, TypeError) as exc:
                result, repairs = None, []
                failure = 'output_length_limit' if finish == 'length' else 'malformed_model_json: ' + str(exc)
            if result is not None:
                if finish == 'length':
                    repairs.append('complete_core_despite_length_finish')
                if finish in ('stop', 'length') and (repairs or content != json.dumps(result)):
                    original = deepcopy(response)
                    normalized = deepcopy(response)
                    normalized['choices'][0]['finish_reason'] = 'stop'
                    normalized['choices'][0]['message']['content'] = json.dumps(result, ensure_ascii=False)
                    if repairs:
                        key = stable_id('output_recovery', {'request': payload, 'response': original})
                        write_json(self.cache.directory / 'model_output_recoveries' / (key + '.json'),
                            {'wire_request': payload, 'response': original, 'recovered_result': result,
                             'repairs': repairs, 'semantic_decisions_invented': False})
                    return normalized
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            failure = 'malformed_model_json: ' + str(exc)
        if failure:
            key = stable_id('output_failure', {'request': payload, 'response': response})
            path = self.cache.directory / 'model_output_failures' / (key + '.json')
            write_json(path, {'reason': failure, 'wire_request': payload, 'response': response,
                'job_id': os.environ.get('SLURM_JOB_ID') or os.environ.get('SIMULATION_JOB_ID'),
                'configured_max_tokens': self.config.max_tokens})
            with self._lock:
                self.stats['output_failures'] += 1
                for name in ('prompt_tokens', 'completion_tokens'):
                    self.stats[name] += response.get('usage', {}).get(name, 0)
            raise ModelOutputError(f'{failure}; preserved response: {path}')
        return response


def _partition_path(cache, stage, task, payload, fingerprint):
    key = stable_id('output_partition', {'version': 'response-recovery-v1', 'stage': stage,
        'task': task, 'payload': payload, 'backend': fingerprint})
    return cache.directory / 'adaptive_response_partitions' / (key + '.json')


def output_size_hint(backend, stage, task, payload):
    cache = getattr(backend, 'cache', None)
    if cache is None:
        return None
    path = _partition_path(cache, stage, task, payload, backend.fingerprint)
    return read_json(path).get('maximum_items') if path.exists() else None


def remember_smaller_output(backend, stage, task, payload, size, error):
    require(size > 1, 'Cannot split a single evidence item')
    maximum = max(1, size // 2)
    cache = getattr(backend, 'cache', None)
    if cache is not None:
        write_json(_partition_path(cache, stage, task, payload, backend.fingerprint),
            {'status': 'split_required', 'maximum_items': maximum,
             'original_items': size, 'error': str(error), 'coverage_policy': 'all_items_retained'})
    LOGGER.warning('%s output failed; retrying with at most %d items, retaining all evidence',
                   stage, maximum)
    return maximum


def recoverable_cached_call(backend, cache, stage, version, task, payload, decode):
    """Keep old successful cache identities; split only an actually failing retrieval."""
    fingerprint = backend.fingerprint
    path = _partition_path(cache, stage, task, payload, fingerprint)
    rows = payload.get('outside_condition_profiles') if stage == 'natural_similar_outside' else None

    def call(current_payload):
        if hasattr(backend, 'reserve_work_calls'):
            backend.reserve_work_calls((backend.config.retries + 1) ** 2, stage)
        return checked_infer(backend, task, current_payload, decode)

    def split():
        require(rows is not None and len(rows) > 1, 'No exhaustive retrieval partition is available')
        middle = len(rows) // 2
        children = []
        for chunk in (rows[:middle], rows[middle:]):
            child_payload = dict(payload, outside_condition_profiles=chunk)
            shown = {row['case_id'] for row in chunk}
            def child_decode(raw):
                result = decode(raw)
                require(set(result['similar_case_ids']) <= shown, 'Child retrieval invented a case outside its own packet')
                return result
            child = recoverable_cached_call(backend, cache, stage, version, task, child_payload, child_decode)
            children.append({'considered_case_ids': [r['case_id'] for r in chunk], 'result': child})
        expected = [row['case_id'] for row in rows]
        actual = [cid for child in children for cid in child['considered_case_ids']]
        require(actual == expected, 'Adaptive retrieval dropped, reordered or duplicated an eligible case')
        wanted = {cid for child in children for cid in child['result']['similar_case_ids']}
        result = {'similar_case_ids': [cid for cid in expected if cid in wanted],
                  'reason': '\n'.join(child['result']['reason'] for child in children)}
        write_json(path, {'status': 'complete', 'original_items': len(rows),
            'maximum_items': middle, 'children': children, 'all_eligible_cases_covered': True})
        return decode(result)

    def compute():
        if rows is not None and len(rows) > 1 and path.exists():
            return split()
        try:
            return call(payload)
        except ModelOutputError as exc:
            if rows is not None and len(rows) > 1:
                write_json(path, {'status': 'split_required', 'original_items': len(rows),
                    'maximum_items': len(rows) // 2, 'error': str(exc)})
                LOGGER.warning('Retrieval output failed for %d cases; exhaustively splitting both halves', len(rows))
                return split()
            # A small atomic response can be repaired without guessing missing content.
            # Multi-item boundary/theory packets are split by their stateful callers.
            size = (max(len(payload.get('own_support_case_ids', [])), len(payload.get('matched_outside_case_ids', []))) if stage == 'natural_boundary_packets'
                    else len(payload.get('rules', [])) if stage == 'natural_theory_packets'
                    else len(payload.get('theory_situations', [])) if stage == 'natural_heldout_choices'
                    else len(rows or []))
            if size > 1:
                raise
            retries = getattr(getattr(backend, 'config', None), 'retries', 3)
            for attempt in range(retries):
                repaired = dict(payload, output_repair={
                    'attempt': attempt + 1,
                    'instruction': 'Return only the required complete JSON object. Keep the same evidence and decision. Do not repeat IDs. Keep reason/prose brief; no extra analysis.'})
                try:
                    return call(repaired)
                except ModelOutputError:
                    if attempt + 1 == retries:
                        raise
            raise

    return cache.get_or_compute(stage, {'version': version, 'task': task,
        'payload': payload, 'backend': fingerprint}, compute)
