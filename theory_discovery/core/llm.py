"""Explicitly enabled Qwen/OpenAI-compatible JSON transport; inert on import."""
from collections import Counter
import json
import os
import time
import threading
import urllib.error
import urllib.request
from .schemas import require, stable_id
from .storage import write_json
from .reference_transport import short_reference_payload, restore_reference_ids

PLAIN_LANGUAGE_INSTRUCTION = (
    'Use plain language: short, concrete sentences, without jargon or repetition. '
    'Keep JSON keys, IDs and rule logic unchanged.'
)


class QwenJSONClient:
    def __init__(self, config, cache):
        self.config = config
        self.cache = cache
        self.verified = False
        self._call_limit = config.max_calls
        self._tokenizer = None
        self._lock = threading.RLock()
        self.stats = Counter(calls=0, retries=0, prompt_tokens=0, completion_tokens=0)

    @property
    def fingerprint(self):
        return {'adapter': 'qwen-json-v5-plain-language', 'config': vars(self.config)}

    def _request(self, suffix, payload=None):
        require(self.config.enabled, 'LLM inference disabled')
        headers = {'Content-Type': 'application/json'}
        token = os.environ.get(self.config.api_key_env)
        if token:
            headers['Authorization'] = 'Bearer ' + token
        req = urllib.request.Request(self.config.base_url.rstrip('/') + suffix,
                                     data=None if payload is None else json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
            return json.load(response)

    def verify_model(self):
        require(self.config.enabled and bool(self.config.model), 'Configure model before verification')
        models = self._request('/models')
        require(self.config.model in [m['id'] for m in models.get('data', [])], 'Configured model not advertised by endpoint')
        self.verified = True

    def prompt_tokens(self, messages):
        with self._lock:
            return self._count_prompt_tokens(messages)

    def _count_prompt_tokens(self, messages):
        if not self.config.tokenizer_path:
            return None
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, local_files_only=True)
        ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking, return_dict=False)
        return len(ids)

    def reserve_work_calls(self, minimum_remaining, stage):
        """Derive the safety ceiling from a finite workload without changing cache identity."""
        require(type(minimum_remaining) is int and minimum_remaining >= 0, 'Invalid call reservation')
        with self._lock:
            required = self.stats['calls'] + minimum_remaining
            if required > self._call_limit:
                previous = self._call_limit
                self._call_limit = required
                path = self.cache.directory / 'workload_call_budget.json'
                history = json.loads(path.read_text()) if path.exists() else []
                history.append({'stage': stage, 'configured_initial_limit': self.config.max_calls,
                                'previous_effective_limit': previous, 'effective_limit': required,
                                'remaining_attempt_allowance': minimum_remaining})
                write_json(path, history)

    def prompt_fits(self, task, payload, *, repair_reserve=False):
        messages = [{'role': 'system', 'content': task},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        reserve = self.config.max_tokens + self.config.context_safety_margin if repair_reserve else 0
        tokens = self.prompt_tokens(messages)
        return ((tokens is None or tokens + self.config.max_tokens + self.config.context_safety_margin + reserve <= self.config.context_length)
                and sum(len(m['content']) for m in messages) + reserve * 6 <= self.config.max_prompt_chars)

    def infer_checked(self, task, payload, decode):
        current_payload = payload
        for attempt in range(self.config.retries + 1):
            raw = self.infer(task, current_payload)
            try:
                return decode(raw)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                key = stable_id('invalid', {'task': task, 'payload': current_payload, 'response': raw})
                write_json(self.cache.directory / 'unresolved' / (key + '.json'),
                    {'task': task, 'payload': current_payload, 'response': raw, 'error': str(exc), 'attempt': attempt + 1})
                if attempt == self.config.retries:
                    raise ValueError('Structured response failed validation: ' + str(exc)) from exc
                current_payload = dict(payload, validation_feedback={
                    'error': str(exc), 'previous_response': raw, 'repair_attempt': attempt + 1,
                    'instruction': 'Fix the reported error using the required JSON schema. Keep supported content and all required source evidence fields.'})
        raise AssertionError('Unreachable')

    def infer(self, task, payload):
        if PLAIN_LANGUAGE_INSTRUCTION not in task:
            task += "\n" + PLAIN_LANGUAGE_INSTRUCTION
        require(self.config.enabled, 'LLM inference disabled')
        # Verification is required even when inference responses are cached.
        if not self.verified:
            self.verify_model()
        request = {'model': self.config.model, 'temperature': self.config.temperature,
                   'max_tokens': self.config.max_tokens, 'response_format': {'type': 'json_object'},
                   'chat_template_kwargs': {'enable_thinking': self.config.enable_thinking},
                   'messages': [{'role': 'system', 'content': task},
                                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}
        tokens = self.prompt_tokens(request['messages'])
        if tokens is not None:
            require(tokens + self.config.max_tokens + self.config.context_safety_margin <= self.config.context_length, '32K context budget exceeded; split the request without truncation')
        require(sum(len(m['content']) for m in request['messages']) <= self.config.max_prompt_chars,
                'Prompt character budget exceeded; no silent truncation')

        def generate():
            wire_payload, reference_aliases = short_reference_payload(payload)
            wire_request = dict(request, messages=[request['messages'][0],
                {'role': 'user', 'content': json.dumps(wire_payload, ensure_ascii=False)}])
            for attempt in range(self.config.retries + 1):
                with self._lock:
                    require(self.stats['calls'] < self._call_limit, 'LLM call budget exhausted')
                    self.stats['calls'] += 1
                try:
                    response = self._request('/chat/completions', wire_request)
                    require(response.get('model') == self.config.model, 'Response model mismatch')
                    choice = response['choices'][0]
                    require(choice['finish_reason'] == 'stop', 'Incomplete model response')
                    require(self.config.enable_thinking or not choice['message'].get('reasoning_content'), 'Thinking content returned despite no-thinking contract')
                    result = json.loads(choice['message']['content'])
                    require(isinstance(result, dict), 'Expected JSON object')
                    result = restore_reference_ids(result, reference_aliases)
                    usage = response.get('usage', {})
                    with self._lock:
                        for key in ('prompt_tokens', 'completion_tokens'):
                            self.stats[key] += usage.get(key, 0)
                    write_json(self.cache.directory / 'live_usage.json', {'model': self.config.model, 'enable_thinking': self.config.enable_thinking, 'context_length': self.config.context_length, 'last_prompt_tokens': tokens, 'stats': dict(self.stats), 'effective_call_limit': self._call_limit, 'reference_transport': 'request-local-short-labels-v1', 'reference_alias_count': len(reference_aliases)})
                    return {'result': result, 'usage': usage, 'reference_transport': 'request-local-short-labels-v1',
                            'short_id_map': reference_aliases, 'wire_request_hash': stable_id('wire', wire_request)}
                except urllib.error.HTTPError as exc:
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == self.config.retries:
                        raise RuntimeError(f'Model HTTP error {exc.code}') from None
                except (urllib.error.URLError, TimeoutError) as exc:
                    cause = getattr(exc, 'reason', exc)
                    timed_out = isinstance(cause, TimeoutError)
                    failure = {'model': self.config.model, 'request_hash': stable_id('wire', wire_request),
                               'attempt': attempt + 1, 'timeout_seconds': self.config.timeout_seconds,
                               'error_type': type(cause).__name__, 'error': str(cause), 'timed_out': timed_out}
                    write_json(self.cache.directory / 'transport_failures' /
                               (stable_id('transport_failure', [failure, time.time_ns()]) + '.json'), failure)
                    if attempt == self.config.retries:
                        if timed_out:
                            raise RuntimeError(f'Model request timed out after {self.config.timeout_seconds}s '
                                               f'({attempt + 1} attempts); see transport_failures') from exc
                        raise RuntimeError('Model connection failed: ' + type(cause).__name__ + ': ' + str(cause)) from exc
                except (ValueError, KeyError, IndexError, TypeError):
                    if attempt == self.config.retries:
                        raise ValueError('Invalid/incomplete model JSON contract') from None
                self.stats['retries'] += 1
                time.sleep(min(2 ** attempt, 8))
            raise AssertionError('Unreachable')

        return self.cache.get_or_compute('llm_responses', request, generate)['result']

    def infer_batches(self, task, payloads):
        # Bounded sequential batches; no GPU job or worker pool is created here.
        for offset in range(0, len(payloads), self.config.batch_size):
            yield [self.infer(task, payload) for payload in payloads[offset:offset + self.config.batch_size]]


def checked_infer(backend, task, payload, decode):
    if hasattr(backend, 'infer_checked'):
        return backend.infer_checked(task, payload, decode)
    return decode(backend.infer(task, payload))
