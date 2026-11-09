"""Atomic artifacts, content-addressed cache, compact evidence ledger."""
import json
import os
from pathlib import Path
import tempfile
from .schemas import stable_id, to_jsonable

CACHE_VERSION = 'theory-discovery-8-operation-details'


def write_json(path, value):
    # Adapted from former wnc.common.write_json; no dependency on that package.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            temp = handle.name
            json.dump(to_jsonable(value), handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temp, path)
    finally:
        if temp and Path(temp).exists():
            Path(temp).unlink()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Cache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, stage, inputs, compute):
        envelope = {'version': CACHE_VERSION, 'stage': stage, 'inputs': to_jsonable(inputs)}
        key = stable_id('cache', envelope)
        path = self.directory / stage / (key + '.json')
        if path.exists():
            raw = read_json(path)
            if raw['key'] != key or raw['request'] != envelope or raw['value_hash'] != stable_id('value', raw['value']):
                raise ValueError('Cache integrity mismatch')
            self.hits += 1
            return raw['value']
        self.misses += 1
        value = to_jsonable(compute())
        write_json(path, {'key': key, 'request': envelope, 'value': value, 'value_hash': stable_id('value', value)})
        return value


class EvidenceLedger:
    def __init__(self, case_ids):
        self.case_ids = set(case_ids)
        self.events = []

    def record(self, stage, kind, case_ids, details):
        ids = sorted(set(case_ids))
        if not set(ids) <= self.case_ids:
            raise ValueError('Ledger contains unknown evidence IDs')
        event = {'stage': stage, 'kind': kind, 'case_ids': ids, 'details': to_jsonable(details)}
        event['event_id'] = stable_id('evidence', event)
        self.events.append(event)
        return event['event_id']

    def to_dict(self):
        return {'schema_version': '1', 'source_case_ids': sorted(self.case_ids), 'events': self.events}
