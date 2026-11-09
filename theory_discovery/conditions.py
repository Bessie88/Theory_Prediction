"""Extensible predicate catalog and strictly before-only condition detection."""
from .llm import PLAIN_LANGUAGE_INSTRUCTION, checked_infer
import re
from .schemas import BeforeContext, Condition, EvidenceSpan, require, stable_id, to_jsonable, validate_value, resolve_span

CONDITION_TASK = 'Find up to four reusable Before-only features relevant to editing methods, not just topic names. When observed_actions is present, return one to four observable candidate features. Use the actions as focus hints. Look at wording, grammar, sentence structure, attribution, or scope, even when there is no neutrality issue. For example, when just was deleted, examine whether it is used for emphasis, minimization, or temporal meaning in this Before. If no semantic feature is clear, start with the concrete form or structural role of the Before target. For unchanged text, identify a visible wording or structure that could matter for retaining it. Ground every candidate in Before and leave generalization to later comparison. Infer no motives, unseen sources, or future edits. When observed_actions is absent, fewer candidates or none are allowed.\nReturn {"conditions":[{"name":str,"definition":str,"value_type":"boolean"|"number"|"string","detector":object,"discovery_note":str}]}.\nDetectors: {"kind":"semantic"}, {"kind":"contains","text":str,"case_sensitive":bool}, {"kind":"regex","pattern":str}, {"kind":"text_length"}, or {"kind":"context","key":str} for declared pre-edit keys. No extra detector fields. Define semantic criteria and explain relevance briefly. Python links case evidence; omit quotes/offsets.' + "\n" + PLAIN_LANGUAGE_INSTRUCTION


DETECTION_TASK = 'Evaluate the supplied conditions using only Before and available pre-edit context.\nReturn {"values":[{"condition_id":str,"value":bool|number|string|null}]}; return each requested ID once. Use null if undecidable, not false merely because external information is unavailable. Do not predict the edit or infer missing sources. Source case IDs are tracked by Python; no quotes or offsets are needed.' + "\n" + PLAIN_LANGUAGE_INSTRUCTION


class ConditionCatalog:
    def __init__(self, pairs, allowlist):
        self.pairs = pairs
        self.allowlist = allowlist
        self.conditions = {}

    def add(self, condition):
        condition.validate(self.pairs, self.allowlist)
        canonical = {'definition': condition.definition.strip(), 'detector': condition.detector,
                     'value_type': condition.value_type}
        cid = stable_id('condition', canonical)
        require(not condition.condition_id or condition.condition_id == cid, 'Condition ID does not match its definition')
        condition.condition_id = cid
        if cid in self.conditions:
            current = self.conditions[cid]
            for case_id, spans in condition.evidence_by_case.items():
                current.evidence_by_case.setdefault(case_id, []).extend(s for s in spans if s not in current.evidence_by_case.get(case_id, []))
        else:
            self.conditions[cid] = condition
        return cid

    def propose(self, context, case_id, backend):
        require(type(context) is BeforeContext, 'Condition proposer requires BeforeContext')
        raw = backend.infer(CONDITION_TASK, context.to_dict())
        require(set(raw) == {'conditions'} and isinstance(raw['conditions'], list), 'Invalid condition proposal')
        ids = []
        for item in raw['conditions']:
            item = dict(item)
            item.pop('evidence_spans', None)
            spans = []
            ids.append(self.add(Condition(condition_id='', evidence_by_case={case_id: spans}, **item)))
        return ids


class ConditionDetector:
    def __init__(self, backend=None):
        self.backend = backend

    def detect(self, context, conditions, case_id=None):
        return self.detect_with_evidence(context, conditions, case_id)[0]

    def detect_with_evidence(self, context, conditions, case_id=None):
        require(type(context) is BeforeContext, 'Detector accepts BeforeContext only')
        values, evidence, semantic = {}, {}, []
        for condition in conditions:
            cid, spec = condition.condition_id, condition.detector
            kind = spec['kind']
            spans = []
            if kind == 'contains':
                # Escaped regex preserves exact Unicode offsets during case-insensitive matching.
                flags = 0 if spec.get('case_sensitive', False) else re.IGNORECASE
                match = re.search(re.escape(spec['text']), context.before_text, flags)
                value = bool(match)
                if match:
                    spans = [EvidenceSpan('before', match.start(), match.end(), match.group())]
            elif kind == 'regex':
                match = re.search(spec['pattern'], context.before_text)
                value = bool(match)
                if match:
                    spans = [EvidenceSpan('before', match.start(), match.end(), match.group())]
            elif kind == 'text_length':
                value = len(context.before_text)
            elif kind == 'context':
                value = context.pre_context.get(spec['key'])
            elif kind == 'semantic':
                semantic.append(condition)
                continue
            else:
                raise ValueError('Unsupported condition detector')
            validate_value(value, condition.value_type)
            values[cid] = value
            evidence[cid] = {'kind': kind, 'spans': to_jsonable(spans), 'source': 'before_context'}
        if semantic and self.backend is None:
            for condition in semantic:
                values[condition.condition_id] = None
                evidence[condition.condition_id] = {'kind': 'semantic', 'reason': 'no_backend'}
        elif semantic:
            chunk_size = getattr(getattr(self.backend, 'config', None), 'condition_batch_size', 10)
            for offset in range(0, len(semantic), chunk_size):
                chunk = semantic[offset:offset + chunk_size]
                payload = {**context.to_dict(), 'conditions': [
                    {'condition_id': c.condition_id, 'definition': c.definition, 'value_type': c.value_type} for c in chunk]}
                chunk_values, chunk_evidence = checked_infer(self.backend, DETECTION_TASK, payload,
                    lambda raw: decode_semantic_values(context, chunk, raw, case_id))
                values.update(chunk_values)
                evidence.update(chunk_evidence)
        return values, evidence


def decode_semantic_values(context, semantic, raw, case_id=None):
    values, evidence = {}, {}
    expected = {c.condition_id: c for c in semantic}
    require(set(raw) == {'values'} and isinstance(raw['values'], list), 'Invalid detection response')
    returned = [row['condition_id'] for row in raw['values']]
    require(len(returned) == len(set(returned)) and set(returned) == set(expected), 'Missing/duplicate/unknown detector results')
    for item in raw['values']:
        require({'condition_id', 'value'} <= set(item) <= {'condition_id', 'value', 'evidence_spans', 'evidence_case_ids'}, 'Invalid detector fields')
        cid, value = item['condition_id'], item['value']
        validate_value(value, expected[cid].value_type)
        values[cid] = value
        evidence[cid] = {'kind': 'semantic', 'source': 'before_context', 'reference_level': 'case_id',
            'evidence_case_ids': [case_id] if case_id is not None else [], 'spans': [], 'absence_assessment': value is False}
    return values, evidence


CONDITION_REVIEW_TASK = 'Check that conditions are reusable, observable in Before and plausibly related to the supplied observed_actions when available. Reject topic inventories, inferred motives, unseen facts or future edits. Allow paraphrases and concrete lexical, grammatical or structural cues even when no neutrality issue is visible. When observed_actions is present, expect one to four candidates; with no semantic cue, look for an observable form or structural role. When observed_actions is absent, an empty list is allowed. Case-linked evidence is sufficient; acceptance is not empirical validation.\nReturn {"acceptable":bool,"issues":[str]}.' + "\n" + PLAIN_LANGUAGE_INSTRUCTION


def review_condition_quality(backend, context, proposals, *, action_hints=None):
    def decode(raw):
        require(set(raw) == {'acceptable', 'issues'} and type(raw['acceptable']) is bool
                and isinstance(raw['issues'], list) and all(isinstance(x, str) for x in raw['issues']), 'Invalid condition-quality review')
        require(raw['acceptable'] or bool(raw['issues']), 'Rejected condition review needs a reason')
        return raw
    payload = {**context.to_dict(), 'candidate_conditions': proposals['conditions']}
    if action_hints is not None:
        payload['observed_actions'] = action_hints
    return checked_infer(backend, CONDITION_REVIEW_TASK, payload, decode)
