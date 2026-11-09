"""Runtime-validated stage contracts. Canonical serialized form is JSON."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re


def stable_id(prefix, value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return prefix + '_' + hashlib.sha256(raw.encode()).hexdigest()[:20]


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class BeforeContext:
    before_text: str
    pre_context: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class EditingPair:
    case_id: str
    before_text: str
    after_text: str
    pre_context: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    split: str = 'discovery'

    def __post_init__(self):
        require(isinstance(self.case_id, str) and bool(self.case_id), 'Missing case_id')
        require(isinstance(self.before_text, str) and isinstance(self.after_text, str), 'Texts must be strings')
        require(isinstance(self.pre_context, dict) and isinstance(self.metadata, dict), 'Invalid context')
        require(self.split in ('discovery', 'development', 'test'), 'Invalid split')

    def before(self, allowlist):
        return BeforeContext(self.before_text, {k: v for k, v in self.pre_context.items() if k in allowlist})


@dataclass
class EvidenceSpan:
    side: str
    start: int
    end: int
    quote: str

    def validate(self, pair):
        require(self.side in ('before', 'after'), 'Invalid evidence side')
        text = pair.before_text if self.side == 'before' else pair.after_text
        require(type(self.start) is int and type(self.end) is int and 0 <= self.start <= self.end <= len(text), 'Invalid offsets')
        require(text[self.start:self.end] == self.quote, 'Evidence quote/offset mismatch')


@dataclass
class Action:
    action_id: str
    family: str
    method: str
    target_feature: str
    transformation: str
    scope: str
    target_expression: str
    evidence_spans: list[EvidenceSpan]
    diff_indices: list[int]
    semantic: bool = True
    uncertainty_flags: list[str] = field(default_factory=list)
    evidence_case_ids: list[str] = field(default_factory=list)


@dataclass
class Condition:
    condition_id: str
    name: str
    definition: str
    detector: dict
    evidence_by_case: dict[str, list[EvidenceSpan]]
    value_type: str = 'boolean'
    discovery_note: str = ''

    def validate(self, pairs, allowlist):
        require(bool(self.definition.strip()) and bool(self.name.strip()), 'Undefined condition')
        require(self.value_type in ('boolean', 'number', 'string'), 'Invalid condition value_type')
        kind = self.detector.get('kind')
        require(kind in ('contains', 'regex', 'text_length', 'context', 'semantic'), 'Unsupported detector')
        allowed = {'contains': {'kind', 'text', 'case_sensitive'}, 'regex': {'kind', 'pattern'},
                   'text_length': {'kind'}, 'context': {'kind', 'key'}, 'semantic': {'kind'}}[kind]
        # Semantic inference uses definition/value_type, not explanatory detector text.
        # Normalize known model annotations before canonical IDs are computed.
        annotations = {'query', 'description', 'explanation'} if kind == 'semantic' else set()
        extra = set(self.detector) - allowed
        require(extra <= annotations and all(isinstance(self.detector[k], str) for k in extra),
                f'Unexpected detector parameters for {kind}: {sorted(extra)}; allowed keys: {sorted(allowed)}')
        self.detector = {k: v for k, v in self.detector.items() if k in allowed}
        if kind == 'contains':
            require(isinstance(self.detector.get('text'), str) and bool(self.detector['text']), 'Empty contains detector')
            require(self.value_type == 'boolean', 'contains must be boolean')
        if kind == 'regex':
            require(isinstance(self.detector.get('pattern'), str) and len(self.detector['pattern']) <= 500, 'Invalid regex')
            re.compile(self.detector['pattern'])
            require(self.value_type == 'boolean', 'regex must be boolean')
        if kind == 'text_length':
            require(self.value_type == 'number', 'text_length must be numeric')
        if kind == 'context':
            require(self.detector.get('key') in allowlist, 'Context key not declared pre-edit')
        require(bool(self.evidence_by_case), 'Condition needs source case IDs')
        for cid, spans in self.evidence_by_case.items():
            require(cid in pairs and isinstance(spans, list), 'Unknown condition evidence case ID')
            for span in spans:
                require(span.side == 'before', 'Condition grounding must be before-edit')
                # Optional quote/offset details are descriptive; case ID is the evidence key.


@dataclass
class Observation:
    case_id: str
    before_text: str
    after_text: str
    actions: list[Action]
    candidate_conditions: list[str]
    condition_values: dict
    evidence_spans: list[EvidenceSpan]
    uncertainty_flags: list[str]
    pre_context: dict = field(default_factory=dict)
    split: str = 'discovery'
    condition_evidence: dict = field(default_factory=dict)


@dataclass
class ActionGroup:
    group_id: str
    normalized_action: dict
    member_case_ids: list[str]
    action_refs: list[dict]
    representative_edits: list[dict]
    group_size: int
    action_variations: list[str]
    grouping_uncertainty: list[str]
    member_action_signatures: list[dict] = field(default_factory=list)


def validate_value(value, value_type):
    if value is None:
        return
    valid = {'boolean': type(value) is bool, 'number': type(value) in (int, float) and math.isfinite(value),
             'string': type(value) is str}[value_type]
    require(valid, 'Condition result has wrong value type')


def to_jsonable(value):
    if hasattr(value, '__dataclass_fields__'):
        return asdict(value)
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def observation_from_dict(raw):
    raw = dict(raw)
    raw['actions'] = [Action(**dict(a, evidence_spans=[EvidenceSpan(**s) for s in a['evidence_spans']])) for a in raw['actions']]
    raw['evidence_spans'] = [EvidenceSpan(**s) for s in raw['evidence_spans']]
    return Observation(**raw)


def condition_from_dict(raw):
    raw = dict(raw)
    raw['evidence_by_case'] = {k: [EvidenceSpan(**s) for s in spans] for k, spans in raw['evidence_by_case'].items()}
    return Condition(**raw)


def resolve_span(raw, sources, preferred_regions=None):
    """Resolve exact evidence quotes; repair only unambiguous source coordinates."""
    span = EvidenceSpan(**raw)
    require(span.side in sources and isinstance(span.quote, str) and bool(span.quote), 'Invalid evidence source/quote')
    text = sources[span.side]
    if type(span.start) is int and type(span.end) is int and 0 <= span.start <= span.end <= len(text) and text[span.start:span.end] == span.quote:
        return span
    offsets = [match.start() for match in re.finditer(re.escape(span.quote), text)]
    selected = offsets[0] if len(offsets) == 1 else None
    if len(offsets) > 1 and preferred_regions:
        anchored = [offset for offset in offsets if any(offset < end and offset + len(span.quote) > begin
                    for begin, end in preferred_regions.get(span.side, []))]
        if len(anchored) == 1:
            selected = anchored[0]
    if selected is None and len(offsets) > 1 and type(span.start) is int:
        distances = sorted((abs(offset - span.start), offset) for offset in offsets)
        if distances[0][0] < distances[1][0] and distances[0][0] <= max(8, len(span.quote) // 2):
            selected = distances[0][1]
    require(selected is not None,
        f'Exact evidence quote {span.quote!r} on {span.side} occurs at {offsets}; supplied start={span.start}, end={span.end}. Return an existing quote with exact offsets; do not invent or paraphrase it.')
    return EvidenceSpan(span.side, selected, selected + len(span.quote), span.quote)
