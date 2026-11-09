"""Integrated study contracts v9. No native induction schema is modified.

Strict codecs reject extra fields; records use immutable tuples and scalar metadata.
This module performs no I/O or model work at import time.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
import math
from types import UnionType
from typing import ClassVar, Union, get_args, get_origin, get_type_hints
from .schemas import stable_id, require

Scalar = str | int | float | bool | None
SCHEMA_VERSION = "integrated-9"


def encode(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [encode(v) for v in value]
    return value


def decode(annotation, value):
    """Strict, non-coercing JSON decoder, also used to validate constructors."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        for choice in args:
            try:
                return decode(choice, value)
            except (ValueError, TypeError):
                pass
        raise ValueError("Value does not match union")
    if annotation is type(None):
        require(value is None, "Expected null")
        return None
    if origin is tuple:
        require(isinstance(value, (list, tuple)), "Expected array")
        require(len(args) == 2 and args[1] is Ellipsis, "Only variable tuples supported")
        return tuple(decode(args[0], v) for v in value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and issubclass(annotation, Record):
        if type(value) is annotation:
            return value
        require(isinstance(value, dict), "Expected record object")
        expected = {f.name for f in fields(annotation)}
        require(set(value) == expected, "Missing or extra contract fields")
        hints = get_type_hints(annotation)
        decoded = {}
        for key, item in value.items():
            try:
                decoded[key] = decode(hints[key], item)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{annotation.__name__}.{key}: {exc}") from exc
        return annotation(**decoded)
    if annotation in (str, int, bool, float):
        require(type(value) is annotation, "Incorrect scalar type")
        if annotation is float:
            require(math.isfinite(value), "Non-finite number")
        return value
    raise TypeError("Unsupported contract annotation")


def contract_schema(annotation):
    """Machine-readable wire types from the same annotations used by strict decoding."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        return {"anyOf": [contract_schema(choice) for choice in args]}
    if origin is tuple:
        return {"type": "array", "items": contract_schema(args[0])}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [item.value for item in annotation]}
    if isinstance(annotation, type) and issubclass(annotation, Record):
        hints = get_type_hints(annotation)
        return {"type": "object", "properties": {f.name: contract_schema(hints[f.name]) for f in fields(annotation)},
                "required": [f.name for f in fields(annotation)], "additionalProperties": False}
    return {"type": {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}[annotation]}


def typed_equal(decoded, value):
    if type(decoded) is not type(value):
        return False
    if isinstance(decoded, tuple):
        return len(decoded) == len(value) and all(typed_equal(a, b) for a, b in zip(decoded, value))
    return decoded == value


class Record:
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self):
        hints = get_type_hints(type(self))
        for f in fields(self):
            value = getattr(self, f.name)
            decoded = decode(hints[f.name], value)
            # Constructor callers must use typed enums/tuples/records, not raw JSON.
            require(typed_equal(decoded, value), "Use typed constructor values")
        self.validate()

    def validate(self):
        pass

    def to_dict(self):
        return encode(self)

    @classmethod
    def from_dict(cls, raw):
        return decode(cls, raw)


def nonempty(*values):
    require(all(type(v) is str and v.strip() for v in values), "Missing required identity/text")


def unique(values):
    require(len(values) == len(set(values)), "Duplicate identifiers")


def utc_time(value):
    nonempty(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
            "Explicit UTC timestamp required")
    return parsed


class Period(str, Enum):
    EARLY = "EARLY"
    LATER = "LATER"


class Method(str, Enum):
    THEORY = "THEORY"
    NPOV = "NPOV"


class MeasurementMethod(str, Enum):
    CODE = "CODE"
    SEMANTIC = "SEMANTIC"
    HYBRID = "HYBRID"


class PredictionStatus(str, Enum):
    PREDICTED = "PREDICTED"
    NOT_COVERED = "NOT_COVERED"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


class ObservationStatus(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class AnnotationStatus(str, Enum):
    ANNOTATED = "ANNOTATED"
    UNKNOWN = "UNKNOWN"
    OTHER = "OTHER"
    UNCERTAIN = "UNCERTAIN"
    NO_CHANGE = "NO_CHANGE"


@dataclass(frozen=True)
class MetadataItem(Record):
    key: str
    value: Scalar

    def validate(self):
        nonempty(self.key)


@dataclass(frozen=True)
class SourceReference(Record):
    source_id: str
    location: str
    sha256: str
    json_pointer: str

    def validate(self):
        nonempty(self.source_id, self.location)
        require(len(self.sha256) == 64 and all(c in "0123456789abcdef" for c in self.sha256),
                "Source SHA-256 required")


@dataclass(frozen=True)
class EditorKey(Record):
    wiki_id: str
    editor_id: str

    def validate(self):
        nonempty(self.wiki_id, self.editor_id)


@dataclass(frozen=True)
class EditorRevision(Record):
    editor: EditorKey
    revision_id: str
    parent_revision_id: str | None
    timestamp: str | None
    page_id: str | None
    provenance: tuple[SourceReference, ...]
    before_reference: SourceReference | None
    after_reference: SourceReference | None
    editor_verified_human: bool
    missing_fields: tuple[str, ...]

    def validate(self):
        nonempty(self.revision_id)
        if self.timestamp is not None:
            utc_time(self.timestamp)
        for name, value in (("parent_revision_id", self.parent_revision_id),
                            ("timestamp", self.timestamp), ("page_id", self.page_id),
                            ("before_reference", self.before_reference),
                            ("after_reference", self.after_reference)):
            require(value is not None or name in self.missing_fields,
                    "Unavailable source field must be explicitly recorded")
        require(bool(self.provenance), "Source provenance required")


@dataclass(frozen=True)
class CaseIdentity(Record):
    editor: EditorKey
    revision_id: str
    case_id: str
    prediction_unit_id: str
    unit_policy_version: str
    period: Period

    @classmethod
    def create(cls, editor, revision_id, prediction_unit_id, unit_policy_version, period):
        # No row number or outcome participates in identity.
        case_id = stable_id("case", [editor.to_dict(), revision_id,
                                    prediction_unit_id, unit_policy_version])
        return cls(editor, revision_id, case_id, prediction_unit_id, unit_policy_version, period)

    def validate(self):
        nonempty(self.revision_id, self.case_id, self.prediction_unit_id, self.unit_policy_version)
        expected = stable_id("case", [self.editor.to_dict(), self.revision_id,
                                     self.prediction_unit_id, self.unit_policy_version])
        require(self.case_id == expected, "Case identity is not canonical")


@dataclass(frozen=True)
class ChronologicalSplitManifest(Record):
    editor: EditorKey
    cutoff: str
    early_revision_ids: tuple[str, ...]
    later_revision_ids: tuple[str, ...]
    cases: tuple[CaseIdentity, ...]
    split_configuration: tuple[MetadataItem, ...]
    source_provenance: tuple[SourceReference, ...]
    revision_timestamps: tuple[MetadataItem, ...]

    @property
    def manifest_id(self):
        return stable_id("chronological_manifest", [self.schema_version, self.to_dict()])

    def validate(self):
        cutoff = utc_time(self.cutoff)
        unique(self.early_revision_ids + self.later_revision_ids)
        require(bool(self.early_revision_ids) and bool(self.later_revision_ids),
                "Exactly two nonempty periods required")
        unique(tuple(item.key for item in self.revision_timestamps))
        times = {item.key: utc_time(item.value) for item in self.revision_timestamps}
        require(set(times) == set(self.early_revision_ids + self.later_revision_ids),
                "Every selected revision needs a timestamp")
        require(all(times[r] < cutoff for r in self.early_revision_ids) and
                all(times[r] >= cutoff for r in self.later_revision_ids), "Nonchronological split")
        require(self.early_revision_ids == tuple(sorted(self.early_revision_ids, key=lambda r: (times[r], r)))
                and self.later_revision_ids == tuple(sorted(self.later_revision_ids, key=lambda r: (times[r], r))),
                "Manifest revision IDs must follow canonical chronological ordering")
        unique(tuple(c.case_id for c in self.cases))
        for case in self.cases:
            require(case.editor == self.editor, "Mixed editor cohort")
            cohort = self.early_revision_ids if case.period is Period.EARLY else self.later_revision_ids
            require(case.revision_id in cohort, "Case period does not match revision")
        require(bool(self.source_provenance), "Source provenance required")


@dataclass(frozen=True)
class PreEditField(Record):
    name: str
    value: Scalar
    available_at_utc: str
    source_id: str

    def validate(self):
        nonempty(self.name, self.source_id)
        require(self.name not in {"after", "after_text", "diff", "actual_action",
                                  "future_revision", "edit_comment", "edit_size"},
                "Outcome/future metadata cannot be pre-edit context")
        utc_time(self.available_at_utc)


@dataclass(frozen=True)
class PredictionCase(Record):
    identity: CaseIdentity
    before_text: str
    before_hash: str
    context_as_of_utc: str
    pre_context: tuple[PreEditField, ...]

    def validate(self):
        from hashlib import sha256
        require(sha256(self.before_text.encode()).hexdigest() == self.before_hash,
                "Before hash mismatch")
        cutoff = utc_time(self.context_as_of_utc)
        unique(tuple(f.name for f in self.pre_context))
        require(all(utc_time(f.available_at_utc) <= cutoff for f in self.pre_context),
                "Future context forbidden")


@dataclass(frozen=True)
class EvidenceSpan(Record):
    side: str
    start: int
    end: int
    quote: str

    def validate(self):
        require(self.side in ("before", "after"), "Unknown evidence side")
        require(0 <= self.start <= self.end, "Invalid evidence offsets")

    def check_text(self, text):
        require(text[self.start:self.end] == self.quote and self.end <= len(text),
                "Evidence quote differs from source")


@dataclass(frozen=True)
class OutcomeCase(Record):
    identity: CaseIdentity
    before_text: str
    after_text: str | None
    provenance: tuple[SourceReference, ...]
    alignment_status: str = "UNSPECIFIED"
    unit_metadata: tuple[MetadataItem, ...] = ()

    def validate(self):
        require(self.alignment_status in (
            "UNSPECIFIED", "ALIGNED", "UNSUPPORTED_CROSS_UNIT",
            "UNSUPPORTED_BOUNDARY_INSERTION", "UNSUPPORTED_EMPTY_BEFORE"),
            "Unknown outcome alignment status")
        if self.alignment_status == "ALIGNED":
            require(self.after_text is not None, "Aligned outcome requires real After text")
        elif self.alignment_status.startswith("UNSUPPORTED"):
            require(self.after_text is None, "Unsupported unit must not fabricate an After")


@dataclass(frozen=True)
class ActionDefinition(Record):
    """Neutral vocabulary: no theories, conditions, frequencies or provenance paths."""
    action_id: str
    family: str
    method: str
    target_feature: str
    direction: str | None
    scope: str
    description: str
    version: str

    def validate(self):
        nonempty(self.action_id, self.family, self.method, self.scope, self.description, self.version)


@dataclass(frozen=True)
class ActionInstance(Record):
    action_id: str
    target_text: str | None
    description: str
    target_spans: tuple[EvidenceSpan, ...] = ()
    # OTHER is an outcome action ID, never a substitute for a missing theory mapping.

    def validate(self):
        nonempty(self.action_id, self.description)


@dataclass(frozen=True)
class ProseClause(Record):
    text: str
    source_rule_ids: tuple[str, ...]
    original_field_pointer: str
    resolution: str
    logical_role_hint: str | None = None
    original_clause_id: str | None = None

    def validate(self):
        require(self.resolution in ("UNPLANNED", "RESOLVED", "PARTIAL", "UNRESOLVED"),
                "Unknown clause resolution")


@dataclass(frozen=True)
class ActionOption(Record):
    action_ids: tuple[str, ...]
    source_rule_ids: tuple[str, ...]
    source_action_provenance_ids: tuple[str, ...]
    relation: str

    def validate(self):
        require(self.relation in ("UNSPECIFIED", "ALL_OF", "ANY_OF", "CONDITIONAL"),
                "Unknown action relationship")
        require(bool(self.source_rule_ids), "Action option needs source rules")


@dataclass(frozen=True)
class TheoryEvidenceLink(Record):
    case_id: str
    native_case_id: str
    editor: EditorKey
    revision_id: str
    prediction_unit_id: str


@dataclass(frozen=True)
class NativeRepresentation(Record):
    record_id: str
    record_type: str
    original_json: str
    provenance: SourceReference

    def validate(self):
        require(self.record_type in ("RULE", "CONDITION", "ACTION"), "Unknown native record type")


def theory_id_for(editor, native_branch_id, native_theory_hash):
    return stable_id("theory", [editor.to_dict(), native_branch_id, native_theory_hash])


@dataclass(frozen=True)
class TheoryRecord(Record):
    theory_id: str
    editor: EditorKey
    theory_text: str
    native_branch_id: str
    conditions: tuple[ProseClause, ...]
    boundaries: tuple[ProseClause, ...]
    actions: tuple[ActionOption, ...]
    source_case_ids: tuple[str, ...]
    source_rule_ids: tuple[str, ...]
    original_provenance: tuple[SourceReference, ...]
    provenance_status: str
    uncertainty: tuple[str, ...]
    source_revision_ids: tuple[str, ...] = ()
    evidence_links: tuple[TheoryEvidenceLink, ...] = ()
    original_representation_json: str = ""
    knowledge_source: Method = Method.THEORY

    def validate(self):
        nonempty(self.theory_id, self.theory_text, self.native_branch_id)
        require(bool(self.source_rule_ids) and bool(self.original_provenance),
                "Theory source lineage required")
        unique(self.source_case_ids)
        require({link.case_id for link in self.evidence_links} == set(self.source_case_ids),
                "Every source case needs a revision/unit evidence link")
        require({link.revision_id for link in self.evidence_links} == set(self.source_revision_ids),
                "Source revision IDs differ from evidence links")
        require(all(link.editor == self.editor for link in self.evidence_links),
                "Mixed-editor theory evidence")
        if self.knowledge_source is Method.THEORY:
            require(self.theory_id == theory_id_for(
                self.editor, self.native_branch_id, self.original_provenance[0].sha256),
                "Theory ID must be derived from editor, native branch and native theory hash")
        else:
            require(self.theory_id == self.native_branch_id and self.theory_id.startswith("npov_rule_") and
                    not self.source_case_ids and not self.evidence_links and not self.source_revision_ids,
                    "Policy knowledge must have independent policy IDs, never induced editing evidence")
        require(self.provenance_status in ("VERIFIED_SOURCE_LINKS", "PARTIAL", "UNRESOLVED"),
                "Unknown provenance status")


# Shared canonical shape; the knowledge_source tag distinguishes independent policy records.
KnowledgeRuleRecord = TheoryRecord


@dataclass(frozen=True)
class StructuredRow(Record):
    fields: tuple[MetadataItem, ...]


@dataclass(frozen=True)
class StructuredValue(Record):
    schema_name: str
    fields: tuple[MetadataItem, ...]
    rows: tuple[StructuredRow, ...]

    def validate(self):
        nonempty(self.schema_name)


MeasurementValue = Scalar | StructuredValue


@dataclass(frozen=True)
class StructuredField(Record):
    name: str
    value_type: str
    required: bool
    nullable: bool

    def validate(self):
        nonempty(self.name)
        require(self.value_type in ("string", "integer", "number", "boolean"), "Invalid structured field type")


@dataclass(frozen=True)
class FeatureSpecification(Record):
    feature_id: str
    associated_theory_ids: tuple[str, ...]
    associated_rule_ids: tuple[str, ...]
    roles: tuple[str, ...]
    measurement_method: MeasurementMethod
    operational_definition: str
    value_type: str
    allowed_values: tuple[Scalar, ...]
    decision_criteria: tuple[str, ...]
    evidence_requirements: tuple[str, ...]
    method_selection_rationale: str
    extractor_version: str
    prompt_version: str | None
    model_version: str | None
    source_provenance_ids: tuple[str, ...]
    feature_name: str = ""
    original_clause_ids: tuple[str, ...] = ()
    measurement_target: str = ""
    observable_indicators: tuple[str, ...] = ()
    positive_criteria: tuple[str, ...] = ()
    negative_criteria: tuple[str, ...] = ()
    ambiguity_criteria: tuple[str, ...] = ()
    required_input_fields: tuple[str, ...] = ()
    extraction_procedure_json: str = ""
    construct_kind: str = ""
    limitations: tuple[str, ...] = ()
    version: str = ""
    is_proxy: bool = False
    structured_schema_name: str = ""
    structured_fields: tuple[StructuredField, ...] = ()

    def validate(self):
        nonempty(self.feature_id, self.operational_definition, self.extractor_version,
                 self.method_selection_rationale)
        require(self.value_type in ("boolean", "number", "string", "categorical", "numerical", "structured"), "Invalid feature value type")
        require(bool(self.roles) and set(self.roles) <=
                {"THEORY_CONDITION", "THEORY_BOUNDARY", "NPOV_POLICY"}, "Invalid feature roles")
        require(bool(self.decision_criteria) and bool(self.evidence_requirements),
                "Feature criteria/evidence required")
        if self.measurement_method is MeasurementMethod.CODE:
            require(self.prompt_version is None and self.model_version is None,
                    "Objective feature cannot depend on LLM")
        else:
            nonempty(self.prompt_version, self.model_version)


def feature_observation_id_for(identity, feature_id, input_hash, specification_hash, method):
    return stable_id("feature_observation", [identity.case_id, feature_id, input_hash,
                                            specification_hash, method.value])


@dataclass(frozen=True)
class FeatureObservation(Record):
    observation_id: str
    identity: CaseIdentity
    feature_id: str
    extracted_value: MeasurementValue
    evidence: tuple[EvidenceSpan, ...]
    context_evidence_names: tuple[str, ...]
    extraction_method: MeasurementMethod
    status: ObservationStatus
    uncertainty: tuple[str, ...]
    extraction_metadata: tuple[MetadataItem, ...]
    input_hash: str
    specification_hash: str

    def validate(self):
        nonempty(self.observation_id, self.feature_id, self.input_hash, self.specification_hash)
        require(self.observation_id == feature_observation_id_for(
            self.identity, self.feature_id, self.input_hash, self.specification_hash,
            self.extraction_method), "Feature observation identity mismatch")
        require(all(e.side == "before" for e in self.evidence), "After evidence forbidden")
        if self.status is ObservationStatus.KNOWN:
            require(self.extracted_value is not None, "Known feature needs a value")
        else:
            require(self.extracted_value is None, "Unknown/error cannot carry a guessed value")


@dataclass(frozen=True)
class MatchPlan(Record):
    plan_id: str
    method: Method
    owner_id: str
    rule_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    # Declarative syntax validation/meaning is deliberately a later implementation.
    condition_expression_json: str
    exclusion_expression_json: str
    action_options: tuple[ActionOption, ...]
    unresolved_clauses: tuple[str, ...]
    version: str


@dataclass(frozen=True)
class ApplicabilityDecision(Record):
    identity: CaseIdentity
    method: Method
    plan_id: str
    owner_id: str
    rule_ids: tuple[str, ...]
    status: str
    observation_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    decision_id: str = ""

    def validate(self):
        require(self.status in ("TRUE", "FALSE", "UNKNOWN", "CONFLICT"), "Invalid applicability")


@dataclass(frozen=True)
class ActionPrediction(Record):
    identity: CaseIdentity
    method: Method
    applicable_theory_ids: tuple[str, ...]
    applicable_rule_ids: tuple[str, ...]
    predicted_actions: tuple[ActionInstance, ...]
    prediction_status: PredictionStatus
    matched_feature_ids: tuple[str, ...]
    feature_observation_ids: tuple[str, ...]
    study_spec_hash: str
    original_input_hash: str
    reason_codes: tuple[str, ...]
    applicability_decision_ids: tuple[str, ...] = ()
    provenance_id: str = ""
    not_applicable_theory_ids: tuple[str, ...] = ()
    unknown_theory_ids: tuple[str, ...] = ()
    action_trace_ids: tuple[str, ...] = ()
    condition_feature_ids: tuple[str, ...] = ()
    boundary_feature_ids: tuple[str, ...] = ()
    missing_feature_ids: tuple[str, ...] = ()
    applicable_policy_rule_ids: tuple[str, ...] = ()
    not_applicable_policy_rule_ids: tuple[str, ...] = ()
    unknown_policy_rule_ids: tuple[str, ...] = ()

    def validate(self):
        require(self.identity.period is Period.LATER, "Prediction requires Later case")
        if self.prediction_status is PredictionStatus.PREDICTED:
            require(bool(self.predicted_actions) and bool(self.applicable_rule_ids),
                    "Prediction must retain action and rule lineage")
        else:
            require(not self.predicted_actions, "Uncertain predictions cannot masquerade as confident actions")
        if self.method is Method.NPOV:
            require(not self.applicable_theory_ids and not self.not_applicable_theory_ids and
                    not self.unknown_theory_ids, "NPOV cannot cite theory applicability")
        else:
            require(not self.applicable_policy_rule_ids and not self.not_applicable_policy_rule_ids and
                    not self.unknown_policy_rule_ids, "Theory prediction cannot cite policy applicability")
        require(all(span.side == "before" for action in self.predicted_actions for span in action.target_spans),
                "Predicted action targets must be pre-edit evidence")
        nonempty(self.study_spec_hash, self.original_input_hash)


@dataclass(frozen=True)
class ObservedOperation(Record):
    action: ActionInstance
    diff_hunk_ids: tuple[str, ...]
    before_evidence: tuple[EvidenceSpan, ...]
    after_evidence: tuple[EvidenceSpan, ...]
    explanation: str
    confidence: str
    uncertainty: tuple[str, ...]

    def validate(self):
        nonempty(self.explanation)
        require(self.confidence in ("HIGH", "MEDIUM", "LOW", "UNCERTAIN"), "Invalid confidence")
        require(all(e.side == "before" for e in self.before_evidence) and
                all(e.side == "after" for e in self.after_evidence), "Wrong action evidence side")


@dataclass(frozen=True)
class ObservedAction(Record):
    identity: CaseIdentity
    observed_actions: tuple[ActionInstance, ...]
    actual_diff_evidence: tuple[EvidenceSpan, ...]
    annotation_status: AnnotationStatus
    annotation_metadata: tuple[MetadataItem, ...]
    annotator_version: str
    annotation_id: str = ""
    operations: tuple[ObservedOperation, ...] = ()
    explanation: str = ""
    confidence: str = "UNCERTAIN"
    uncertainty: tuple[str, ...] = ()
    annotator_model: str = ""
    annotator_model_family: str | None = None
    annotator_model_version: str | None = None
    prompt_version: str = ""
    input_hash: str = ""
    protocol_hash: str = ""
    unassigned_diff_hunk_ids: tuple[str, ...] = ()

    def validate(self):
        require(self.identity.period is Period.LATER, "Observed action requires Later case")
        nonempty(self.annotator_version)
        if self.annotation_id:
            nonempty(self.input_hash, self.protocol_hash, self.explanation, self.prompt_version, self.annotator_model)
            require(tuple(o.action for o in self.operations) == self.observed_actions, "Annotation actions/evidence differ")
            require(self.confidence in ("HIGH", "MEDIUM", "LOW", "UNCERTAIN"), "Invalid annotation confidence")
            if self.annotation_status in (AnnotationStatus.UNKNOWN, AnnotationStatus.NO_CHANGE):
                require(not self.observed_actions, "Unavailable/no-change record cannot invent an operation")


@dataclass(frozen=True)
class HumanReviewRecord(Record):
    identity: CaseIdentity
    model_annotation: ObservedAction
    reviewer_decision: str | None
    corrected_label: tuple[ActionInstance, ...] | None
    review_status: str
    reviewer_id: str | None
    rationale: str | None
    review_mode: str = "UNASSIGNED"
    packet_id: str = ""
    response_id: str = ""
    completed_at_utc: str | None = None
    notes: str = ""
    reviewed_operations: tuple[ObservedOperation, ...] = ()

    def validate(self):
        require(self.identity == self.model_annotation.identity, "Review case mismatch")
        require(self.review_status in ("PENDING", "CONFIRMED", "CORRECTED", "UNRESOLVED", "NOT_SELECTED"),
                "Invalid review state")
        if self.review_status in ("CONFIRMED", "CORRECTED"):
            nonempty(self.reviewer_id, self.reviewer_decision)
        if self.review_status == "CORRECTED":
            require(self.corrected_label is not None, "Correction cannot omit corrected labels")
        if self.review_status in ("CONFIRMED", "CORRECTED", "UNRESOLVED"):
            nonempty(self.reviewer_id, self.reviewer_decision, self.packet_id, self.response_id, self.completed_at_utc)
            utc_time(self.completed_at_utc)
            require(self.review_mode in ("BLIND_ANNOTATION", "LABEL_VERIFICATION"), "Review mode required")
        else:
            require(self.reviewer_id is None and self.completed_at_utc is None and not self.response_id and
                    self.corrected_label is None and not self.reviewed_operations,
                    "Pending/unselected is not a completed human review")

    @property
    def human_verified(self):
        return self.review_status in ("CONFIRMED", "CORRECTED") and bool(self.response_id)


@dataclass(frozen=True)
class Metric(Record):
    name: str
    value: float | None
    numerator: int
    denominator: int

    def validate(self):
        nonempty(self.name)
        require(self.denominator >= 0 and self.numerator >= 0, "Negative metric counts")
        if self.denominator == 0:
            require(self.value is None, "Empty denominator must be null")


@dataclass(frozen=True)
class RuleAnalysis(Record):
    rule_id: str
    metrics: tuple[Metric, ...]


@dataclass(frozen=True)
class EvaluationResult(Record):
    method: Method
    editor: EditorKey
    case_ids: tuple[str, ...]
    coverage: Metric
    prediction_metrics: tuple[Metric, ...]
    per_rule_analysis: tuple[RuleAnalysis, ...]
    disagreement_statistics: tuple[Metric, ...]
    revision_cluster_metrics: tuple[Metric, ...]
    evaluation_version: str

    def validate(self):
        unique(self.case_ids)
        nonempty(self.evaluation_version)


def require_same_cases(*groups):
    """Exact identity join: no row-order join and no dropped/duplicate cases."""
    require(bool(groups), "At least one case collection required")
    maps = []
    for group in groups:
        index = {row.identity.case_id: row.identity for row in group}
        require(len(index) == len(group), "Duplicate case IDs")
        maps.append(index)
    require(all(index == maps[0] for index in maps[1:]), "Case identities differ across branches")
