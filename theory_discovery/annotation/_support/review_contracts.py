"""Separate reviewer-facing packets, coordinator selection audit and human submissions."""
from dataclasses import dataclass
from theory_discovery.core.workflow_contracts import (
    Record, CaseIdentity, PredictionCase, MeasurementValue, StructuredField, MeasurementMethod,
    EvidenceSpan, ActionInstance, ObservedOperation, PreEditField, Scalar, MetadataItem, utc_time,
)
from .annotation_contracts import AnnotationResponse, TextDiff
from .action_comparison import FrozenActionProtocol
from theory_discovery.core.schemas import require

MODES = ("BLIND_ANNOTATION", "LABEL_VERIFICATION")
STRATA = ("RANDOM", "THEORY_ONLY_SUCCESS", "NPOV_ONLY_SUCCESS", "BOTH_SUCCESS", "BOTH_FAILURE",
          "NOT_COVERED", "UNKNOWN", "CONFLICT", "RARE_ACTION")


@dataclass(frozen=True)
class SamplingQuota(Record):
    stratum: str
    limit: int

    def validate(self):
        require(self.stratum in STRATA and self.limit >= 0, "Invalid review sampling quota")


@dataclass(frozen=True)
class ReviewSettings(Record):
    action_mode: str = "BLIND_ANNOTATION"
    feature_mode: str = "BLIND_ANNOTATION"
    seed: str = "unconfigured"
    quotas: tuple[SamplingQuota, ...] = ()
    feature_limit: int = 0
    rare_action_max_cases: int = 2
    permitted_reviewers: tuple[str, ...] = ()

    def validate(self):
        require(self.action_mode in MODES and self.feature_mode in MODES and self.feature_limit >= 0 and
                self.rare_action_max_cases > 0, "Invalid review settings")
        require(len({q.stratum for q in self.quotas}) == len(self.quotas), "Duplicate sampling stratum")


@dataclass(frozen=True)
class ReviewAssignment(Record):
    identity: CaseIdentity
    subject_id: str
    mode: str


@dataclass(frozen=True)
class ReviewAssignments(Record):
    assignments: tuple[ReviewAssignment, ...]


@dataclass(frozen=True)
class SamplingAuditRow(Record):
    identity: CaseIdentity
    strata: tuple[str, ...]
    selected: bool


@dataclass(frozen=True)
class ReviewSamplingAudit(Record):
    seed: str
    rows: tuple[SamplingAuditRow, ...]
    stratum_counts: tuple[MetadataItem, ...]
    interpretation: str


@dataclass(frozen=True)
class FeatureReviewDefinition(Record):
    feature_id: str
    feature_name: str
    operational_definition: str
    value_type: str
    allowed_values: tuple[Scalar, ...]
    decision_criteria: tuple[str, ...]
    positive_criteria: tuple[str, ...]
    negative_criteria: tuple[str, ...]
    ambiguity_criteria: tuple[str, ...]
    evidence_requirements: tuple[str, ...]
    structured_schema_name: str
    structured_fields: tuple[StructuredField, ...]
    measurement_method: MeasurementMethod
    limitations: tuple[str, ...]
    version: str


@dataclass(frozen=True)
class FeatureReviewLabel(Record):
    value: MeasurementValue
    status: str
    evidence: tuple[EvidenceSpan, ...]
    context_evidence_names: tuple[str, ...]
    evidence_scope: str
    explanation: str
    uncertainty: tuple[str, ...]


@dataclass(frozen=True)
class FeatureReviewSourceRecord(Record):
    identity: CaseIdentity
    subject_id: str
    case: PredictionCase
    definition: FeatureReviewDefinition
    model_label: FeatureReviewLabel
    observation_hash: str
    specification_hash: str


@dataclass(frozen=True)
class FeatureReviewSource(Record):
    records: tuple[FeatureReviewSourceRecord, ...]


@dataclass(frozen=True)
class ActionReviewLabel(Record):
    status: str
    operations: tuple[ObservedOperation, ...]
    explanation: str
    confidence: str
    uncertainty: tuple[str, ...]
    unassigned_diff_hunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReviewItem(Record):
    task_id: str
    subject_id: str
    identity: CaseIdentity
    mode: str
    before_text: str
    after_text: str | None
    actual_diff: TextDiff | None
    context: tuple[PreEditField, ...]
    availability: str
    limitations: tuple[str, ...]
    feature_definition: FeatureReviewDefinition | None
    action_label: ActionReviewLabel | None
    feature_label: FeatureReviewLabel | None

    def validate(self):
        require(self.mode in MODES, "Unknown review mode")
        if self.mode == "BLIND_ANNOTATION":
            require(self.action_label is None and self.feature_label is None,
                    "Blind packet must omit model label, explanation and model-selected evidence")


@dataclass(frozen=True)
class ReviewPacket(Record):
    packet_id: str
    kind: str
    protocol: FrozenActionProtocol
    items: tuple[ReviewItem, ...]

    def validate(self):
        require(self.kind in ("ACTION", "FEATURE"), "Unknown review target kind")
        require(len({i.task_id for i in self.items}) == len(self.items) and
                len({i.subject_id for i in self.items}) == len(self.items), "Duplicate review task/subject")
        require(len({i.mode for i in self.items}) <= 1, "Do not mix blind and visible-label tasks in one packet")
        if self.kind == "FEATURE":
            require(all(i.after_text is None and i.actual_diff is None and i.action_label is None and
                        i.feature_definition is not None for i in self.items),
                    "Feature reviewer must see Before only")
        else:
            require(all(i.feature_definition is None and i.feature_label is None for i in self.items),
                    "Action review must not expose feature/theory mappings")


@dataclass(frozen=True)
class ReviewExport(Record):
    packet: ReviewPacket
    html_document: str
    response_template_json: str


@dataclass(frozen=True)
class HumanResponse(Record):
    task_id: str
    reviewer_id: str
    completed_at_utc: str
    mode: str
    decision: str
    action_annotation: AnnotationResponse | None
    feature_annotation: FeatureReviewLabel | None
    notes: str

    def validate(self):
        require(self.reviewer_id.strip() and self.mode in MODES, "Real reviewer identity/mode required")
        utc_time(self.completed_at_utc)
        allowed = ("ASSIGN", "UNCERTAIN") if self.mode == "BLIND_ANNOTATION" else ("ACCEPT", "CORRECT", "UNCERTAIN")
        require(self.decision in allowed, "Blind reviewers must assign independently before seeing a model label")


@dataclass(frozen=True)
class HumanResponseBatch(Record):
    packet_id: str
    responses: tuple[HumanResponse, ...]

    def validate(self):
        require(len({r.task_id for r in self.responses}) == len(self.responses), "Duplicate completed human response")


@dataclass(frozen=True)
class FeatureHumanReview(Record):
    identity: CaseIdentity
    subject_id: str
    model_label: FeatureReviewLabel
    corrected_label: FeatureReviewLabel | None
    review_status: str
    mode: str
    reviewer_id: str | None
    completed_at_utc: str | None
    response_id: str
    packet_id: str
    notes: str

    def validate(self):
        require(self.review_status in ("PENDING", "NOT_SELECTED", "CONFIRMED", "CORRECTED", "UNRESOLVED"),
                "Invalid feature review status")
        if self.review_status in ("CONFIRMED", "CORRECTED", "UNRESOLVED"):
            require(bool(self.reviewer_id) and self.completed_at_utc and self.response_id and self.packet_id,
                    "Real completed human submission required")
            utc_time(self.completed_at_utc)
            require(self.mode in MODES, "Completed human feature review mode required")
            require(self.review_status != "CORRECTED" or self.corrected_label is not None, "Correction missing")
        else:
            require(self.reviewer_id is None and self.completed_at_utc is None and not self.response_id and
                    self.corrected_label is None, "Pending is not human verified")

    @property
    def human_verified(self):
        return self.review_status in ("CONFIRMED", "CORRECTED") and bool(self.response_id)


@dataclass(frozen=True)
class FeatureHumanReviews(Record):
    records: tuple[FeatureHumanReview, ...]


@dataclass(frozen=True)
class ReviewImportAudit(Record):
    submissions: HumanResponseBatch | None


@dataclass(frozen=True)
class ReviewSubmission(Record):
    editor: "EditorKey"
    action_responses: HumanResponseBatch | None = None
    feature_responses: HumanResponseBatch | None = None


from theory_discovery.core.workflow_contracts import EditorKey
