"""Evaluation/analysis records. No computation or source I/O on import."""
from dataclasses import dataclass
from theory_discovery.core.workflow_contracts import (
    Record, CaseIdentity, EditorKey, EditorRevision, SourceReference, ActionInstance,
    Method, Metric, MetadataItem, EvidenceSpan,
)
from theory_discovery.core.schemas import require

@dataclass(frozen=True)
class EvaluationContext(Record):
    revisions: tuple[EditorRevision, ...]
    early_revision_count: int
    later_revision_count: int
    cutoff_utc: str
    unit_version: str
    source_references: tuple[SourceReference, ...]

@dataclass(frozen=True)
class GoldLabel(Record):
    identity: CaseIdentity
    actions: tuple[ActionInstance, ...]
    scorable: bool
    source: str
    status: str
    human_status: str
    annotation_id: str
    response_id: str
    reason: str

@dataclass(frozen=True)
class CaseScore(Record):
    identity: CaseIdentity
    method: Method
    prediction_status: str
    usable: bool
    gold_scorable: bool
    match: str
    predicted_ids: tuple[str, ...]
    actual_ids: tuple[str, ...]
    matched_owner_ids: tuple[str, ...]
    gold_source: str
    reason: str

@dataclass(frozen=True)
class LabelPerformance(Record):
    action_id: str
    tp: int
    fp: int
    fn: int
    support: int
    precision: float | None
    recall: float | None
    f1: float | None

@dataclass(frozen=True)
class MethodPerformance(Record):
    method: Method
    case_ids: tuple[str, ...]
    metrics: tuple[Metric, ...]
    labels: tuple[LabelPerformance, ...]
    macro_f1: float | None
    macro_label_ids: tuple[str, ...]
    counts: tuple[MetadataItem, ...]
    owner_metrics: tuple["OwnerPerformance", ...]

@dataclass(frozen=True)
class OwnerPerformance(Record):
    owner_id: str
    matched_cases: int
    scored_matched_cases: int
    usable_matched_cases: int
    exact_matches: int
    accuracy: float | None
    attribution: str

@dataclass(frozen=True)
class EvaluationDetails(Record):
    editor: EditorKey
    protocol_hash: str
    evaluation_protocol_hash: str
    study_hash: str
    gold: tuple[GoldLabel, ...]
    scores: tuple[CaseScore, ...]
    methods: tuple[MethodPerformance, ...]
    model_reference_methods: tuple[MethodPerformance, ...]
    human_only_methods: tuple[MethodPerformance, ...]
    annotation_reliability: tuple[Metric, ...]
    reliability_by_mode: tuple[MetadataItem, ...]
    feature_review_counts: tuple[MetadataItem, ...]
    limitations: tuple[str, ...]

@dataclass(frozen=True)
class AblationChange(Record):
    identity: CaseIdentity
    with_status: str
    without_status: str
    with_action_ids: tuple[str, ...]
    without_action_ids: tuple[str, ...]
    with_match: str
    without_match: str
    excluded: bool
    error_prevented_by_abstention: bool
    changed: bool

@dataclass(frozen=True)
class BoundaryAnalysis(Record):
    original_study_hash: str
    feature_artifact_hash: str
    protocol: str
    with_boundary: MethodPerformance
    without_boundary: MethodPerformance
    changes: tuple[AblationChange, ...]
    limitations: tuple[str, ...]
    without_predictions_json: str
    without_provenance_json: str

@dataclass(frozen=True)
class DisagreementCase(Record):
    identity: CaseIdentity
    category: str
    theory_status: str
    npov_status: str
    theory_match: str
    npov_match: str
    abstention_statuses: tuple[str, ...]

@dataclass(frozen=True)
class QualitativeCase(Record):
    identity: CaseIdentity
    category: str
    revision_metadata_json: str
    before_text: str
    after_text: str | None
    actual_diff_json: str
    theory_json: str
    conditions_boundaries_json: str
    theory_prediction_json: str
    npov_rules_json: str
    npov_prediction_json: str
    raw_annotation_json: str
    human_review_json: str
    evidence_json: str

@dataclass(frozen=True)
class PatternCount(Record):
    category: str
    pattern: str
    case_ids: tuple[str, ...]
    revision_ids: tuple[str, ...]
    page_ids: tuple[str, ...]
    missing_page_cases: int

@dataclass(frozen=True)
class EditorAnalysis(Record):
    editor: EditorKey
    disagreements: tuple[DisagreementCase, ...]
    qualitative_cases: tuple[QualitativeCase, ...]
    recurring_patterns: tuple[PatternCount, ...]
    temporal_bins: tuple[MetadataItem, ...]
    confounding_counts: tuple[MetadataItem, ...]
    interpretation: str

@dataclass(frozen=True)
class ResearchReport(Record):
    editor: EditorKey
    markdown: str
    template_version: str
    evaluation_hash: str
    case_ids: tuple[str, ...]
    action_protocol_hash: str
    evaluation_protocol_hash: str
    theory_registry_hash: str
    performance_json: str

@dataclass(frozen=True)
class TransferComparison(Record):
    source_editor: EditorKey
    target_editor: EditorKey
    source_registry_hash: str
    target_case_ids: tuple[str, ...]
    target_in_source_training: bool
    protocol_id: str
    same_case_reference_accuracy: float | None
    transfer_accuracy: float | None
    matched_topic_control: bool

@dataclass(frozen=True)
class CohortAnalysis(Record):
    editor_reports: tuple[ResearchReport, ...]
    transfer_comparisons: tuple[TransferComparison, ...]
    controlled_style_conclusion_supported: bool
    markdown: str
