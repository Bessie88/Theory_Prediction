"""Typed stage payloads. Workers get these objects, never the coordinator registry."""
from dataclasses import dataclass
from .workflow_contracts import (
    Record, EditorKey, EditorRevision, ChronologicalSplitManifest, PredictionCase,
    OutcomeCase, SourceReference, ActionDefinition, TheoryRecord, FeatureSpecification,
    FeatureObservation, MatchPlan, ApplicabilityDecision, ActionPrediction, ObservedAction,
    HumanReviewRecord, EvaluationResult, MetadataItem, NativeRepresentation, Method, Period, require_same_cases,
)
from .workflow_config import (
    SourceSettings, SplitSettings, UnitSettings, DiscoverySettings, ExtractionSettings,
    AnnotationSettings, PolicySettings, EvaluationSettings,
)
from .policy_contracts import PolicySnapshot, PolicyRuleRegistry, BaselineComparability
from .action_comparison import FrozenActionProtocol, ActionProtocolSettings
from .annotation_contracts import AnnotationCases, AnnotationAudit
from .review_contracts import (
    ReviewSettings, ReviewAssignments, ReviewSamplingAudit, FeatureReviewSource,
    ReviewExport, HumanResponseBatch, ReviewImportAudit, FeatureHumanReviews,
)
from .evaluation_contracts import EvaluationContext, EvaluationDetails, BoundaryAnalysis, EditorAnalysis, ResearchReport
from .schemas import require, stable_id
from .application_contracts import (
    ApplicationSettings, FrozenAuthority, TheoryDeclaration, ActionCompatibility, ActionAlias,
    FrozenTheoryRegistry, TheoryFeatureObservations,
    TheoryApplicabilityTrace, PredictionProvenanceArtifact,
)
from .measurement_contracts import (
    ClauseSource, ClauseBinding, MeasurementReview, FeatureRegistrySnapshot,
    FrozenMeasurements, MeasurementRuntime, CaseConditionLabels,
)


class StageNotImplemented(NotImplementedError):
    """Intentional boundary: no placeholder ever synthesizes research results."""


@dataclass(frozen=True)
class SplitArtifact(Record):
    manifest: ChronologicalSplitManifest


@dataclass(frozen=True)
class EarlyRecords(Record):
    records: tuple[OutcomeCase, ...]
    revisions: tuple[EditorRevision, ...]

    def validate(self):
        require(all(r.identity.period is Period.EARLY for r in self.records), "Only EARLY induction")


@dataclass(frozen=True)
class EarlyBeforeRecords(Record):
    records: tuple[PredictionCase, ...]

    def validate(self):
        require(all(r.identity.period is Period.EARLY for r in self.records), "Only EARLY operationalization")


@dataclass(frozen=True)
class LaterRecords(Record):
    records: tuple[PredictionCase, ...]

    def validate(self):
        require(all(r.identity.period is Period.LATER for r in self.records), "Only LATER prediction")


@dataclass(frozen=True)
class OutcomeRecords(Record):
    records: tuple[OutcomeCase, ...]

    def validate(self):
        require(all(r.identity.period is Period.LATER for r in self.records), "Only LATER outcomes")


@dataclass(frozen=True)
class NeutralActionCatalog(Record):
    version: str
    actions: tuple[ActionDefinition, ...]


@dataclass(frozen=True)
class LegacyCaseAlias(Record):
    canonical_case_id: str
    native_case_id: str
    cohort_id: str


@dataclass(frozen=True)
class NativeTheoryBundle(Record):
    """Read-only references, never original theory mutations."""
    theory: SourceReference
    sidecars: tuple[SourceReference, ...]
    case_aliases: tuple[LegacyCaseAlias, ...]
    generation_mode: str
    rule_representation: str
    integration_method: str
    native_version: str

    def validate(self):
        require(self.generation_mode == "natural_language_predictive_theory" and
                self.rule_representation == "natural_language" and
                self.integration_method == "within_method_group_then_all_groups" and
                self.native_version == "method-group-then-global-prose-v1",
                "Unsupported native theory generation")


@dataclass(frozen=True)
class TheoryBundle(Record):
    editor: EditorKey
    theories: tuple[TheoryRecord, ...]
    native_source: SourceReference
    excluded_rule_ids: tuple[str, ...]
    diagnostics: tuple[str, ...]
    source_rules: tuple[NativeRepresentation, ...] = ()
    original_conditions: tuple[NativeRepresentation, ...] = ()
    source_actions: tuple[NativeRepresentation, ...] = ()
    case_aliases: tuple[LegacyCaseAlias, ...] = ()
    excluded_rules_json: str = "[]"
    knowledge_source: Method = Method.THEORY
    source_snapshot_id: str | None = None
    unresolved_source_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class MeasurementPlan(Record):
    specifications: tuple[FeatureSpecification, ...]
    match_plans: tuple[MatchPlan, ...]
    unresolved: tuple[str, ...]
    clauses: tuple[ClauseSource, ...] = ()
    bindings: tuple[ClauseBinding, ...] = ()
    reviews: tuple[MeasurementReview, ...] = ()
    feature_registry: FeatureRegistrySnapshot | None = None
    source_theory_hash: str = ""
    application_authorities: tuple[FrozenAuthority, ...] = ()
    application_declarations: tuple[TheoryDeclaration, ...] = ()
    action_compatibility: tuple[ActionCompatibility, ...] = ()
    action_aliases: tuple[ActionAlias, ...] = ()
    knowledge_source: Method = Method.THEORY

    def validate(self):
        require(all(p.method is self.knowledge_source for p in self.match_plans), "Plan source mismatch")


@dataclass(frozen=True)
class PolicyPlan(Record):
    policy_version: str
    specifications: tuple[FeatureSpecification, ...]
    match_plans: tuple[MatchPlan, ...]
    source_snapshot: SourceReference
    unresolved: tuple[str, ...]
    rules: PolicyRuleRegistry | None = None
    canonical_bundle: TheoryBundle | None = None
    measurement: MeasurementPlan | None = None

    def validate(self):
        require(all(p.method is Method.NPOV for p in self.match_plans), "NPOV plans only")


@dataclass(frozen=True)
class StudySpecification(Record):
    """Prediction-safe projection: no source paths, source cases or raw theory ledger."""
    theory_features: tuple[FeatureSpecification, ...]
    policy_features: tuple[FeatureSpecification, ...]
    theory_plans: tuple[MatchPlan, ...]
    policy_plans: tuple[MatchPlan, ...]
    catalog: NeutralActionCatalog
    freeze_hash: str
    conflict_policy_version: str
    measurement_freeze: FrozenMeasurements
    theory_registry: FrozenTheoryRegistry
    policy_snapshot_hash: str
    policy_registry: FrozenTheoryRegistry | None = None
    policy_measurement_freeze: FrozenMeasurements | None = None
    policy_rule_registry_hash: str = ""
    require_shared_action_ids: bool = False
    action_protocol_hash: str = ""
    evaluation_protocol_hash: str = ""


@dataclass(frozen=True)
class FeatureBatch(Record):
    cases: tuple[PredictionCase, ...]
    observations: tuple[FeatureObservation, ...]
    study_spec_hash: str

    def validate(self):
        case_map = {c.identity.case_id: c.identity for c in self.cases}
        require(len(case_map) == len(self.cases), "Duplicate feature cases")
        require(all(o.identity == case_map.get(o.identity.case_id) for o in self.observations),
                "Feature observation belongs to another case")
        keys = [(o.identity.case_id, o.feature_id) for o in self.observations]
        require(len(keys) == len(set(keys)), "Duplicate feature observation")


@dataclass(frozen=True)
class ApplicabilityBatch(Record):
    decisions: tuple[ApplicabilityDecision, ...]
    traces: tuple[TheoryApplicabilityTrace, ...] = ()
    study_spec_hash: str = ""
    feature_input_artifact_hash: str = ""
    knowledge_source: Method = Method.THEORY
    variant: str = "WITH_BOUNDARIES"


@dataclass(frozen=True)
class PredictionBatch(Record):
    method: Method
    predictions: tuple[ActionPrediction, ...]
    study_spec_hash: str
    action_protocol_hash: str = ""
    variant: str = "WITH_BOUNDARIES"

    def validate(self):
        require(all(p.method is self.method and p.study_spec_hash == self.study_spec_hash
                    for p in self.predictions), "Prediction branch/freeze mismatch")
        ids = [p.identity.case_id for p in self.predictions]
        require(len(ids) == len(set(ids)), "Duplicate prediction case")


@dataclass(frozen=True)
class AnnotationBatch(Record):
    annotations: tuple[ObservedAction, ...]

    def validate(self):
        require(len({a.identity.case_id for a in self.annotations}) == len(self.annotations), "Duplicate annotation case")
        require(all(a.annotation_id and a.input_hash and a.protocol_hash and a.explanation and
                    a.annotator_model and a.prompt_version for a in self.annotations),
                "Registered annotations require complete independent annotation provenance")


@dataclass(frozen=True)
class HumanReviewBatch(Record):
    reviews: tuple[HumanReviewRecord, ...]


@dataclass(frozen=True)
class EvaluationBatch(Record):
    results: tuple[EvaluationResult, ...]


@dataclass(frozen=True)
class DisagreementReport(Record):
    editor: EditorKey
    case_ids: tuple[str, ...]
    statistics: tuple[MetadataItem, ...]
    analysis_version: str


@dataclass(frozen=True)
class DataPreparationInput(Record):
    editor: EditorKey
    source: SourceSettings
    split: SplitSettings
    unit: UnitSettings
    supplied_vocabulary: tuple[ActionDefinition, ...]


@dataclass(frozen=True)
class DataPreparationOutput(Record):
    context: EvaluationContext
    split: SplitArtifact
    early: EarlyRecords
    early_before: EarlyBeforeRecords
    later: LaterRecords
    outcomes: OutcomeRecords
    catalog: NeutralActionCatalog

    def validate(self):
        manifest = self.split.manifest
        require_same_cases(self.early.records, self.early_before.records)
        require_same_cases(self.later.records, self.outcomes.records)
        identities = {c.case_id: c for c in manifest.cases}
        all_records = self.early.records + self.outcomes.records
        require({r.identity.case_id: r.identity for r in all_records} == identities,
                "Prepared case sets differ from split manifest")
        require(all(r.identity.editor == manifest.editor for r in all_records), "Mixed editor")


@dataclass(frozen=True)
class DiscoveryWorkspace(Record):
    root: str
    experiment_id: str
    configuration_id: str
    execution_authorized: bool
    models_authorized: bool


@dataclass(frozen=True)
class DiscoveryInput(Record):
    early: EarlyRecords
    split: SplitArtifact
    settings: DiscoverySettings
    workspace: DiscoveryWorkspace


@dataclass(frozen=True)
class DiscoveryOutput(Record):
    native: NativeTheoryBundle


@dataclass(frozen=True)
class TheoryAdapterInput(Record):
    native: NativeTheoryBundle
    split: SplitArtifact


@dataclass(frozen=True)
class TheoryAdapterOutput(Record):
    theories: TheoryBundle


@dataclass(frozen=True)
class MeasurementPlanningInput(Record):
    theories: TheoryBundle
    early_before: EarlyBeforeRecords
    catalog: NeutralActionCatalog
    settings: ExtractionSettings
    runtime: MeasurementRuntime
    application: ApplicationSettings = ApplicationSettings()


@dataclass(frozen=True)
class MeasurementPlanningOutput(Record):
    measurement: MeasurementPlan
    feature_specifications: FeatureRegistrySnapshot


@dataclass(frozen=True)
class PolicySnapshotInput(Record):
    policy: PolicySettings
    execution_authorized: bool = False


@dataclass(frozen=True)
class PolicySnapshotOutput(Record):
    policy_snapshot: PolicySnapshot


@dataclass(frozen=True)
class PolicyAdapterInput(Record):
    policy_snapshot: PolicySnapshot
    catalog: NeutralActionCatalog
    policy: PolicySettings
    runtime: MeasurementRuntime


@dataclass(frozen=True)
class PolicyAdapterOutput(Record):
    policy_rules: PolicyRuleRegistry


@dataclass(frozen=True)
class NPOVPreparationInput(Record):
    early_before: EarlyBeforeRecords
    catalog: NeutralActionCatalog
    policy: PolicySettings
    settings: ExtractionSettings
    policy_rules: PolicyRuleRegistry
    runtime: MeasurementRuntime


@dataclass(frozen=True)
class NPOVPreparationOutput(Record):
    policy_plan: PolicyPlan


@dataclass(frozen=True)
class FreezeInput(Record):
    evaluation_settings: EvaluationSettings
    action_protocol: FrozenActionProtocol
    theories: TheoryBundle
    application: ApplicationSettings
    feature_specifications: FeatureRegistrySnapshot
    measurement: MeasurementPlan
    policy_plan: PolicyPlan
    catalog: NeutralActionCatalog


@dataclass(frozen=True)
class FreezeOutput(Record):
    study: StudySpecification


@dataclass(frozen=True)
class FeatureExtractionInput(Record):
    later: LaterRecords
    study: StudySpecification
    settings: ExtractionSettings
    runtime: MeasurementRuntime


@dataclass(frozen=True)
class FeatureExtractionOutput(Record):
    features: FeatureBatch
    feature_review_source: FeatureReviewSource
    condition_labels: tuple[CaseConditionLabels, ...] = ()


@dataclass(frozen=True)
class ApplicabilityInput(Record):
    features: FeatureBatch
    study: StudySpecification
    execution_authorized: bool = False


@dataclass(frozen=True)
class ApplicabilityOutput(Record):
    theory_applicability: ApplicabilityBatch
    theory_feature_observations: TheoryFeatureObservations
    npov_applicability: ApplicabilityBatch
    npov_feature_observations: TheoryFeatureObservations
    baseline_comparability: BaselineComparability

    @property
    def applicability(self):
        return self.theory_applicability


@dataclass(frozen=True)
class TheoryPredictionInput(Record):
    features: FeatureBatch
    applicability: ApplicabilityBatch
    study: StudySpecification
    execution_authorized: bool = False


@dataclass(frozen=True)
class TheoryPredictionOutput(Record):
    theory_action_predictions: PredictionBatch
    prediction_provenance: PredictionProvenanceArtifact

    @property
    def theory_predictions(self):
        return self.theory_action_predictions

    def validate(self):
        require(self.theory_action_predictions.method is Method.THEORY, "Incorrect prediction method")
        require(self.theory_action_predictions.study_spec_hash == self.prediction_provenance.study_spec_hash,
                "Prediction/provenance freeze mismatch")
        require_same_cases(self.theory_action_predictions.predictions, self.prediction_provenance.records)
        index = {p.identity.case_id: p for p in self.prediction_provenance.records}
        require(all(p.provenance_id == index[p.identity.case_id].provenance_id
                    for p in self.theory_action_predictions.predictions),
                "Prediction/provenance link mismatch")


@dataclass(frozen=True)
class NPOVPredictionInput(Record):
    features: FeatureBatch
    applicability: ApplicabilityBatch
    study: StudySpecification
    execution_authorized: bool = False


@dataclass(frozen=True)
class NPOVPredictionOutput(Record):
    npov_predictions: PredictionBatch
    npov_prediction_provenance: PredictionProvenanceArtifact

    def validate(self):
        require(self.npov_predictions.method is Method.NPOV and
                self.npov_prediction_provenance.knowledge_source is Method.NPOV, "Incorrect prediction method")
        require(self.npov_predictions.study_spec_hash == self.npov_prediction_provenance.study_spec_hash,
                "Policy prediction/provenance freeze mismatch")
        require_same_cases(self.npov_predictions.predictions, self.npov_prediction_provenance.records)
        index = {r.identity.case_id: r for r in self.npov_prediction_provenance.records}
        require(all(p.provenance_id == index[p.identity.case_id].provenance_id
                    for p in self.npov_predictions.predictions), "Policy provenance link mismatch")


@dataclass(frozen=True)
class ActionProtocolInput(Record):
    catalog: NeutralActionCatalog
    settings: ActionProtocolSettings
    execution_authorized: bool = False


@dataclass(frozen=True)
class ActionProtocolOutput(Record):
    action_protocol: FrozenActionProtocol


@dataclass(frozen=True)
class AnnotationDataInput(Record):
    outcomes: OutcomeRecords
    later: LaterRecords
    settings: AnnotationSettings
    execution_authorized: bool = False


@dataclass(frozen=True)
class AnnotationDataOutput(Record):
    annotation_cases: AnnotationCases


@dataclass(frozen=True)
class AnnotationInput(Record):
    """Only independent actual edits, neutral frozen guidelines and model settings."""
    annotation_cases: AnnotationCases
    action_protocol: FrozenActionProtocol
    settings: AnnotationSettings
    runtime: MeasurementRuntime
    extractor_families: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnnotationOutput(Record):
    annotations: AnnotationBatch
    annotation_audit: AnnotationAudit


@dataclass(frozen=True)
class ReviewSamplingInput(Record):
    theory_predictions: PredictionBatch
    npov_predictions: PredictionBatch
    annotations: AnnotationBatch
    feature_review_source: FeatureReviewSource
    action_protocol: FrozenActionProtocol
    settings: ReviewSettings
    execution_authorized: bool = False


@dataclass(frozen=True)
class ReviewSamplingOutput(Record):
    action_assignments: ReviewAssignments
    feature_assignments: ReviewAssignments
    sampling_audit: ReviewSamplingAudit


@dataclass(frozen=True)
class ActionReviewExportInput(Record):
    annotation_cases: AnnotationCases
    annotations: AnnotationBatch
    action_assignments: ReviewAssignments
    action_protocol: FrozenActionProtocol
    execution_authorized: bool = False


@dataclass(frozen=True)
class ActionReviewExportOutput(Record):
    action_review_export: ReviewExport


@dataclass(frozen=True)
class FeatureReviewExportInput(Record):
    feature_review_source: FeatureReviewSource
    feature_assignments: ReviewAssignments
    action_protocol: FrozenActionProtocol
    execution_authorized: bool = False


@dataclass(frozen=True)
class FeatureReviewExportOutput(Record):
    feature_review_export: ReviewExport


@dataclass(frozen=True)
class HumanReviewInput(Record):
    annotation_cases: AnnotationCases
    annotations: AnnotationBatch
    action_review_export: ReviewExport
    action_protocol: FrozenActionProtocol
    settings: ReviewSettings
    execution_authorized: bool = False
    responses: HumanResponseBatch | None = None


@dataclass(frozen=True)
class HumanReviewOutput(Record):
    reviews: HumanReviewBatch
    action_review_audit: ReviewImportAudit


@dataclass(frozen=True)
class FeatureReviewImportInput(Record):
    feature_review_source: FeatureReviewSource
    feature_review_export: ReviewExport
    settings: ReviewSettings
    execution_authorized: bool = False
    responses: HumanResponseBatch | None = None


@dataclass(frozen=True)
class FeatureReviewImportOutput(Record):
    feature_reviews: FeatureHumanReviews
    feature_review_audit: ReviewImportAudit


@dataclass(frozen=True)
class EvaluationInput(Record):
    context: EvaluationContext
    study: StudySpecification
    action_protocol: FrozenActionProtocol
    feature_reviews: FeatureHumanReviews
    theory_predictions: PredictionBatch
    npov_predictions: PredictionBatch
    annotations: AnnotationBatch
    reviews: HumanReviewBatch
    later: LaterRecords
    settings: EvaluationSettings
    prediction_provenance: PredictionProvenanceArtifact
    npov_prediction_provenance: PredictionProvenanceArtifact
    execution_authorized: bool = False

    def validate(self):
        from .action_comparison import validate_protocol
        validate_protocol(self.action_protocol)
        require(self.study.evaluation_protocol_hash == stable_id("evaluation_protocol", self.settings.to_dict()),
                "Evaluation policy differs from frozen study")
        require(self.theory_predictions.action_protocol_hash == self.npov_predictions.action_protocol_hash ==
                self.action_protocol.protocol_hash, "Prediction comparison protocol differs from annotation")
        require(all(a.protocol_hash == self.action_protocol.protocol_hash for a in self.annotations.annotations),
                "Evaluation annotation/label protocol mismatch")
        require(self.theory_predictions.method is Method.THEORY and
                self.npov_predictions.method is Method.NPOV, "Incorrect evaluation branch")
        require(self.prediction_provenance.knowledge_source is Method.THEORY and
                self.npov_prediction_provenance.knowledge_source is Method.NPOV,
                "Evaluation provenance sources differ")
        require(self.theory_predictions.study_spec_hash == self.npov_predictions.study_spec_hash ==
                self.prediction_provenance.study_spec_hash == self.npov_prediction_provenance.study_spec_hash,
                "Branches use different study freezes")
        require(self.study.freeze_hash == self.theory_predictions.study_spec_hash,
                "Evaluation study/prediction mismatch")
        for batch, provenance in ((self.theory_predictions, self.prediction_provenance),
                                  (self.npov_predictions, self.npov_prediction_provenance)):
            records = {r.identity.case_id: r for r in provenance.records}
            source = self.study.theory_registry if batch.method is Method.THEORY else self.study.policy_registry
            require(source is not None and all(p.provenance_id == records[p.identity.case_id].provenance_id and
                    records[p.identity.case_id].source_registry_hash == source.registry_hash for p in batch.predictions),
                    "Prediction provenance/knowledge registry mismatch")
        raw = {a.identity.case_id: a for a in self.annotations.annotations}
        require(all(r.model_annotation == raw.get(r.identity.case_id) for r in self.reviews.reviews),
                "Human review cannot overwrite raw model annotations")
        eligible = {c.identity.case_id: c.identity for c in self.later.records}
        require(all(r.identity == eligible.get(r.identity.case_id) for r in self.feature_reviews.records) and
                len({r.subject_id for r in self.feature_reviews.records}) == len(self.feature_reviews.records),
                "Feature review identities differ from evaluation manifest")
        require_same_cases(self.theory_predictions.predictions, self.npov_predictions.predictions,
                           self.annotations.annotations, self.reviews.reviews, self.later.records,
                           self.prediction_provenance.records, self.npov_prediction_provenance.records)


@dataclass(frozen=True)
class EvaluationOutput(Record):
    evaluation: EvaluationBatch
    evaluation_details: EvaluationDetails


@dataclass(frozen=True)
class DisagreementInput(Record):
    evaluation_details: EvaluationDetails
    annotation_cases: AnnotationCases
    context: EvaluationContext
    theories: TheoryBundle
    policy_rules: PolicyRuleRegistry
    action_protocol: FrozenActionProtocol
    feature_reviews: FeatureHumanReviews
    npov_prediction_provenance: PredictionProvenanceArtifact
    prediction_provenance: PredictionProvenanceArtifact
    evaluation: EvaluationBatch
    theory_predictions: PredictionBatch
    npov_predictions: PredictionBatch
    annotations: AnnotationBatch
    reviews: HumanReviewBatch
    later: LaterRecords
    settings: EvaluationSettings


@dataclass(frozen=True)
class DisagreementOutput(Record):
    disagreement: DisagreementReport
    editor_analysis: EditorAnalysis


@dataclass(frozen=True)
class BoundaryAnalysisInput(Record):
    features: FeatureBatch
    study: StudySpecification
    theory_predictions: PredictionBatch
    evaluation_details: EvaluationDetails
    action_protocol: FrozenActionProtocol
    execution_authorized: bool = False

@dataclass(frozen=True)
class BoundaryAnalysisOutput(Record):
    boundary_analysis: BoundaryAnalysis

@dataclass(frozen=True)
class FinalReportInput(Record):
    context: EvaluationContext
    theories: TheoryBundle
    features: FeatureBatch
    study: StudySpecification
    evaluation_details: EvaluationDetails
    editor_analysis: EditorAnalysis
    boundary_analysis: BoundaryAnalysis
    execution_authorized: bool = False

@dataclass(frozen=True)
class FinalReportOutput(Record):
    research_report: ResearchReport
