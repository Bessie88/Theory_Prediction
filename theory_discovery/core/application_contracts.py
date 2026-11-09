"""Frozen, source-backed theory application contracts; no I/O or model work."""
from dataclasses import dataclass
from .workflow_contracts import (
    Record, EditorKey, CaseIdentity, ActionDefinition, ActionInstance, FeatureObservation, Method,
    FeatureSpecification, MeasurementValue, EvidenceSpan, MetadataItem, ApplicabilityDecision,
)
from .schemas import require, stable_id


@dataclass(frozen=True)
class ApplicationSettings(Record):
    version: str = "theory-application-v1"
    authorities_json: tuple[str, ...] = ()
    declarations_json: tuple[str, ...] = ()
    compatibility_json: tuple[str, ...] = ()
    action_aliases_json: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrozenAuthority(Record):
    authority_id: str
    kind: str
    version: str
    provenance_sha256: str
    source_theory_sha256: str | None
    source_pointers: tuple[str, ...]
    rationale: str
    established_before_later: bool

    def validate(self):
        require(self.kind in ("SOURCE_THEORY", "SOURCE_POLICY", "INDEPENDENT_PROTOCOL"), "Unknown application authority")
        require(self.established_before_later and bool(self.version) and bool(self.rationale.strip()),
                "Application protocol must be established before Later use")
        require(len(self.provenance_sha256) == 64 and
                all(c in "0123456789abcdef" for c in self.provenance_sha256), "Authority hash required")


@dataclass(frozen=True)
class ClauseOmission(Record):
    clause_id: str
    reason: str
    authority_id: str


@dataclass(frozen=True)
class ActionDirective(Record):
    directive_id: str
    source_action_id: str
    source_rule_ids: tuple[str, ...]
    when_expression_json: str
    target_mode: str
    target_feature_id: str | None
    authority_id: str

    def validate(self):
        require(self.target_mode in ("SOURCE_LEVEL", "ALL_EVIDENCE_SPANS"), "Unknown target localization")
        require((self.target_mode == "ALL_EVIDENCE_SPANS") == (self.target_feature_id is not None),
                "Span localization needs an explicit Before feature")
        require(bool(self.source_rule_ids), "Action directive needs original rule sources")


@dataclass(frozen=True)
class TheoryDeclaration(Record):
    theory_id: str
    applicability_expression_json: str
    directives: tuple[ActionDirective, ...]
    composition: str
    omitted_clauses: tuple[ClauseOmission, ...]
    authority_id: str
    fidelity_rationale: str

    def validate(self):
        require(self.composition in ("SINGLE", "ALL_SUPPORTED", "EXCLUSIVE_ALTERNATIVES", "ALLOWABLE_ALTERNATIVES"),
                "Action composition must be explicitly source-backed")
        require(bool(self.fidelity_rationale.strip()), "Source fidelity rationale required")


@dataclass(frozen=True)
class ActionCompatibility(Record):
    left_action_id: str
    right_action_id: str
    relation: str
    authority_id: str
    rationale: str

    def validate(self):
        require(self.left_action_id != self.right_action_id and
                self.relation in ("COMPATIBLE", "INCOMPATIBLE"), "Invalid compatibility relation")
        require(bool(self.rationale.strip()), "Compatibility needs a frozen justification")


@dataclass(frozen=True)
class ActionAlias(Record):
    source_action_id: str
    catalog_action_id: str
    authority_id: str
    rationale: str


@dataclass(frozen=True)
class FrozenSourceAction(Record):
    source_action_id: str
    original_json: str
    semantic_signature: str
    definition: ActionDefinition | None
    mapping_status: str
    source_pointer: str

    def validate(self):
        require(self.mapping_status in ("NATIVE", "CATALOG_MAPPED", "UNMAPPED", "OTHER"), "Invalid action mapping status")


@dataclass(frozen=True)
class FrozenTheoryApplication(Record):
    theory_id: str
    native_branch_id: str
    source_rule_ids: tuple[str, ...]
    clause_ids: tuple[str, ...]
    applicability_expression_json: str
    directives: tuple[ActionDirective, ...]
    composition: str
    source_action_ids: tuple[str, ...]
    omitted_clauses: tuple[ClauseOmission, ...]
    authority_id: str | None
    logic_resolution: str
    limitations: tuple[str, ...]

    def validate(self):
        require(self.logic_resolution in ("DECLARED", "UNRESOLVED"), "Unknown frozen logic state")
        require(self.composition in ("SINGLE", "ALL_SUPPORTED", "EXCLUSIVE_ALTERNATIVES", "ALLOWABLE_ALTERNATIVES", "UNRESOLVED"),
                "Unknown frozen action composition")


@dataclass(frozen=True)
class FrozenTheoryRegistry(Record):
    editor: EditorKey
    source_theory_hash: str
    theories: tuple[FrozenTheoryApplication, ...]
    source_actions: tuple[FrozenSourceAction, ...]
    authorities: tuple[FrozenAuthority, ...]
    compatibility: tuple[ActionCompatibility, ...]
    version: str
    registry_hash: str
    action_aliases: tuple[ActionAlias, ...] = ()
    knowledge_source: Method = Method.THEORY
    source_snapshot_id: str | None = None
    unresolved_source_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureWitness(Record):
    feature_id: str
    observation_id: str | None
    value: MeasurementValue
    observation_status: str
    evidence: tuple[EvidenceSpan, ...]
    metadata_names: tuple[str, ...]
    valid: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ClauseEvaluation(Record):
    clause_id: str
    logical_role: str
    measured_truth: bool | None
    satisfied_truth: bool | None
    resolution: str
    features: tuple[FeatureWitness, ...]


@dataclass(frozen=True)
class LogicNodeTrace(Record):
    path: str
    operator: str
    truth: bool | None
    clause_id: str | None


@dataclass(frozen=True)
class ActionGateTrace(Record):
    directive_id: str
    source_action_id: str
    truth: bool | None
    nodes: tuple[LogicNodeTrace, ...]


@dataclass(frozen=True)
class TheoryApplicabilityTrace(Record):
    identity: CaseIdentity
    decision_id: str
    theory_id: str
    clauses: tuple[ClauseEvaluation, ...]
    logic_nodes: tuple[LogicNodeTrace, ...]
    action_gates: tuple[ActionGateTrace, ...]
    missing_feature_ids: tuple[str, ...]
    exclusion_reasons: tuple[str, ...]
    reason_codes: tuple[str, ...]
    knowledge_source: Method = Method.THEORY


@dataclass(frozen=True)
class TheoryFeatureObservations(Record):
    identities: tuple[CaseIdentity, ...]
    observations: tuple[FeatureObservation, ...]
    study_spec_hash: str
    feature_input_artifact_hash: str
    knowledge_source: Method = Method.THEORY


@dataclass(frozen=True)
class PredictedActionTrace(Record):
    action_trace_id: str
    action: ActionInstance
    source_action_ids: tuple[str, ...]
    theory_ids: tuple[str, ...]
    source_rule_ids: tuple[str, ...]
    directive_ids: tuple[str, ...]
    target_observation_ids: tuple[str, ...]
    target_spans: tuple[EvidenceSpan, ...]
    owner_ids: tuple[str, ...] = ()
    knowledge_source: Method = Method.THEORY


@dataclass(frozen=True)
class ActionConflict(Record):
    left_action_id: str
    right_action_id: str
    relation: str
    authority_id: str | None
    reason: str


@dataclass(frozen=True)
class PredictionProvenance(Record):
    provenance_id: str
    identity: CaseIdentity
    study_spec_hash: str
    theory_registry_hash: str
    feature_input_artifact_hash: str
    applicability_decision_ids: tuple[str, ...]
    applicable_theory_ids: tuple[str, ...]
    not_applicable_theory_ids: tuple[str, ...]
    unknown_theory_ids: tuple[str, ...]
    condition_features: tuple[FeatureWitness, ...]
    boundary_features: tuple[FeatureWitness, ...]
    clause_evaluations: tuple[ClauseEvaluation, ...]
    exclusions: tuple[str, ...]
    candidate_actions: tuple[PredictedActionTrace, ...]
    conflicts: tuple[ActionConflict, ...]
    reason_codes: tuple[str, ...]
    knowledge_source: Method = Method.THEORY
    source_registry_hash: str = ""
    matched_policy_rule_ids: tuple[str, ...] = ()
    excluded_policy_rule_ids: tuple[str, ...] = ()
    unknown_policy_rule_ids: tuple[str, ...] = ()
    policy_snapshot_id: str | None = None


@dataclass(frozen=True)
class PredictionProvenanceArtifact(Record):
    records: tuple[PredictionProvenance, ...]
    study_spec_hash: str
    knowledge_source: Method = Method.THEORY


def theory_registry_hash(registry):
    return stable_id("frozen_theory_registry", {
        key: value for key, value in registry.to_dict().items() if key != "registry_hash"})


def study_hash(study):
    return stable_id("study_freeze", {key: value for key, value in study.to_dict().items() if key != "freeze_hash"})
