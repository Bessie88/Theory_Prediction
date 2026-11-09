"""One declarative dependency graph, with typed port bindings (no stage execution)."""
from dataclasses import dataclass
from . import stage_interfaces as io


@dataclass(frozen=True)
class Binding:
    input_field: str
    producer: str
    output_field: str


@dataclass(frozen=True)
class StageSpec:
    name: str
    version: str
    module: str
    input_type: type
    output_type: type
    bindings: tuple[Binding, ...]
    # Output field -> access class. Modules receive payloads, not registry metadata.
    access: tuple[tuple[str, str], ...]

    @property
    def dependencies(self):
        return tuple(dict.fromkeys(b.producer for b in self.bindings))


def b(field, producer, output=None):
    return Binding(field, producer, output or field)


STAGES = (
    StageSpec("data_preparation", "8", "data_preparation",
              io.DataPreparationInput, io.DataPreparationOutput, (),
              (("context", "EVALUATION_ONLY"), ("split", "COORDINATION"), ("early", "EARLY_PRIVATE"),
               ("early_before", "EARLY_PRIVATE"), ("later", "PREDICTION_SAFE"),
               ("outcomes", "EVALUATION_ONLY"), ("catalog", "NEUTRAL_VOCABULARY"))),
    StageSpec("action_protocol", "6", "action_protocol",
              io.ActionProtocolInput, io.ActionProtocolOutput,
              (b("catalog", "data_preparation"),), (("action_protocol", "NEUTRAL_VOCABULARY"),)),
    StageSpec("annotation_data_preparation", "6", "annotation_data",
              io.AnnotationDataInput, io.AnnotationDataOutput,
              (b("outcomes", "data_preparation"), b("later", "data_preparation")),
              (("annotation_cases", "EVALUATION_ONLY"),)),
    StageSpec("existing_theory_discovery", "3", "discovery_bridge",
              io.DiscoveryInput, io.DiscoveryOutput,
              (b("early", "data_preparation"), b("split", "data_preparation")),
              (("native", "EARLY_PRIVATE"),)),
    StageSpec("theory_adapter", "2", "theory_adapter",
              io.TheoryAdapterInput, io.TheoryAdapterOutput,
              (b("native", "existing_theory_discovery"), b("split", "data_preparation")),
              (("theories", "EARLY_PRIVATE"),)),
    StageSpec("measurement_planning", "13", "measurement_planner",
              io.MeasurementPlanningInput, io.MeasurementPlanningOutput,
              (b("theories", "theory_adapter"), b("early_before", "data_preparation"),
               b("catalog", "data_preparation")), (("measurement", "EARLY_PRIVATE"), ("feature_specifications", "EARLY_PRIVATE"))),
    StageSpec("policy_snapshot", "5", "policy_snapshot",
              io.PolicySnapshotInput, io.PolicySnapshotOutput, (),
              (("policy_snapshot", "COORDINATION"),)),
    StageSpec("npov_adapter", "10", "policy_adapter",
              io.PolicyAdapterInput, io.PolicyAdapterOutput,
              (b("policy_snapshot", "policy_snapshot"), b("catalog", "data_preparation")),
              (("policy_rules", "EARLY_PRIVATE"),)),
    StageSpec("npov_preparation", "18", "npov",
              io.NPOVPreparationInput, io.NPOVPreparationOutput,
              (b("early_before", "data_preparation"), b("catalog", "data_preparation"),
               b("policy_rules", "npov_adapter")),
              (("policy_plan", "EARLY_PRIVATE"),)),
    StageSpec("freeze_study", "7", "study_freeze",
              io.FreezeInput, io.FreezeOutput,
              (b("action_protocol", "action_protocol"), b("theories", "theory_adapter"),
               b("feature_specifications", "measurement_planning"),
               b("measurement", "measurement_planning"), b("policy_plan", "npov_preparation"),
               b("catalog", "data_preparation")), (("study", "PREDICTION_SAFE"),)),
    StageSpec("feature_extraction", "11", "feature_extractor",
              io.FeatureExtractionInput, io.FeatureExtractionOutput,
              (b("later", "data_preparation"), b("study", "freeze_study")),
              (("features", "PREDICTION_SAFE"), ("feature_review_source", "EVALUATION_ONLY"),
               ("condition_labels", "EVALUATION_ONLY"))),
    StageSpec("applicability", "5", "applicability",
              io.ApplicabilityInput, io.ApplicabilityOutput,
              (b("features", "feature_extraction"), b("study", "freeze_study")),
              (("theory_applicability", "PREDICTION_SAFE"), ("theory_feature_observations", "PREDICTION_SAFE"),
               ("npov_applicability", "PREDICTION_SAFE"), ("npov_feature_observations", "PREDICTION_SAFE"),
               ("baseline_comparability", "PREDICTION_SAFE"))),
    StageSpec("theory_prediction", "5", "theory_predictor",
              io.TheoryPredictionInput, io.TheoryPredictionOutput,
              (b("features", "feature_extraction"), b("applicability", "applicability", "theory_applicability"),
               b("study", "freeze_study")), (("theory_action_predictions", "PREDICTION_RESULT"), ("prediction_provenance", "PREDICTION_RESULT"))),
    StageSpec("npov_prediction", "5", "npov",
              io.NPOVPredictionInput, io.NPOVPredictionOutput,
              (b("features", "feature_extraction"), b("applicability", "applicability", "npov_applicability"),
               b("study", "freeze_study")), (("npov_predictions", "PREDICTION_RESULT"), ("npov_prediction_provenance", "PREDICTION_RESULT"))),
    StageSpec("independent_action_annotation", "9", "action_annotator",
              io.AnnotationInput, io.AnnotationOutput,
              (b("annotation_cases", "annotation_data_preparation"), b("action_protocol", "action_protocol")),
              (("annotations", "EVALUATION_ONLY"), ("annotation_audit", "EVALUATION_ONLY"))),
    StageSpec("review_sampling", "6", "review_sampling",
              io.ReviewSamplingInput, io.ReviewSamplingOutput,
              (b("theory_predictions", "theory_prediction", "theory_action_predictions"),
               b("npov_predictions", "npov_prediction"), b("annotations", "independent_action_annotation"),
               b("feature_review_source", "feature_extraction"), b("action_protocol", "action_protocol")),
              (("action_assignments", "COORDINATION"), ("feature_assignments", "COORDINATION"),
               ("sampling_audit", "EVALUATION_ONLY"))),
    StageSpec("action_review_export", "6", "action_review_export",
              io.ActionReviewExportInput, io.ActionReviewExportOutput,
              (b("annotation_cases", "annotation_data_preparation"),
               b("annotations", "independent_action_annotation"), b("action_assignments", "review_sampling"),
               b("action_protocol", "action_protocol")), (("action_review_export", "REVIEWER_PACKET"),)),
    StageSpec("feature_review_export", "6", "feature_review_export",
              io.FeatureReviewExportInput, io.FeatureReviewExportOutput,
              (b("feature_review_source", "feature_extraction"), b("feature_assignments", "review_sampling"),
               b("action_protocol", "action_protocol")), (("feature_review_export", "REVIEWER_PACKET"),)),
    StageSpec("human_verification", "6", "human_review",
              io.HumanReviewInput, io.HumanReviewOutput,
              (b("annotation_cases", "annotation_data_preparation"),
               b("annotations", "independent_action_annotation"), b("action_review_export", "action_review_export"),
               b("action_protocol", "action_protocol")),
              (("reviews", "EVALUATION_ONLY"), ("action_review_audit", "EVALUATION_ONLY"))),
    StageSpec("feature_review_import", "6", "feature_review_import",
              io.FeatureReviewImportInput, io.FeatureReviewImportOutput,
              (b("feature_review_source", "feature_extraction"), b("feature_review_export", "feature_review_export")),
              (("feature_reviews", "EVALUATION_ONLY"), ("feature_review_audit", "EVALUATION_ONLY"))),
    StageSpec("evaluation", "8", "evaluation",
              io.EvaluationInput, io.EvaluationOutput,
              (b("context", "data_preparation"), b("study", "freeze_study"), b("action_protocol", "action_protocol"), b("feature_reviews", "feature_review_import"),
               b("theory_predictions", "theory_prediction", "theory_action_predictions"),
               b("prediction_provenance", "theory_prediction"),
               b("npov_predictions", "npov_prediction"),
               b("npov_prediction_provenance", "npov_prediction"),
               b("annotations", "independent_action_annotation"),
               b("reviews", "human_verification"), b("later", "data_preparation")),
              (("evaluation", "EVALUATION_ONLY"), ("evaluation_details", "EVALUATION_ONLY"))),
    StageSpec("disagreement_analysis", "7", "disagreement",
              io.DisagreementInput, io.DisagreementOutput,
              (b("evaluation_details", "evaluation"), b("annotation_cases", "annotation_data_preparation"),
               b("context", "data_preparation"), b("theories", "theory_adapter"), b("policy_rules", "npov_adapter"),
               b("action_protocol", "action_protocol"), b("feature_reviews", "feature_review_import"),
               b("evaluation", "evaluation"), b("theory_predictions", "theory_prediction", "theory_action_predictions"),
               b("prediction_provenance", "theory_prediction"),
               b("npov_predictions", "npov_prediction"),
               b("npov_prediction_provenance", "npov_prediction"),
               b("annotations", "independent_action_annotation"),
               b("reviews", "human_verification"), b("later", "data_preparation")),
              (("disagreement", "EVALUATION_ONLY"), ("editor_analysis", "EVALUATION_ONLY"))),
    StageSpec("boundary_analysis", "7", "boundary_analysis", io.BoundaryAnalysisInput, io.BoundaryAnalysisOutput,
              (b("features", "feature_extraction"), b("study", "freeze_study"),
               b("theory_predictions", "theory_prediction", "theory_action_predictions"),
               b("evaluation_details", "evaluation"), b("action_protocol", "action_protocol")),
              (("boundary_analysis", "EVALUATION_ONLY"),)),
    StageSpec("final_report", "7", "final_report", io.FinalReportInput, io.FinalReportOutput,
              (b("context", "data_preparation"), b("theories", "theory_adapter"),
               b("features", "feature_extraction"), b("study", "freeze_study"),
               b("evaluation_details", "evaluation"), b("editor_analysis", "disagreement_analysis"),
               b("boundary_analysis", "boundary_analysis")), (("research_report", "EVALUATION_ONLY"),)),
)
# Optional diagnostic is addressable by the same coordinator/registry but never
# included in the default primary pipeline or evaluation dependency chain.
from .direct_policy_control import direct_policy_stage_spec
OPTIONAL_STAGES = (direct_policy_stage_spec(),)
BY_NAME = {s.name: s for s in STAGES + OPTIONAL_STAGES}


def dependency_order(selected=None, include_dependencies=True):
    """Return a plan only; never import or run handlers."""
    requested = tuple(s.name for s in STAGES) if selected is None else (selected,)
    result, active, done = [], set(), set()

    def visit(name):
        if name not in BY_NAME:
            raise ValueError("Unknown stage: " + name)
        if name in active:
            raise ValueError("Stage dependency cycle")
        if name in done:
            return
        active.add(name)
        if include_dependencies:
            for upstream in BY_NAME[name].dependencies:
                visit(upstream)
        active.remove(name)
        done.add(name)
        result.append(BY_NAME[name])
    for name in requested:
        visit(name)
    return tuple(result)


def downstream_stages(name):
    if name not in BY_NAME:
        raise ValueError("Unknown stage")
    found = {name}
    changed = True
    while changed:
        changed = False
        for spec in STAGES + OPTIONAL_STAGES:
            if spec.name not in found and set(spec.dependencies) & found:
                found.add(spec.name)
                changed = True
    return frozenset(found)
