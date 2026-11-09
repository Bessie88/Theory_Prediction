"""NPOV is a source adapter to the existing planner/matcher/action engine."""
from dataclasses import replace
from theory_discovery.core.stage_interfaces import (
    NPOVPreparationInput, NPOVPreparationOutput, NPOVPredictionInput,
    NPOVPredictionOutput, MeasurementPlanningInput, PolicyPlan,
)
from theory_discovery.prediction.policy_projection import planning_bundle, attach_policy_application
from theory_discovery.prediction.policy_contracts import PolicySourceUnavailable
from theory_discovery.prediction.theory_action_prediction import predict_knowledge
from theory_discovery.core.workflow_contracts import Method
from theory_discovery.core.schemas import require
from .measurement_planner import run as shared_planning


def prepare(inputs: NPOVPreparationInput, *, progress=None) -> NPOVPreparationOutput:
    require(inputs.runtime.execution_authorized, "Policy measurement planning execution disabled")
    require(bool(inputs.early_before.records), "An explicitly selected editor's EARLY manifest is required")
    editor = inputs.early_before.records[0].identity.editor
    require(all(c.identity.editor == editor for c in inputs.early_before.records), "Mixed policy planning editors")
    rules = inputs.policy_rules
    bundle = planning_bundle(rules, editor, inputs.catalog)
    # Same algorithm/model/budget. Only source-specific frozen definitions differ.
    settings = replace(inputs.settings,
        seed_specifications_json=inputs.policy.seed_specifications_json,
        seed_bindings_json=inputs.policy.seed_bindings_json)
    planned = shared_planning(MeasurementPlanningInput(
        bundle, inputs.early_before, inputs.catalog, settings, inputs.runtime), progress=progress).measurement
    planned = attach_policy_application(planned, bundle, rules)
    return NPOVPreparationOutput(PolicyPlan(
        rules.snapshot.revision_id, planned.specifications, planned.match_plans,
        rules.snapshot_reference, planned.unresolved, rules, bundle, planned))


def predict(inputs: NPOVPredictionInput) -> NPOVPredictionOutput:
    require(inputs.execution_authorized, "Policy prediction execution disabled")
    if inputs.study.policy_registry is None:
        raise PolicySourceUnavailable("POLICY_SOURCE_UNAVAILABLE: policy registry not frozen")
    predictions, provenance = predict_knowledge(
        inputs.features, inputs.applicability, inputs.study, Method.NPOV)
    return NPOVPredictionOutput(predictions, provenance)
