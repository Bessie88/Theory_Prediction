"""Frozen boundary counterfactual. No feature extraction or model calls."""
from itertools import product
from .theory_logic import evaluate_logic, truth_or, clause_references
from theory_discovery.core.schemas import require, stable_id
from theory_discovery.core.application_contracts import LogicNodeTrace
WITH = "WITH_BOUNDARIES"
WITHOUT = "WITHOUT_BOUNDARIES_EXISTENTIAL_V1"

def evaluate_application(expression, values, boundary_ids, mode, gate=None):
    if mode == WITH:
        return evaluate_logic(gate if gate is not None else expression, values)
    require(mode == WITHOUT, "Unknown ablation protocol")
    joint = {"op": "and", "args": [expression, gate]} if gate is not None else expression
    removed = sorted(clause_references(joint) & set(boundary_ids))
    if len(removed) > 8:
        return None, (LogicNodeTrace("$ablation", "BOUNDARY_BUDGET_EXCEEDED", None, None),)
    outcomes = [evaluate_logic(joint, {**values, **dict(zip(removed, assignment))})[0]
                for assignment in product((False, True), repeat=len(removed))]
    truth = truth_or(outcomes)
    return truth, (LogicNodeTrace("$ablation", "EXISTS_BOUNDARIES", truth, None),)

def analyze_boundaries(inputs):
    from theory_discovery.prediction.theory_applicability import apply_knowledge
    from theory_discovery.prediction.theory_action_prediction import predict_knowledge
    from theory_discovery.evaluation.shared_evaluation import score_predictions, summarize
    from theory_discovery.evaluation.evaluation_contracts import BoundaryAnalysis, AblationChange
    from theory_discovery.core.workflow_contracts import Method
    import json
    require(inputs.execution_authorized, "Ablation disabled")
    require(inputs.theory_predictions.study_spec_hash == inputs.study.freeze_hash ==
            inputs.evaluation_details.study_hash, "Ablation freeze mismatch")
    decisions, _ = apply_knowledge(inputs.features, inputs.study, Method.THEORY)
    original, _ = predict_knowledge(inputs.features, decisions, inputs.study, Method.THEORY)
    require(original == inputs.theory_predictions, "Primary prediction differs from deterministic replay")
    decisions, _ = apply_knowledge(inputs.features, inputs.study, Method.THEORY, WITHOUT)
    relaxed, relaxed_provenance = predict_knowledge(inputs.features, decisions, inputs.study, Method.THEORY, WITHOUT)
    rows = score_predictions(original, inputs.evaluation_details.gold, inputs.action_protocol)
    other = score_predictions(relaxed, inputs.evaluation_details.gold, inputs.action_protocol)
    index = {r.identity.case_id: r for r in other}
    changes = []
    for r in rows:
        o = index[r.identity.case_id]
        excluded = o.usable and not r.usable
        changes.append(AblationChange(r.identity, r.prediction_status, o.prediction_status,
            r.predicted_ids, o.predicted_ids, r.match, o.match, excluded,
            excluded and o.gold_scorable and o.match in ("NO_MATCH", "PARTIAL"),
            (r.prediction_status, r.predicted_ids) != (o.prediction_status, o.predicted_ids)))
    return BoundaryAnalysis(inputs.study.freeze_hash, stable_id("feature_batch", inputs.features.to_dict()), WITHOUT,
        summarize(Method.THEORY, rows, inputs.action_protocol),
        summarize(Method.THEORY, other, inputs.action_protocol), tuple(changes),
        ("Existential boundary-variable projection preserves arbitrary AND/OR/NOT, not a naive true substitution.",
         "Action support projects applicability AND its gate jointly. Conditions stay measured.",
         "Projection is per action; original action composition/conflict handling is retained.",
         "More than eight boundary variables per formula yields explicit UNKNOWN.",
         "Errors prevented by abstention are not new correct predictions. Report coverage and paired changes together.",
         "Frozen diagnostic only; no retuning, new features or causal interpretation."),
        json.dumps(relaxed.to_dict(), sort_keys=True), json.dumps(relaxed_provenance.to_dict(), sort_keys=True))
