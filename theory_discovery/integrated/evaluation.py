"""One central evaluator for both frozen prediction branches and the same gold view."""
from theory_discovery.core.stage_interfaces import EvaluationInput, EvaluationOutput, EvaluationBatch
from theory_discovery.core.workflow_contracts import EvaluationResult, RuleAnalysis
from theory_discovery.evaluation.shared_evaluation import evaluate, metric
from theory_discovery.core.schemas import require

def run(inputs: EvaluationInput) -> EvaluationOutput:
    require(inputs.execution_authorized, "Evaluation disabled")
    details = evaluate(inputs)
    results = tuple(EvaluationResult(m.method, details.editor, m.case_ids, m.metrics[0], m.metrics,
        tuple(RuleAnalysis(r.owner_id, (
            metric("matched_cases_per_dataset", r.matched_cases, len(m.case_ids)),
            metric("exact_ensemble_accuracy_among_scorable_owner_matches", r.exact_matches, r.scored_matched_cases)))
            for r in m.owner_metrics), (), tuple(x for x in m.metrics if x.name.startswith("revision_")), inputs.settings.version)
        for m in details.methods)
    return EvaluationOutput(EvaluationBatch(results), details)
