"""Connected deterministic source-action prediction, without outcomes/annotation."""
from theory_discovery.core.stage_interfaces import TheoryPredictionInput, TheoryPredictionOutput
from theory_discovery.prediction.theory_action_prediction import predict_actions
from theory_discovery.core.schemas import require


def run(inputs: TheoryPredictionInput) -> TheoryPredictionOutput:
    require(inputs.execution_authorized, "Theory prediction execution disabled")
    predictions, provenance = predict_actions(inputs.features, inputs.applicability, inputs.study)
    return TheoryPredictionOutput(predictions, provenance)
