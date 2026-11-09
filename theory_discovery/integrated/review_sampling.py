"""Selection worker; private success/stratum audit is never sent to reviewers."""
from theory_discovery.core.stage_interfaces import ReviewSamplingInput, ReviewSamplingOutput
from theory_discovery.annotation.review_sampling import select_reviews
from theory_discovery.core.schemas import require


def run(inputs: ReviewSamplingInput) -> ReviewSamplingOutput:
    require(inputs.execution_authorized, "Human review sampling disabled")
    return ReviewSamplingOutput(*select_reviews(inputs.theory_predictions, inputs.npov_predictions,
        inputs.annotations, inputs.feature_review_source, inputs.action_protocol, inputs.settings))
