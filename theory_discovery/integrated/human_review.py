"""Completed Action response ingestion; absent replies remain pending."""
from theory_discovery.core.stage_interfaces import HumanReviewInput, HumanReviewOutput, HumanReviewBatch
from theory_discovery.annotation.review_import import import_action_reviews
from theory_discovery.core.schemas import require


def run(inputs: HumanReviewInput) -> HumanReviewOutput:
    require(inputs.execution_authorized, "Human review import disabled")
    records, audit = import_action_reviews(inputs.action_review_export.packet,
        inputs.annotation_cases, inputs.annotations, inputs.action_protocol, inputs.responses, inputs.settings)
    return HumanReviewOutput(HumanReviewBatch(records), audit)
