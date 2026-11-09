"""Completed Action response ingestion; absent replies remain pending."""
from ..stage_interfaces import HumanReviewInput, HumanReviewOutput, HumanReviewBatch
from ..review_import import import_action_reviews
from ..schemas import require


def run(inputs: HumanReviewInput) -> HumanReviewOutput:
    require(inputs.execution_authorized, "Human review import disabled")
    records, audit = import_action_reviews(inputs.action_review_export.packet,
        inputs.annotation_cases, inputs.annotations, inputs.action_protocol, inputs.responses, inputs.settings)
    return HumanReviewOutput(HumanReviewBatch(records), audit)
