"""Registered reviewer-facing Action packet; no sampling reasons or predictions."""
from ..stage_interfaces import ActionReviewExportInput, ActionReviewExportOutput
from ..review_export import export_action_reviews
from ..schemas import require


def run(inputs: ActionReviewExportInput) -> ActionReviewExportOutput:
    require(inputs.execution_authorized, "Action review export disabled")
    return ActionReviewExportOutput(export_action_reviews(inputs.annotation_cases, inputs.annotations,
        inputs.action_assignments, inputs.action_protocol))
