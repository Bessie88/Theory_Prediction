"""Registered Before-only feature packet; no ownership, predicted Actions or After."""
from ..stage_interfaces import FeatureReviewExportInput, FeatureReviewExportOutput
from ..review_export import export_feature_reviews
from ..schemas import require


def run(inputs: FeatureReviewExportInput) -> FeatureReviewExportOutput:
    require(inputs.execution_authorized, "Feature review export disabled")
    return FeatureReviewExportOutput(export_feature_reviews(inputs.feature_review_source,
        inputs.feature_assignments, inputs.action_protocol))
