"""Registered Before-only feature packet; no ownership, predicted Actions or After."""
from theory_discovery.core.stage_interfaces import FeatureReviewExportInput, FeatureReviewExportOutput
from theory_discovery.annotation.review_export import export_feature_reviews
from theory_discovery.core.schemas import require


def run(inputs: FeatureReviewExportInput) -> FeatureReviewExportOutput:
    require(inputs.execution_authorized, "Feature review export disabled")
    return FeatureReviewExportOutput(export_feature_reviews(inputs.feature_review_source,
        inputs.feature_assignments, inputs.action_protocol))
