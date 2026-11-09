"""Completed feature review results are evaluation artifacts, never feature overwrites."""
from ..stage_interfaces import FeatureReviewImportInput, FeatureReviewImportOutput
from ..review_import import import_feature_reviews
from ..schemas import require


def run(inputs: FeatureReviewImportInput) -> FeatureReviewImportOutput:
    require(inputs.execution_authorized, "Human feature review import disabled")
    reviews, audit = import_feature_reviews(inputs.feature_review_export.packet,
        inputs.feature_review_source, inputs.responses, inputs.settings)
    return FeatureReviewImportOutput(reviews, audit)
