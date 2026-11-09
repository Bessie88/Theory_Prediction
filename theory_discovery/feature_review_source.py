"""Sanitized feature review projection; removes ownership, Theory/Policy and source-rule maps."""
from dataclasses import fields
from .review_contracts import (
    FeatureReviewDefinition, FeatureReviewLabel, FeatureReviewSourceRecord, FeatureReviewSource,
)
from .measurement_contracts import specification_hash
from .schemas import require, stable_id


def build_feature_review_source(features, specification_snapshot):
    specs = {s.feature_id: s for s in specification_snapshot.specifications}
    cases = {c.identity.case_id: c for c in features.cases}
    records = []
    for observation in features.observations:
        spec = specs[observation.feature_id]
        case = cases[observation.identity.case_id]
        require(observation.identity == case.identity and
                observation.specification_hash == specification_hash(spec) and
                observation.input_hash == stable_id("feature_input", case.to_dict()),
                "Feature review source/specification lineage mismatch")
        definition = FeatureReviewDefinition(**{f.name: getattr(spec, f.name) for f in fields(FeatureReviewDefinition)})
        metadata = {m.key: m.value for m in observation.extraction_metadata}
        label = FeatureReviewLabel(observation.extracted_value, observation.status.value,
            observation.evidence, observation.context_evidence_names,
            metadata.get("evidence_scope", "UNAVAILABLE"),
            metadata.get("rationale") or "; ".join(observation.uncertainty) or observation.status.value,
            observation.uncertainty)
        records.append(FeatureReviewSourceRecord(
            observation.identity, observation.observation_id, case, definition, label,
            stable_id("raw_feature_observation", observation.to_dict()), observation.specification_hash))
    return FeatureReviewSource(tuple(records))
