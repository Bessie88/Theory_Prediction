"""Feature specification registry view committed through the ONE artifact registry."""
from dataclasses import replace
from .measurement_contracts import (
    FeatureRegistrySnapshot, FrozenMeasurements, specification_hash, validate_specification, procedure_for,
)
from .workflow_contracts import stable_id
from .condition_normalization import validate_binding
from .semantic_features import PROMPT_VERSION, SYSTEM_PROMPT
from .schemas import require


def feature_id_for(spec):
    # Meaning/procedure identity shared by Condition and Boundary roles.
    fields = ("feature_name", "operational_definition", "value_type", "allowed_values",
              "measurement_target", "observable_indicators", "positive_criteria", "negative_criteria",
              "ambiguity_criteria", "required_input_fields", "extraction_procedure_json",
              "construct_kind", "structured_schema_name", "structured_fields", "version",
              "evidence_requirements", "is_proxy", "limitations")
    return stable_id("feature", {key: spec.to_dict()[key] for key in fields})


class FeatureSpecificationRegistry:
    def __init__(self, specifications=()):
        self._specifications = {}
        for spec in specifications:
            self.add(spec)

    def add(self, spec):
        validate_specification(spec)
        require(spec.feature_id == feature_id_for(spec), "Noncanonical feature ID")
        previous = self._specifications.get(spec.feature_id)
        if previous is not None:
            # Only provenance/ownership associations may be merged. Semantics cannot drift.
            merged = {}
            for field in ("associated_theory_ids", "associated_rule_ids", "roles",
                          "original_clause_ids", "source_provenance_ids"):
                merged[field] = tuple(sorted(set(getattr(previous, field)) | set(getattr(spec, field))))
            candidate = replace(previous, **merged)
            other = replace(spec, **merged)
            require(candidate == other, "Feature ID collision with different procedure/version")
            spec = candidate
        self._specifications[spec.feature_id] = spec

    def get(self, feature_id, version=None):
        spec = self._specifications[feature_id]
        require(version is None or spec.version == version, "Requested feature version unavailable")
        return spec

    def snapshot(self):
        specs = tuple(self._specifications[key] for key in sorted(self._specifications))
        prompts = tuple(sorted({s.prompt_version for s in specs if s.prompt_version}))
        versions = tuple(sorted({s.extractor_version for s in specs} |
                                {step.tool_version for s in specs for step in procedure_for(s).steps}))
        prompt_records = ((PROMPT_VERSION, SYSTEM_PROMPT),) if prompts else ()
        key = stable_id("feature_registry", {
            "specifications": [s.to_dict() for s in specs], "prompts": prompt_records,
            "extractors": versions})
        return FeatureRegistrySnapshot(specs, key, prompts, versions,
                                       SYSTEM_PROMPT if prompts else "", "feature-payload-v1")


def freeze_measurements(snapshot, clauses, bindings, reviews, source_theory_hash):
    require(len(source_theory_hash) == 64 and all(c in "0123456789abcdef" for c in source_theory_hash),
            "Original theory SHA-256 required for freezing")
    by_id = {s.clause_id: s for s in clauses}
    require(len(by_id) == len(clauses) and {b.clause_id for b in bindings} == set(by_id),
            "Every original Condition/Boundary needs an explicit binding/status")
    require(len(bindings) == len(by_id), "Duplicate clause binding")
    for binding in bindings:
        validate_binding(by_id[binding.clause_id], binding, snapshot.specifications)
    registry = FeatureSpecificationRegistry(snapshot.specifications)
    require(registry.snapshot() == snapshot, "Feature registry/prompt/version snapshot changed")
    unresolved = tuple(sorted(b.clause_id for b in bindings
                              if b.resolution not in ("RESOLVED", "NO_ADDITIONAL_BOUNDARY")))
    review_completed = bool(reviews) and all(r.status == "REVIEWED" for r in reviews)
    digest = stable_id("measurement_freeze", {
        "snapshot": snapshot.to_dict(), "clauses": [c.to_dict() for c in clauses],
        "bindings": [b.to_dict() for b in bindings], "reviews": [r.to_dict() for r in reviews],
        "source_theory_hash": source_theory_hash})
    return FrozenMeasurements(snapshot, clauses, bindings, reviews, source_theory_hash,
                              digest, unresolved, review_completed)
