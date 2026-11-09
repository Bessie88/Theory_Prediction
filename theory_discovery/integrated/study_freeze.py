"""Freeze two independent knowledge sources against ONE feature and Action registry."""
from dataclasses import replace
from ..stage_interfaces import FreezeInput, FreezeOutput, StudySpecification
from ..feature_spec_registry import FeatureSpecificationRegistry, freeze_measurements
from ..theory_application_registry import compile_theory_registry, compile_knowledge_registry, validate_study
from ..application_contracts import study_hash, ApplicationSettings
from ..policy_contracts import policy_registry_hash, PolicySourceUnavailable
from ..policy_projection import planning_bundle, attach_policy_application
from ..workflow_contracts import Method
from ..action_comparison import validate_protocol
from ..schemas import require, stable_id


def run(inputs: FreezeInput) -> FreezeOutput:
    validate_protocol(inputs.action_protocol)
    require(inputs.action_protocol.actions == inputs.catalog.actions and
            inputs.action_protocol.catalog_version == inputs.catalog.version, "Private annotation label space forbidden")
    measurement, policy = inputs.measurement, inputs.policy_plan
    if policy.rules is None or policy.canonical_bundle is None or policy.measurement is None:
        raise PolicySourceUnavailable("POLICY_SOURCE_UNAVAILABLE: no verified policy planning bundle")
    require(policy.rules.registry_hash == policy_registry_hash(policy.rules), "Policy rules changed")
    require(policy.source_snapshot == policy.rules.snapshot_reference and
            policy.canonical_bundle.source_snapshot_id == policy.rules.snapshot.snapshot_id and
            policy.canonical_bundle.editor == inputs.theories.editor and
            policy.measurement.knowledge_source is Method.NPOV, "Policy projection/source/editor mismatch")
    expected_bundle = planning_bundle(policy.rules, inputs.theories.editor, inputs.catalog)
    require(policy.canonical_bundle == expected_bundle and
            policy.measurement == attach_policy_application(policy.measurement, expected_bundle, policy.rules) and
            not policy.measurement.action_compatibility and
            policy.policy_version == policy.rules.snapshot.revision_id,
            "Policy projection/application differs from registered source rules")
    require(measurement.feature_registry == inputs.feature_specifications,
            "Theory feature registry and measurement plan differ")
    for planned in (measurement, policy.measurement):
        require(planned.feature_registry is not None, "Missing source measurement registry")
        available = {s.feature_id: s for s in planned.feature_registry.specifications}
        require(all(available.get(s.feature_id) == s for s in planned.specifications),
                "Active definition differs from registered version")
    require(policy.specifications == policy.measurement.specifications and
            policy.match_plans == policy.measurement.match_plans, "Policy planning ports differ")
    registry = FeatureSpecificationRegistry(measurement.specifications + policy.specifications)
    snapshot = registry.snapshot()
    frozen = freeze_measurements(snapshot, measurement.clauses, measurement.bindings,
                                 measurement.reviews, measurement.source_theory_hash)
    pf = freeze_measurements(snapshot, policy.measurement.clauses, policy.measurement.bindings,
                             policy.measurement.reviews, policy.measurement.source_theory_hash)
    theory_registry = compile_theory_registry(inputs.theories, measurement, inputs.application, inputs.catalog)
    # The theory application's configured source declarations must not authorize policy rules.
    # Policy declarations/aliases come exclusively from the verified policy adapter.
    policy_registry = compile_knowledge_registry(
        policy.canonical_bundle, policy.measurement, ApplicationSettings(), inputs.catalog)
    study = StudySpecification(
        measurement.specifications, policy.specifications, measurement.match_plans,
        policy.match_plans, inputs.catalog, "", "source-backed-compatibility-v1",
        frozen, theory_registry, policy.source_snapshot.sha256, policy_registry, pf,
        policy.rules.registry_hash, True, inputs.action_protocol.protocol_hash)
    require(inputs.evaluation_settings.protocol_frozen, "Evaluation policy is not frozen")
    study = replace(study, evaluation_protocol_hash=stable_id("evaluation_protocol", inputs.evaluation_settings.to_dict()))
    study = replace(study, freeze_hash=study_hash(study))
    validate_study(study)
    return FreezeOutput(study)
