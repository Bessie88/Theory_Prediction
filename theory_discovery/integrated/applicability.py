"""One feature artifact -> two source-tagged runs of the SAME deterministic matcher."""
from ..stage_interfaces import ApplicabilityInput, ApplicabilityOutput
from ..theory_applicability import apply_knowledge
from ..policy_contracts import BaselineComparability, PolicySourceUnavailable
from ..workflow_contracts import Method
from ..schemas import require, stable_id


def run(inputs: ApplicabilityInput) -> ApplicabilityOutput:
    require(inputs.execution_authorized, "Rule applicability execution disabled")
    if inputs.study.policy_registry is None:
        raise PolicySourceUnavailable("POLICY_SOURCE_UNAVAILABLE: both frozen knowledge sources required")
    theory, theory_features = apply_knowledge(inputs.features, inputs.study, Method.THEORY)
    policy, policy_features = apply_knowledge(inputs.features, inputs.study, Method.NPOV)
    require(theory.feature_input_artifact_hash == policy.feature_input_artifact_hash,
            "Sources received different prediction inputs")
    comparison = BaselineComparability(
        theory.feature_input_artifact_hash, tuple(c.identity.case_id for c in inputs.features.cases),
        stable_id("neutral_action_catalog", inputs.study.catalog.to_dict()),
        inputs.study.theory_registry.registry_hash, inputs.study.policy_registry.registry_hash,
        inputs.study.policy_registry.source_snapshot_id, "PredictionCase-before-v1",
        "one-frozen-measurement-executor-v1", inputs.study.require_shared_action_ids)
    return ApplicabilityOutput(theory, theory_features, policy, policy_features, comparison)
