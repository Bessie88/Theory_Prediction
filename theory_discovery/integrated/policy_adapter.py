"""Official snapshot and neutral catalog ONLY -> independently registered policy rules."""
from theory_discovery.core.stage_interfaces import PolicyAdapterInput, PolicyAdapterOutput
from theory_discovery.prediction.policy_adapter import adapt_policy, PROMPT_VERSION
from theory_discovery.features.measurement_runtime import configured_backend
from theory_discovery.core.schemas import require


def run(inputs: PolicyAdapterInput) -> PolicyAdapterOutput:
    require(inputs.runtime.execution_authorized, "Policy conversion execution disabled")
    backend = None
    if inputs.policy.conversion_enabled and inputs.policy.converted_rule_packet_json is None:
        backend = configured_backend(inputs.policy.conversion_model, inputs.runtime, PROMPT_VERSION)
    return PolicyAdapterOutput(adapt_policy(inputs.policy_snapshot, inputs.catalog, backend,
                                           inputs.policy.converted_rule_packet_json))
