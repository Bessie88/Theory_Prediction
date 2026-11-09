"""Explicit opt-in raw-policy diagnostic using the shared registry/configuration."""
from ..direct_policy_control import (
    DirectPolicyControlInput, DirectPolicyControlOutput, direct_policy_control, PROMPT_VERSION,
)
from ..measurement_runtime import configured_backend
from ..schemas import require


def run(inputs: DirectPolicyControlInput) -> DirectPolicyControlOutput:
    require(inputs.policy.direct_control_enabled and inputs.runtime.execution_authorized and
            inputs.runtime.models_authorized, "Direct policy diagnostic disabled")
    backend = configured_backend(inputs.policy.direct_control_model, inputs.runtime, PROMPT_VERSION)
    return DirectPolicyControlOutput(direct_policy_control(
        inputs.policy_snapshot, inputs.later.records, inputs.catalog,
        inputs.policy, inputs.runtime, backend))
