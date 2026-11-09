"""Freeze one neutral ontology/guidelines/comparison before Later processing."""
from dataclasses import replace
from theory_discovery.core.stage_interfaces import ActionProtocolInput, ActionProtocolOutput
from theory_discovery.annotation.action_comparison import FrozenActionProtocol, protocol_hash
from theory_discovery.core.schemas import require


def run(inputs: ActionProtocolInput) -> ActionProtocolOutput:
    require(inputs.execution_authorized, "Action protocol freeze disabled")
    settings = inputs.settings
    protocol = FrozenActionProtocol(inputs.catalog.version, inputs.catalog.actions,
        settings.guidelines_version, settings.guidelines, settings.general_guidelines, settings.comparison, "")
    return ActionProtocolOutput(replace(protocol, protocol_hash=protocol_hash(protocol)))
