"""Explicit, gated local official-policy ingestion; no downloader."""
from theory_discovery.core.stage_interfaces import PolicySnapshotInput, PolicySnapshotOutput
from theory_discovery.prediction.policy_ingestion import ingest_policy
from theory_discovery.core.schemas import require


def run(inputs: PolicySnapshotInput) -> PolicySnapshotOutput:
    require(inputs.execution_authorized, "Policy ingestion execution disabled")
    return PolicySnapshotOutput(ingest_policy(inputs.policy))
