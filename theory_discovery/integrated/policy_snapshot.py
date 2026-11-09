"""Explicit, gated local official-policy ingestion; no downloader."""
from ..stage_interfaces import PolicySnapshotInput, PolicySnapshotOutput
from ..policy_ingestion import ingest_policy
from ..schemas import require


def run(inputs: PolicySnapshotInput) -> PolicySnapshotOutput:
    require(inputs.execution_authorized, "Policy ingestion execution disabled")
    return PolicySnapshotOutput(ingest_policy(inputs.policy))
