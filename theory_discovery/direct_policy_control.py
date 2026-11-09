"""Optional raw-policy diagnostic interface; excluded from the primary DAG.

No model/client is constructed here. A future caller must explicitly opt in,
supply a permitted backend and commit DirectPolicyControlBatch through the SAME
artifact registry using direct_policy_stage_spec(). Primary NPOV ports cannot be
overwritten by this separate stage/field/type.
"""
from dataclasses import dataclass
from typing import Protocol
from .workflow_contracts import (
    Record, PredictionCase, CaseIdentity, ActionInstance, PredictionStatus, Period,
)
from .policy_contracts import PolicySnapshot, PolicyCitation
from .policy_ingestion import validate_citation
from .schemas import require, stable_id
from .llm import PLAIN_LANGUAGE_INSTRUCTION

PROMPT_VERSION = "direct-official-npov-control-v2"
SYSTEM = (
    "Use only the supplied official NPOV snapshot and this Before to identify supported catalog Actions. "
    "Keep allowable choices and uncertainty; do not invent an unspecified edit. Cite exact policy spans. "
    "Return only status, action_ids, citations, uncertainty and action_composition in the supplied schema."
) + "\n" + PLAIN_LANGUAGE_INSTRUCTION


class DirectPolicyBackend(Protocol):
    def complete(self, system: str, payload: dict) -> dict:
        ...


@dataclass(frozen=True)
class DirectPolicyControlRecord(Record):
    identity: CaseIdentity
    diagnostic_id: str
    policy_snapshot_id: str
    input_hash: str
    catalog_hash: str
    status: str
    candidate_action_ids: tuple[str, ...]
    citations: tuple[PolicyCitation, ...]
    uncertainty: tuple[str, ...]
    action_composition: str
    prompt_version: str

    def validate(self):
        require(self.identity.period is Period.LATER and
                self.status in ("PREDICTED", "NOT_COVERED", "UNKNOWN", "CONFLICT"),
                "Invalid raw-policy diagnostic")
        require(self.action_composition in ("SINGLE", "ALL_SUPPORTED", "ALLOWABLE_ALTERNATIVES", "UNRESOLVED"),
                "Diagnostic must preserve operation ambiguity")
        require(self.status != "PREDICTED" or
                (self.action_composition in ("SINGLE", "ALL_SUPPORTED") and
                 (self.action_composition != "SINGLE" or len(self.candidate_action_ids) == 1)),
                "Allowable alternatives cannot masquerade as a unique prediction")
        require(self.status != "PREDICTED" or (self.candidate_action_ids and self.citations),
                "Confident diagnostic needs policy evidence")


@dataclass(frozen=True)
class DirectPolicyControlBatch(Record):
    records: tuple[DirectPolicyControlRecord, ...]


@dataclass(frozen=True)
class DirectPolicyControlInput(Record):
    policy_snapshot: PolicySnapshot
    later: "LaterRecords"
    catalog: "NeutralActionCatalog"
    policy: "PolicySettings"
    runtime: "MeasurementRuntime"


@dataclass(frozen=True)
class DirectPolicyControlOutput(Record):
    direct_npov_control: DirectPolicyControlBatch


def direct_policy_control(snapshot, cases, catalog, settings, runtime, backend):
    # Check all gates before any backend invocation.
    require(settings.direct_control_enabled and runtime.execution_authorized and
            runtime.models_authorized and backend is not None, "Direct policy diagnostic disabled")
    catalog_ids = {action.action_id for action in catalog.actions}
    catalog_hash = stable_id("neutral_action_catalog", catalog.to_dict())
    records = []
    for case in cases:
        require(type(case) is PredictionCase and case.identity.period is Period.LATER,
                "Direct policy diagnostic requires the same Before-only case type")
        raw = backend.complete(SYSTEM, {
            "official_policy_text": snapshot.content, "policy_metadata": snapshot.metadata.to_dict(),
            "case": case.to_dict(), "neutral_action_catalog": [a.to_dict() for a in catalog.actions],
            "prompt_version": PROMPT_VERSION,
            "output_schema": {"status": ["PREDICTED", "NOT_COVERED", "UNKNOWN", "CONFLICT"],
                "action_ids": "catalog ID array", "citations": "PolicyCitation array",
                "uncertainty": "string array", "action_composition":
                ["SINGLE", "ALL_SUPPORTED", "ALLOWABLE_ALTERNATIVES", "UNRESOLVED"]}})
        require(isinstance(raw, dict) and set(raw) == {"status", "action_ids", "citations", "uncertainty", "action_composition"},
                "Invalid diagnostic response fields")
        require(isinstance(raw["action_ids"], list) and isinstance(raw["citations"], list) and
                isinstance(raw["uncertainty"], list) and
                all(type(s) is str for s in raw["uncertainty"]), "Invalid diagnostic response types")
        actions = tuple(raw["action_ids"])
        require(all(type(a) is str for a in actions) and len(set(actions)) == len(actions) and
                set(actions) <= catalog_ids, "Diagnostic cannot extend the common label space")
        citations = tuple(PolicyCitation.from_dict(row) for row in raw["citations"])
        for citation in citations:
            validate_citation(snapshot, citation)
        record = DirectPolicyControlRecord(
            case.identity, stable_id("direct_policy_diagnostic", [
                case.to_dict(), snapshot.metadata.snapshot_id, catalog_hash, runtime.configuration_id]),
            snapshot.metadata.snapshot_id, case.before_hash, catalog_hash, raw["status"], actions,
            citations, tuple(raw["uncertainty"]), raw["action_composition"], PROMPT_VERSION)
        records.append(record)
    return DirectPolicyControlBatch(tuple(records))


def direct_policy_stage_spec():
    # Lazy imports avoid a second orchestrator and an import cycle.
    from .stage_dependencies import StageSpec, b
    return StageSpec("direct_npov_control", "6", "direct_npov_control",
        DirectPolicyControlInput, DirectPolicyControlOutput,
        (b("policy_snapshot", "policy_snapshot"), b("later", "data_preparation"),
         b("catalog", "data_preparation")), (("direct_npov_control", "PREDICTION_RESULT"),))


# Only used by Record's explicit future deserializer; no execution on import.
from .stage_interfaces import LaterRecords, NeutralActionCatalog
from .workflow_config import PolicySettings
from .measurement_contracts import MeasurementRuntime
