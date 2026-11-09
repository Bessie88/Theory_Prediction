"""Blind annotation/diff contracts. No prediction, theory, policy or correctness fields."""
from dataclasses import dataclass
from .workflow_contracts import (
    Record, CaseIdentity, EvidenceSpan, PreEditField, ObservedAction, Method, Period, ActionDefinition,
)
from .action_comparison import FrozenActionProtocol
from .schemas import require


@dataclass(frozen=True)
class TextToken(Record):
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class DiffHunk(Record):
    hunk_id: str
    granularity: str
    operation: str
    before: EvidenceSpan
    after: EvidenceSpan
    before_token_start: int
    before_token_end: int
    after_token_start: int
    after_token_end: int
    before_token_exact: bool
    after_token_exact: bool

    def validate(self):
        require(self.granularity in ("CHARACTER", "TOKEN") and
                self.operation in ("EQUAL", "INSERT", "DELETE", "REPLACE"), "Invalid text opcode")
        require(self.before.side == "before" and self.after.side == "after", "Wrong diff side")


@dataclass(frozen=True)
class TextDiff(Record):
    before_hash: str
    after_hash: str
    version: str
    before_tokens: tuple[TextToken, ...]
    after_tokens: tuple[TextToken, ...]
    character_hunks: tuple[DiffHunk, ...]
    token_hunks: tuple[DiffHunk, ...]

    @property
    def changed_hunks(self):
        return tuple(h for h in self.character_hunks if h.operation != "EQUAL")


@dataclass(frozen=True)
class AnnotationCase(Record):
    identity: CaseIdentity
    before_text: str
    after_text: str | None
    actual_diff: TextDiff | None
    context: tuple[PreEditField, ...]
    input_hash: str
    availability: str
    limitations: tuple[str, ...]

    def validate(self):
        require(self.identity.period is Period.LATER and self.input_hash, "Later annotation identity required")
        require(self.availability in ("AVAILABLE", "UNAVAILABLE", "TOO_LARGE"), "Invalid annotation availability")
        if self.availability == "AVAILABLE":
            require(self.after_text is not None and self.actual_diff is not None, "Available pair/diff required")
        else:
            require(self.actual_diff is None, "Unavailable units must not invent a diff")


@dataclass(frozen=True)
class AnnotationCases(Record):
    cases: tuple[AnnotationCase, ...]


@dataclass(frozen=True)
class AnnotationOperationResponse(Record):
    action_id: str
    diff_hunk_ids: tuple[str, ...]
    before_evidence: tuple[EvidenceSpan, ...]
    after_evidence: tuple[EvidenceSpan, ...]
    explanation: str
    confidence: str
    uncertainty: tuple[str, ...]


@dataclass(frozen=True)
class AnnotationResponse(Record):
    status: str
    operations: tuple[AnnotationOperationResponse, ...]
    unassigned_diff_hunk_ids: tuple[str, ...]
    explanation: str
    confidence: str
    uncertainty: tuple[str, ...]
    new_actions: tuple[ActionDefinition, ...] = ()

    def validate(self):
        require(self.status in ("ANNOTATED", "OTHER", "UNCERTAIN", "NO_CHANGE"), "Invalid annotation status")
        require(self.explanation.strip() and self.confidence in ("HIGH", "MEDIUM", "LOW", "UNCERTAIN"),
                "Annotation explanation/confidence required")


@dataclass(frozen=True)
class AnnotatorIndependence(Record):
    annotator_family: str | None
    feature_extractor_families: tuple[str, ...]
    status: str
    rationale: str


@dataclass(frozen=True)
class AnnotationAttempt(Record):
    identity: CaseIdentity
    input_hash: str
    raw_response_json: str
    validation_status: str
    reason_codes: tuple[str, ...]
    independence: AnnotatorIndependence


@dataclass(frozen=True)
class AnnotationAudit(Record):
    attempts: tuple[AnnotationAttempt, ...]
