"""Before-only targeting separated from evaluation-only outcome alignment."""
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol
import difflib
import re
from .workflow_contracts import Record, EditorKey, PreEditField, utc_time
from .schemas import stable_id, require


@dataclass(frozen=True)
class RevisionBefore(Record):
    editor: EditorKey
    revision_id: str
    parent_revision_id: str
    before_text: str
    parent_timestamp: str
    pre_context: tuple[PreEditField, ...]

    def validate(self):
        at = utc_time(self.parent_timestamp)
        require(all(utc_time(field.available_at_utc) <= at for field in self.pre_context),
                "Targeting cannot see future context")


@dataclass(frozen=True)
class UnitAnchor(Record):
    prediction_unit_id: str
    start: int
    end: int
    policy_version: str


class BeforeOnlyTargeter(Protocol):
    def enumerate_units(self, before: RevisionBefore) -> tuple[UnitAnchor, ...]:
        """No target After, diff, labels, or LoadedRevision argument is accepted."""
        ...


def provided_sentence_unit(before, sentence_pair_id, version):
    """Identity for an externally fixed edited pair; never enumerate its sentences.

    Sample selection is explicitly conditional on an edit. Only the supplied ID,
    Before and protocol version define identity; After/diff are not accepted here.
    """
    require(type(sentence_pair_id) is str and bool(sentence_pair_id.strip()) and bool(version),
            "A provided sentence pair needs an explicit stable ID and policy version")
    digest = sha256(before.before_text.encode()).hexdigest()
    return UnitAnchor(stable_id("provided_sentence_unit", [sentence_pair_id, digest, version]),
                      0, len(before.before_text), version)


class BuiltinBeforeTargeter:
    def __init__(self, kind, version):
        require(kind in ("sentence_v1", "whole_revision"), "Configure a supported Before-only targeter")
        require(bool(version), "Explicit targeting policy version required")
        self.kind, self.version = kind, version

    def enumerate_units(self, before):
        text = before.before_text
        cuts = [0]
        if self.kind == "sentence_v1":
            cuts.extend(match.end() for match in re.finditer(r"(?<=[.!?])\s+", text)
                        if match.end() < len(text))
        cuts.append(len(text))
        digest = sha256(text.encode()).hexdigest()
        return tuple(UnitAnchor(stable_id("unit", [digest, a, b, self.kind, self.version]),
                                a, b, self.version) for a, b in zip(cuts, cuts[1:]))


def validate_units(before, units):
    require(bool(units), "Every revision needs a visible target/unsupported placeholder")
    require(len({u.prediction_unit_id for u in units}) == len(units), "Duplicate prediction units")
    cursor = 0
    for unit in units:
        require(unit.start == cursor and unit.start <= unit.end <= len(before.before_text),
                "Units must partition Before exactly, in source order")
        require(unit.end > unit.start or not before.before_text, "Empty internal prediction unit")
        cursor = unit.end
    require(cursor == len(before.before_text), "Targeting dropped Before content")


@dataclass(frozen=True)
class AlignedOutcome(Record):
    prediction_unit_id: str
    after_text: str | None
    alignment_status: str


def align_outcomes(before_text, after_text, units):
    """Evaluation-only operation, called AFTER unit IDs/anchors have been fixed.

    Cross-unit edits and insertions at internal boundaries are unsupported, never
    fabricated as no-change cases or used to remove Later prediction targets.
    """
    ops = difflib.SequenceMatcher(None, before_text, after_text, autojunk=False).get_opcodes()
    rows = []
    internal = {u.start for u in units[1:]}
    for unit in units:
        pieces, status = [], "ALIGNED"
        if not before_text:
            rows.append(AlignedOutcome(unit.prediction_unit_id, None, "UNSUPPORTED_EMPTY_BEFORE"))
            continue
        for tag, i, j, k, l in ops:
            if tag == "equal":
                a, b = max(i, unit.start), min(j, unit.end)
                if a < b:
                    pieces.append(after_text[k + a - i:k + b - i])
            elif i == j:
                if i in internal and i in (unit.start, unit.end):
                    status = "UNSUPPORTED_BOUNDARY_INSERTION"
                elif (unit.start < i < unit.end or i == 0 == unit.start or
                      i == len(before_text) == unit.end):
                    pieces.append(after_text[k:l])
            elif max(i, unit.start) < min(j, unit.end):
                if unit.start <= i and j <= unit.end:
                    pieces.append(after_text[k:l])
                else:
                    status = "UNSUPPORTED_CROSS_UNIT"
        rows.append(AlignedOutcome(unit.prediction_unit_id,
                                   "".join(pieces) if status == "ALIGNED" else None, status))
    if all(row.alignment_status == "ALIGNED" for row in rows):
        require("".join(row.after_text for row in rows) == after_text,
                "Alignment did not account for all After content")
    return tuple(rows)
