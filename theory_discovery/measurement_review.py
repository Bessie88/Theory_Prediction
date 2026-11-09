"""EARLY-only review packets and issue recording, not feature labeling."""
import json
from dataclasses import dataclass
from .workflow_contracts import Record, PredictionCase, Period, Method
from .measurement_contracts import MeasurementReview
from .schemas import require


@dataclass(frozen=True)
class ReviewExample(Record):
    case: PredictionCase
    evidence_role: str


def review_examples(source, theories, early_before):
    cases = {c.identity.case_id: c for c in early_before.records}
    require(len(cases) == len(early_before.records) and
            all(c.identity.period is Period.EARLY and c.identity.editor == theories.editor
                for c in cases.values()), "Measurement review is EARLY/editor-specific")
    if theories.knowledge_source is Method.NPOV:
        return tuple(ReviewExample(case, "EARLY_CONTEXT_ONLY") for case in early_before.records)
    aliases = {a.native_case_id: a.canonical_case_id for a in theories.case_aliases}
    require(set(source.source_rule_ids) <= {r.record_id for r in theories.source_rules},
            "Source rules needed for measurement review are missing")
    groups = {"SUPPORTING": set(), "CONTRASTING": set(), "AMBIGUOUS": set(), "UNCLASSIFIED_CONSTRUCTION": set()}
    for representation in theories.source_rules:
        if representation.record_id not in source.source_rule_ids:
            continue
        row = json.loads(representation.original_json)
        stats = row.get("evidence_statistics", {})
        groups["SUPPORTING"].update(row.get("supporting_case_ids", []))
        groups["CONTRASTING"].update(row.get("contradicting_case_ids", []))
        groups["AMBIGUOUS"].update(stats.get("contested_case_ids", []))
        groups["UNCLASSIFIED_CONSTRUCTION"].update(stats.get("unclassified_case_ids", []))
    examples = []
    for role, ids in groups.items():
        for native_id in sorted(ids):
            require(native_id in aliases and aliases[native_id] in cases,
                    "Review evidence missing canonical EARLY identity")
            examples.append(ReviewExample(cases[aliases[native_id]], role))
    return tuple(examples)


def pending_review(source, examples, binding):
    def ids(role):
        return tuple(sorted({e.case.identity.case_id for e in examples if e.evidence_role == role}))
    return MeasurementReview(source.clause_id, ids("SUPPORTING"), ids("CONTRASTING"),
        ids("AMBIGUOUS"), (),
        binding.limitations if binding.resolution == "NON_OBSERVABLE" else (), "PENDING",
        ids("UNCLASSIFIED_CONSTRUCTION"), ids("EARLY_CONTEXT_ONLY"))


def record_review(previous, inconsistencies=(), non_observable_reasons=(), reviewed=False):
    from dataclasses import replace
    return replace(previous, inconsistencies=tuple(inconsistencies),
                   non_observable_reasons=tuple(non_observable_reasons),
                   status="REVIEWED" if reviewed else "PENDING")
