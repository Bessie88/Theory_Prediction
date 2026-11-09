"""Outcome-only projection into a strict annotation view with allowlisted context."""
from .annotation_contracts import AnnotationCase, AnnotationCases
from .text_diff import extract_diff
from theory_discovery.core.workflow_contracts import require_same_cases
from theory_discovery.core.schemas import require, stable_id


def annotation_case_hash(identity, before, after, context, availability):
    return stable_id("annotation_input", [identity.to_dict(), before, after,
                     [v.to_dict() for v in context], availability])


def prepare_annotation_cases(outcomes, later, context_allowlist=(), max_characters=200000):
    require_same_cases(outcomes.records, later.records)
    safe = {c.identity.case_id: c for c in later.records}
    cases = []
    for row in outcomes.records:
        before = safe[row.identity.case_id]
        require(before.before_text == row.before_text, "Prediction/annotation Before texts differ")
        # Never forward outcome metadata, source filenames, prediction paths or a global config.
        context = tuple(v for v in before.pre_context if v.name in context_allowlist)
        availability, limitations, diff = "AVAILABLE", (), None
        if row.after_text is None:
            availability, limitations = "UNAVAILABLE", ("Actual After unavailable for this prediction unit.", row.alignment_status)
        elif len(row.before_text) + len(row.after_text) > max_characters:
            availability, limitations = "TOO_LARGE", ("Exact pair exceeds configured diff/annotation budget; no truncation.",)
        else:
            diff = extract_diff(row.before_text, row.after_text)
        key = annotation_case_hash(row.identity, row.before_text, row.after_text, context, availability)
        cases.append(AnnotationCase(row.identity, row.before_text, row.after_text, diff,
                                   context, key, availability, limitations))
    return AnnotationCases(tuple(cases))
