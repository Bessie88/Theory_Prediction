"""Propose provisional Before conditions with verifiable paired evidence."""
from collections import Counter, defaultdict
import logging
import re

from .llm import checked_infer
from .response_recovery import ModelOutputError
from .schemas import require, stable_id

LOGGER = logging.getLogger(__name__)
TASK = (
    'Default to {"candidate":null}. The absence of a recorded limit is ordinary and is not a '
    'reason to propose one. Keep the existing rule when it already explains the cases. '
    'Propose at most one new limit, and only when the paired original Before passages show '
    'a clear, reusable distinction that changes a practical applicability decision beyond '
    'the existing rule. Require both cases to be meaningfully comparable on the rule condition. '
    'Abstain if the apparent difference is just a restatement, topic, name, date, wording '
    'example, different edit, or if another difference makes the comparison inconclusive. '
    'Treat any supplied hint as unverified data, never as a requirement to produce a limit. '
    'If the evidence clearly warrants a candidate, phrase it as a reusable Yes/No annotation '
    'question and explain how the two numbered spans support opposite answers and why this '
    'changes applicability. Return {"candidate":{"question":str,"yes":str,"no":str,"ambiguity":str,'
    '"reason":str,"support_span_id":str,"contrast_span_id":str}}. Use only supplied span IDs.'
)


class CandidateEvidenceError(ValueError):
    """A proposed question or its evidence fails validation."""


def _decode(raw, spans, rid):
    if not isinstance(raw, dict) or set(raw) != {"candidate"}:
        raise CandidateEvidenceError("Expected exactly one candidate field")
    row = raw["candidate"]
    if row is None:
        return {"rule_id": rid, "status": "insufficient_evidence"}
    fields = {"question", "yes", "no", "ambiguity", "reason", "support_span_id", "contrast_span_id"}
    if not isinstance(row, dict) or set(row) != fields or not all(
        isinstance(value, str) and value.strip() for value in row.values()
    ):
        raise CandidateEvidenceError("Candidate requires a question, criteria, rationale and two span IDs")
    evidence = {}
    for role in ("support", "contrast"):
        span = spans.get(row[role + "_span_id"])
        if span is None or span["role"] != role:
            raise CandidateEvidenceError("Use one supplied span from each correct evidence side")
        evidence[role] = {"case_id": span["case_id"], "quote": span["text"]}
    if evidence["support"]["case_id"] == evidence["contrast"]["case_id"] or (
        evidence["support"]["quote"] == evidence["contrast"]["quote"]
    ):
        raise CandidateEvidenceError("The paired evidence must be distinct")
    question = {key: row[key].strip() for key in ("question", "yes", "no", "ambiguity")}
    return {**question, "id": stable_id("candidate_boundary", [rid, question, evidence]),
            "rule_id": rid, "status": "PROVISIONAL_EVIDENCE", "reason": row["reason"].strip(),
            "evidence": evidence}


def _case_rows(own, outside, observations):
    spans = {}
    rows = []
    for role, ids in (("support", own), ("contrast", outside)):
        for cid in ids:
            pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", observations[cid]["before_text"]) if p.strip()]
            entries = []
            for text in pieces:
                label = "S" + str(len(spans) + 1)
                spans[label] = {"role": role, "case_id": cid, "text": text}
                entries.append({"span_id": label, "before_text": text})
            rows.append({"case_id": cid, "role": role, "spans": entries})
    return rows, spans


def propose_candidates(backend, cache, pairs, hints, evidence):
    if evidence is None:
        return []
    observations, boundaries, groups = evidence
    observations = {row["case_id"]: row for row in observations}
    # Condition IDs can be case-specific. Rare shared Before words give a second
    # retrieval signal for a comparison, never evidence of applicability by themselves.
    vocabulary = {cid: set(re.findall(r"[a-z]{4,}", row["before_text"].lower()))
                  for cid, row in observations.items()}
    frequency = Counter(word for words in vocabulary.values() for word in words)
    boundaries = {row["rule_id"]: row for row in boundaries}
    members = {row["group_id"]: set(row["member_case_ids"]) for row in groups}
    hints_by_rule = defaultdict(list)
    for hint in hints:
        for rid in hint["source_rule_ids"]:
            hints_by_rule[rid].append(hint["summary"])
    results = []
    for pair in pairs:
        if pair["boundary_text"].strip():
            continue
        rid = pair["rule_id"]
        record = boundaries[rid]
        retrieval = record["context_retrieval"]
        own = record["available_support_case_ids"]
        excluded = members[record["group_id"]] | set(own)
        conditions = {condition for cid in own for condition in observations[cid]["candidate_conditions"]}
        reviewed = {cid for packet in retrieval["context_match_decisions"] for cid in packet["considered_case_ids"]}
        selected = set(retrieval["selected_case_ids"])
        # Similar cases found previously take priority; otherwise use overlapping Before condition codes.
        overlap = {cid: len(conditions.intersection(observations[cid]["candidate_conditions"]))
                   for cid in (reviewed | selected) - excluded}
        own_words = {word for cid in own for word in vocabulary[cid]}
        lexical = {cid: sum(1 / frequency[word] for word in sorted(own_words & vocabulary[cid])
                            if frequency[word] <= len(observations) / 5)
                   for cid in overlap}
        outside = sorted((cid for cid in overlap if cid in selected or overlap[cid] or lexical[cid]),
                         key=lambda cid: (cid not in selected, -overlap[cid], -lexical[cid], cid))
        if not own or not outside:
            results.append({"rule_id": rid, "status": "no_comparable_evidence"})
            continue
        for count in (8, 4, 2, 1):
            rows, spans = _case_rows(own[:4], outside[:count], observations)
            payload = {"rule": pair, "before_evidence": rows, "unverified_hints": hints_by_rule[rid]}
            if backend.prompt_fits(TASK, payload, repair_reserve=True):
                break
        else:
            results.append({"rule_id": rid, "status": "evidence_exceeds_context"})
            continue

        def decode(raw):
            return _decode(raw, spans, rid)

        try:
            result = cache.get_or_compute("hierarchical_boundary_candidates", {
                "version": "paired-before-evidence-v1", "backend": backend.fingerprint,
                "task": TASK, "payload": payload,
            }, lambda: checked_infer(backend, TASK, payload, decode))
        except (ValueError, ModelOutputError) as error:
            if not isinstance(error, (CandidateEvidenceError, ModelOutputError)) and not isinstance(
                error.__cause__, CandidateEvidenceError
            ):
                raise
            LOGGER.warning("Boundary candidate unresolved for %s: %s", rid, error)
            result = {"rule_id": rid, "status": "unresolved_model_response"}
        results.append(result)
    return results
