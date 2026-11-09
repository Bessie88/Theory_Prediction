"""Explicit source-reviewed corrections for two archived logic conversions.

This is an offline correction utility, never an automatic validator relaxation.
The original source, model response, Action catalog and provenance stay separate.
"""
from copy import deepcopy
from .schemas import require

SOURCE_SHA256 = "c6bd715fc12be28f676140dc2937668065a28e021b145f8222b7f3e37ccb4d3f"
VERSION = "source-reviewed-boundary-corrections-v1"
CORRECTIONS = {
    "theory_2db80d478f6f898a31d7": {
        "clause_id": "measurement_clause_4fe58c17328f49b8b759",
        "pointer": "/branches/8/boundary_text",
        "source": "Applies to participial phrases obscuring agency, passive/active voice shifts for subjectivity, and aspect shifts for temporal/scope consistency.",
        "role": "REQUIREMENT",
        "truth_scope": "PROPOSED_OPERATION_SCOPE",
        "predicate": "The proposed operation concerns a participial phrase obscuring agency, a passive/active voice shift for subjectivity, or an aspect shift for temporal/scope consistency.",
        "rationale": "Retain the branch boundary as its own positive scope restriction. The source does not establish equivalence to all subordinate rules, so do not certify it as redundant.",
    },
    "theory_c4dc6a626251f8e2f5ab": {
        "clause_id": "measurement_clause_b571fb9cdbee473f714a",
        "pointer": "/branches/11/boundary_text",
        "source": "Applies to misspellings, date formats, noun number, quotation marks, and punctuation substitutions. Article deletion applies when specific contextual conditions (numerical/quantified details, causal/simulation context) are met. Excludes essential context or factual data.",
        "role": "EXCLUSION",
        "truth_scope": "PROPOSED_OPERATION_SCOPE",
        "predicate": "The proposed correction removes or changes essential contextual or factual information, rather than only its grammatical or typographic form.",
        "rationale": "Retain the explicit branch-wide exclusion as NOT(excluded operation). This describes the proposed operation's scope; the mere presence of facts in Before does not disqualify a formatting correction.",
    },
}

def _references(node):
    if node.get("op") == "clause":
        return {node["clause_id"]}
    children = node.get("args", [])
    if "arg" in node:
        children = [node["arg"]]
    return set().union(*(_references(n) for n in children)) if children else set()

def correct_archived_review(raw, payload):
    """Return an explicitly annotated amendment; never infer a general rule."""
    theory_id = payload["theory_id"]
    require(theory_id in CORRECTIONS, "No source-reviewed correction for this Theory")
    spec = CORRECTIONS[theory_id]
    require(payload.get("source_sha256") == SOURCE_SHA256, "Correction source hash changed")
    require(raw.get("theory_id") == theory_id and raw.get("actions_unchanged") is True,
            "Correction requires unchanged source/Action identity")
    sources = [c for c in payload["source_clauses"] if c["clause_id"] == spec["clause_id"]]
    require(len(sources) == 1 and sources[0]["source_kind"] == "BOUNDARY" and
            sources[0]["source_pointer"] == spec["pointer"], "Correction boundary identity changed")
    docs = {d["source_id"]: d["text"] for d in payload["source_documents"]}
    require(docs[sources[0]["source_id"]] == spec["source"], "Correction source text changed")
    require(spec["clause_id"] not in _references(raw["applicability"]),
            "Correction only repairs an omitted boundary; do not insert twice")
    out = deepcopy(raw)
    clauses = [c for c in out["clauses"] if c["clause_id"] == spec["clause_id"]]
    require(len(clauses) == 1, "Correction requires exactly one existing boundary disposition")
    target = clauses[0]
    require(target["role"] in ({"EXCLUSION"} if spec["role"] == "EXCLUSION"
                              else {"REDUNDANT_SCOPE", "EXCLUSION"}),
            "Unexpected archived boundary role")
    target.clear()
    target.update(clause_id=spec["clause_id"], role=spec["role"],
                  predicate_text=spec["predicate"], truth_scope=spec["truth_scope"])
    gate = {"op": "clause", "clause_id": spec["clause_id"]}
    if spec["role"] == "EXCLUSION":
        gate = {"op": "not", "arg": gate}
    # Only this reviewed branch-wide boundary is added; all existing AND/OR/NOT
    # relationships remain verbatim inside the first child.
    out["applicability"] = {"op": "and", "args": [out["applicability"], gate]}
    audit = {"correction_version": VERSION, "author_type": "CODEX_AGENT_SOURCE_REVIEW",
             "theory_id": theory_id, "clause_id": spec["clause_id"],
             "source_sha256": SOURCE_SHA256, "source_pointer": spec["pointer"],
             "source_text": spec["source"], "rationale": spec["rationale"],
             "not_a_model_regeneration": True, "production_approved": False}
    return out, audit
