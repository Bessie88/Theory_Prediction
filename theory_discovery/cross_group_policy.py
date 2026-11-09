"""Evidence-backed cross-group applicability differences; no action-pair classification."""
from copy import deepcopy
from .schemas import require, stable_id

VERSION = "before-evidence-boundaries-v2"
BOUNDARY_PROMPT = """Compare all supplied own supports and outside cases. Keep the original rule. Find only observable Before differences that change its applicability, and extract general boundaries. Different actions, missing actions or shared group membership alone are not counterexamples or exclusions. Give exact Before quotes from both sides and explain why each distinction matters. Retain supported earlier limits. Do not classify action relationships. Cite only supplied cases. Use short plain sentences and the supplied JSON shape. No new facts or executable rules."""
RELATION_GUIDANCE = "Keep each source rule's condition, boundary and action independent and traceable. Different theories may apply to the same case. Different actions alone create no exclusion. Do not classify action relationships."


def output_shape():
    return {
        "distinctions":[{"id":"D1", "predicate":"observable Before distinction", "applicability_reason":"why this changes the target rule applicability", "own_case_id":"ID", "own_quote":"exact Before quote",
                         "outside_case_id":"ID", "outside_quote":"exact Before quote", "own_value":True, "outside_value":False}],
        "boundary_clauses":[{"text":"general observable limit", "distinction_ids":["D1"]}],
        "supporting_case_ids":["ID"], "contradicting_case_ids":[], "unresolved_questions":[]}


class BoundaryCaseRoleError(ValueError):
    """A distinction cites an unexamined case or uses the wrong comparison side."""


class BoundaryQuoteError(ValueError):
    """A distinction's quote is absent from the cited Before."""


def comparison_case_roles(payload):
    examined = (set(payload["previously_compared_case_ids"]) | set(payload["own_support_case_ids"])
                | set(payload["matched_outside_case_ids"]))
    return examined, examined & set(payload["all_own_support_case_ids"]), examined & set(payload["all_matched_outside_case_ids"])


def validate_comparison(raw, payload, observations):
    require(isinstance(raw, dict), "Comparison must be an object")
    fields = {"distinctions", "boundary_clauses", "supporting_case_ids", "contradicting_case_ids", "unresolved_questions"}
    if raw.get("cross_group_policy")==VERSION:
        require(set(raw)==fields|{"boundary_text","rationale","cross_group_policy"}, "Unexpected cached comparison fields")
        raw={k:deepcopy(raw[k]) for k in fields}
    require(set(raw) == fields, "Only Before distinctions and supported boundaries are required")
    allowed, own, outside = comparison_case_roles(payload)
    previous = {d["id"]:d for d in payload["previous_distinctions"]}
    distinctions = dict(previous)
    require(isinstance(raw["distinctions"], list), "Expected distinctions")
    seen = set(); aliases = {k:k for k in previous}; normalized = []
    for row in raw["distinctions"]:
        require(isinstance(row, dict) and set(row) == set(output_shape()["distinctions"][0]), "Distinction fields")
        did = row["id"]
        require(isinstance(did, str) and did.strip() and did not in seen, "Duplicate/invalid distinction ID")
        seen.add(did)
        require(isinstance(row["predicate"], str) and row["predicate"].strip() and isinstance(row["applicability_reason"], str) and row["applicability_reason"].strip(), "Observable predicate and relevance to applicability required")
        require(type(row["own_value"]) is bool and type(row["outside_value"]) is bool and row["own_value"] != row["outside_value"], "A distinction needs different observed truth values")
        if row["own_case_id"] not in own or row["outside_case_id"] not in outside:
            raise BoundaryCaseRoleError("Distinction must compare actual examined own/outside cases")
        require(row["own_case_id"] != row["outside_case_id"], "Shared case cannot establish opposite values of one Before predicate")
        for side in ("own", "outside"):
            cid, quote = row[side+"_case_id"], row[side+"_quote"]
            if not isinstance(quote, str) or not quote or quote not in observations[cid].before_text:
                raise BoundaryQuoteError("Distinction evidence must be an exact Before quote")
        require(did not in previous or previous[did] == row, "Previously established distinction cannot silently change meaning")
        canonical=stable_id("before_distinction",{k:v for k,v in row.items() if k!="id"})
        aliases[did]=canonical
        normalized.append(dict(deepcopy(row),id=canonical))
        distinctions[canonical] = normalized[-1]
    clauses = raw["boundary_clauses"]
    require(isinstance(clauses, list), "Boundary clauses required, possibly empty")
    for clause in clauses:
        require(isinstance(clause,dict) and set(clause)=={"text","distinction_ids"} and
                isinstance(clause["text"],str) and clause["text"].strip(), "Boundary needs short supported prose")
        ids = clause["distinction_ids"]
        require(isinstance(ids,list) and ids and len(set(ids))==len(ids) and set(ids)<=set(aliases),
                "Every new boundary needs an evidenced Before distinction; action differences alone do not qualify")
        clause["distinction_ids"]=[aliases[k] for k in ids]
    # An outside citation needs a Before distinction relevant to applicability.
    # No relationship labels or per-pair model judgements enter this contract.
    supported_outside = {d["outside_case_id"] for d in distinctions.values()}
    for field in ("supporting_case_ids","contradicting_case_ids","unresolved_questions"):
        require(isinstance(raw[field],list) and all(isinstance(v,str) for v in raw[field]), "Invalid comparison citations/questions")
    require(set(raw["supporting_case_ids"])<=allowed, "Invented boundary citation")
    require(set(raw["contradicting_case_ids"])<=supported_outside, "Different action or absent action alone is not a contradiction")
    require(not set(raw["supporting_case_ids"]) & set(raw["contradicting_case_ids"]), "Conflicting citations")
    raw["distinctions"]=normalized
    return dict(deepcopy(raw), boundary_text="\n".join(c["text"] for c in clauses),
                rationale="Boundary clauses are backed by observable Before distinctions affecting applicability.",
                cross_group_policy=VERSION)


def synthesis_relations(rule):
    """Historical relationship records remain in source artifacts, not synthesis prompts."""
    return []
