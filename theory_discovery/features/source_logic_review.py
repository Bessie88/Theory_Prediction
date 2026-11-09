"""Source-only joint Condition/Boundary logic review; Action remains separate."""
import json
from copy import deepcopy
from hashlib import sha256
from theory_discovery.core.schemas import require,stable_id

VERSION="source-logic-review-v2"  # Source payload identity: unchanged to reuse valid checkpoints.
TRANSLATION_VERSION="source-logic-review-v3"
PROMPT="""Translate the original Theory's CLEAR MEANING directly into a joint Condition/Boundary applicability expression.
The Theory need not literally contain AND, OR or NOT. Natural-language implications, alternatives, negation and exceptions may be translated semantically.
Do not require a verbatim quote or citation for an operator or clause. No evidence/quotes should be returned.
Do not choose a blanket AND/OR convention: preserve this Theory's actual meaning.
Use unknown ONLY for genuine semantic ambiguity or missing relationships, NEVER just for absence of explicit connective words.
Actions remain in a SEPARATE read-only catalog. Do not turn an editing operation or desired result into a Before prerequisite.
Account for every registered clause, separating the predicate from its Action:
REQUIREMENT, EXCLUSION, EXCEPTION, SUPPORTING, RESULT_REQUIREMENT, REDUNDANT_SCOPE, or UNKNOWN.
Use REQUIREMENT (not the input source_kind CONDITION) as the output role for a necessary predicate.
Every REQUIREMENT/EXCLUSION/EXCEPTION must occur in the AST, including general branch scope and boundaries.
Do not omit a necessary restriction or negate an already positive applicability requirement merely because its source_kind is BOUNDARY.
If a general scope is fully represented by other AST clauses, use REDUNDANT_SCOPE with covered_by:[those clause IDs] and a brief reason; do not discard any independent exclusion.
For genuinely unclear clauses, use UNKNOWN and an unknown node with clause_ids:[affected IDs]. An entirely unknown root may cover all clauses.
RESULT truth_scope requires RESULT_REQUIREMENT role and must stay outside the applicability AST. Proposed-operation scope is distinct from observed Before facts.
Supporting evidence is not a hard gate; output requirements are not prerequisites already satisfied by Before.
Return only:
{"theory_id":"supplied ID","clauses":[
 {"clause_id":"ID","role":"one of the roles","predicate_text":"brief predicate without its Action",
  "truth_scope":"BEFORE|PRE_EDIT_METADATA|PROPOSED_OPERATION_SCOPE|RESULT|UNRESOLVED"}],
 "applicability":node,"uncertainties":["only genuine semantic ambiguities"],"actions_unchanged":true}
Nodes: {"op":"clause","clause_id":"ID"}, {"op":"and" or "or","args":[nodes]},
{"op":"not","arg":node}, {"op":"unknown","reason":"specific unclear meaning"}.
Reference only supplied clause IDs; no new conditions, Actions, weights or thresholds.
Be compact: short predicate descriptions, no repeated source paragraphs, quotes, evidence, explanations or step-by-step analysis.
Do not read cases, predict edits, or infer test outcomes."""


ROLES={"REQUIREMENT","EXCLUSION","EXCEPTION","SUPPORTING","RESULT_REQUIREMENT","REDUNDANT_SCOPE","UNKNOWN"}

def payload_for(theory,bundle,measurement):
    documents={};clauses=[]
    def document(text):
        identity=stable_id("logic_source",text)
        documents.setdefault(identity,{"source_id":identity,"text":text})
        return identity
    document(theory.theory_text)
    for c in measurement.clauses:
        if c.theory_id!=theory.theory_id:continue
        clauses.append({"clause_id":c.clause_id,"source_kind":"CONDITION" if c.role=="CONDITION" else "BOUNDARY",
                        "source_pointer":c.source_pointer,"source_rule_ids":list(c.source_rule_ids),"source_id":document(c.original_text)})
    native=json.loads(theory.original_representation_json)
    for key in ("situation","interpretation","boundary_text"):
        if isinstance(native.get(key),str) and native[key].strip():document(native[key])
    action_ids={k for option in theory.actions for k in option.action_ids}
    actions=[{"action_id":a.record_id,
              "source_rule_ids":sorted({key for option in theory.actions if a.record_id in option.action_ids for key in option.source_rule_ids}),
              "original_action":json.loads(a.original_json)}
             for a in bundle.source_actions if a.record_id in action_ids]
    return {"review_version":VERSION,"theory_id":theory.theory_id,"native_branch_id":theory.native_branch_id,
            "source_sha256":bundle.native_source.sha256,"source_documents":list(documents.values()),
            "source_clauses":clauses,"actions_read_only":actions,
            "contract":"Joint Condition/Boundary source logic; Action separate; no old application or Before/After cases"}

def normalize_review(raw,payload):
    """Repair unambiguous wire presentation only; never add gates or infer truth."""
    raw=deepcopy(raw); repairs=[]
    if not isinstance(raw,dict):return raw,repairs
    root=raw.get("applicability")
    if isinstance(root,dict):
        for field in ("actions_unchanged","uncertainties"):
            if field not in root:continue
            require(field not in raw or raw[field]==root[field],
                    "Conflicting top-level/root "+field+"; cannot normalize")
            raw[field]=root.pop(field);repairs.append("hoisted_root_"+field)
    source={c["clause_id"]:c for c in payload["source_clauses"]}
    if isinstance(raw.get("clauses"),list):
        for clause in raw["clauses"]:
            if not isinstance(clause,dict):continue
            if clause.get("role")=="CONDITION":
                cid=clause.get("clause_id")
                require(cid in source and source[cid].get("source_kind")=="CONDITION",
                        "CONDITION role alias requires a supplied CONDITION clause: "+str(cid))
                clause["role"]="REQUIREMENT";repairs.append("condition_role_alias:"+cid)
    return raw,repairs

def validate_review(raw,payload):
    raw,repairs=normalize_review(raw,payload)
    fields={"theory_id","clauses","applicability","uncertainties","actions_unchanged"}
    require(isinstance(raw,dict),"Review must be a JSON object")
    require(set(raw)==fields,
            "Review top-level fields mismatch; missing="+str(sorted(fields-set(raw)))+
            "; extra="+str(sorted(set(raw)-fields))+"; Actions must remain separate")
    require(raw["theory_id"]==payload["theory_id"] and raw["actions_unchanged"] is True,"Source/action identity changed")
    allowed={r["clause_id"] for r in payload["source_clauses"]}
    require(isinstance(raw["clauses"],list),"Clause dispositions required")
    dispositions={}
    for c in raw["clauses"]:
        require(isinstance(c,dict),"Clause disposition must be an object")
        cid=c.get("clause_id")
        require(cid in allowed,"Foreign clause_id: "+str(cid))
        require(cid not in dispositions,"Duplicate clause_id: "+str(cid))
        require({"clause_id","role","predicate_text","truth_scope"} <= set(c) <=
                {"clause_id","role","predicate_text","truth_scope","evidence","reason","covered_by"},
                "Invalid disposition fields for "+cid+": "+str(sorted(c)))
        require(c["role"] in ROLES,"Invalid role for "+cid+": "+str(c["role"])+"; use "+str(sorted(ROLES)))
        require(isinstance(c["predicate_text"],str) and c["predicate_text"].strip() and
                c["truth_scope"] in {"BEFORE","PRE_EDIT_METADATA","PROPOSED_OPERATION_SCOPE","RESULT","UNRESOLVED"},
                "Predicate must be explicit and distinct from Action: "+cid)
        if c["truth_scope"]=="RESULT":
            require(c["role"]=="RESULT_REQUIREMENT","RESULT predicate must use RESULT_REQUIREMENT and stay outside applicability: "+cid)
        if c["role"]=="REDUNDANT_SCOPE":
            require(isinstance(c.get("covered_by"),list) and c["covered_by"] and
                    all(isinstance(k,str) and k in allowed and k!=cid for k in c["covered_by"]) and
                    isinstance(c.get("reason"),str) and c["reason"].strip(),
                    "REDUNDANT_SCOPE needs existing covered_by IDs and a semantic reason: "+cid)
        else:require("covered_by" not in c,"covered_by only allowed for REDUNDANT_SCOPE: "+cid)
        dispositions[cid]=c
    require(set(dispositions)==allowed,"Every Condition and Boundary must be accounted for; missing="+str(sorted(allowed-set(dispositions))))
    unknowns=[];unknown_refs=set()
    def visit(node,depth=0):
        require(isinstance(node,dict) and depth<=32,"Invalid logic AST")
        op=node.get("op")
        if op=="unknown":
            require({"op","reason"} <= set(node) <= {"op","reason","evidence","clause_ids"} and
                    isinstance(node["reason"],str) and node["reason"].strip(),"Unknown needs an explicit reason")
            covered=node.get("clause_ids",list(allowed) if depth==0 else [])
            require(isinstance(covered,list) and all(isinstance(k,str) and k in allowed for k in covered),
                    "Unknown clause_ids must refer to supplied clauses")
            unknown_refs.update(covered);unknowns.append(node["reason"]);return set()
        if op=="clause":
            cid=node.get("clause_id")
            require({"op","clause_id"} <= set(node) <= {"op","clause_id","evidence"} and cid in allowed,
                    "Action/foreign ID in applicability: "+str(cid))
            require(dispositions[cid]["role"] in {"REQUIREMENT","EXCLUSION","EXCEPTION"},
                    "Supporting/result/unknown/redundant clause cannot become a hard gate: "+cid)
            require(dispositions[cid]["truth_scope"] in {"BEFORE","PRE_EDIT_METADATA","PROPOSED_OPERATION_SCOPE"},
                    "Result/undefined predicate cannot become an applicability condition: "+cid)
            return {cid}
        if op in ("and","or"):
            require({"op","args"} <= set(node) <= {"op","args","evidence"} and
                    isinstance(node["args"],list) and len(node["args"])>=2,
                    "AND/OR needs at least two operands and only op/args fields")
            return set().union(*(visit(n,depth+1) for n in node["args"]))
        require(op=="not" and {"op","arg"} <= set(node) <= {"op","arg","evidence"},
                "Unsupported source operator/fields; no weights/defaults")
        return visit(node["arg"],depth+1)
    refs=visit(raw["applicability"])
    hard={k for k,c in dispositions.items() if c["role"] in {"REQUIREMENT","EXCLUSION","EXCEPTION","UNKNOWN"}}
    missing=hard-refs-unknown_refs
    require(not missing,"Unaccounted necessary clauses: "+str(sorted(missing))+
            ". Include their meaning in the AST; if a scope is fully represented, explicitly use REDUNDANT_SCOPE with covered_by and reason. Never omit an independent exclusion.")
    for cid,c in dispositions.items():
        if c["role"]=="REDUNDANT_SCOPE":
            require(set(c["covered_by"])<=refs,
                    "REDUNDANT_SCOPE must point directly to hard clauses actually present in AST: "+cid)
    require(isinstance(raw["uncertainties"],list) and all(isinstance(x,str) for x in raw["uncertainties"]),
            "Uncertainties must be explicit")
    require(not unknowns or raw["uncertainties"],"Unknown AST must retain uncertainty")
    result=dict(raw,review_status="UNRESOLVED" if unknowns or any(c["role"]=="UNKNOWN" for c in dispositions.values()) else "SEMANTIC_TRANSLATION",
                unknown_relations=unknowns,production_approved=False)
    if repairs:result["format_recoveries"]=repairs
    return result
