"""Three-valued lazy execution and source-action proof gates for reviewed logic."""
from copy import deepcopy
from .schemas import require
from .theory_logic import evaluate_logic

VERSION = "reviewed-rule-first-v2"
BOOLEAN_LABEL_POLICY = "frozen-boolean-action-rules-v1"

def normalize(node, negative=False):
    """NNF is logically equivalent; source expressions are never overwritten."""
    op=node["op"]
    if op=="unknown":return {"op":"constant","value":"UNKNOWN"}
    if op=="constant":
        value=node["value"]
        return {"op":"constant","value":({"TRUE":"FALSE","FALSE":"TRUE"}.get(value,value) if negative else value)}
    if op=="not":return normalize(node["arg"],not negative)
    if op in ("and","or"):
        return {"op":({"and":"or","or":"and"}[op] if negative else op),
                "args":[normalize(c,negative) for c in node["args"]]}
    require(op=="clause","Only reviewed source clauses are executable")
    leaf={"op":"clause","clause_id":node["clause_id"]}
    return {"op":"not","arg":leaf} if negative else leaf

def refs(node):
    if node["op"]=="clause":return {node["clause_id"]}
    if node["op"]=="not":return refs(node["arg"])
    return set().union(*(refs(c) for c in node.get("args",[]))) if node.get("args") else set()

def value(node,values):
    return evaluate_logic(normalize(node),values)[0]

def lazy_value(node,reader,path="$"):
    op=node["op"];trace=[path]
    if op=="unknown":return None,trace
    if op=="constant":return {"TRUE":True,"FALSE":False,"UNKNOWN":None}[node["value"]],trace
    if op=="clause":return reader(node["clause_id"]),trace
    if op=="not":
        result,child=lazy_value(node["arg"],reader,path+".not")
        return (None if result is None else not result),trace+child
    require(op in ("and","or"),"Unsupported lazy operator")
    unknown=False
    for i,child in enumerate(node["args"]):
        result,visited=lazy_value(child,reader,path+"."+str(i));trace+=visited
        if result is (False if op=="and" else True):return result,trace
        unknown |= result is None
    return (None if unknown else op=="and"),trace

def next_clause(node,values,eligible):
    """One live unresolved leaf. Unknown leaves never count as false."""
    if value(node,values) is not None:return None
    op=node["op"]
    if op=="clause":
        cid=node["clause_id"]
        return cid if cid in eligible else None
    if op=="not":return next_clause(node["arg"],values,eligible)
    for child in node.get("args",[]):
        cid=next_clause(child,values,eligible)
        if cid is not None:return cid
    return None

def source_proof_gate(expression,clause_id):
    """Keep exactly proofs containing this original rule's condition literal.

    AND siblings remain mandatory. OR alternatives that do not cite this
    source rule cannot license its Action. This is proof selection, never
    a new default combination of source conditions.
    """
    tree=normalize(expression)
    require(clause_id in refs(tree),"Action source condition absent from reviewed logic")
    def restrict(node):
        if node["op"]=="or":
            selected=[restrict(c) for c in node["args"] if clause_id in refs(c)]
            require(selected,"Missing source proof")
            return selected[0] if len(selected)==1 else {"op":"or","args":selected}
        if node["op"]=="and":
            return {"op":"and","args":[restrict(c) if clause_id in refs(c) else deepcopy(c) for c in node["args"]]}
        return deepcopy(node)
    return restrict(tree)

def any_gate(gates):
    require(gates,"Cannot invent a source Action gate")
    return gates[0] if len(gates)==1 else {"op":"or","args":gates}

def comparison(expression,values):
    visited=[]
    def read(cid):visited.append(cid);return values.get(cid)
    lazy,paths=lazy_value(expression,read)
    full=value(expression,values)
    require(lazy is full,"Lazy/full source logic mismatch")
    return {"truth":lazy,"full_truth":full,"equal":True,
            "read_clause_ids":list(dict.fromkeys(visited)),
            "skipped_clause_ids":sorted(refs(expression)-set(visited)),
            "visited_paths":paths}

def compile_owner(theory,measurement,review,bundle):
    clauses={c.clause_id:c for c in measurement.clauses if c.theory_id==theory.theory_id}
    require(set(clauses)=={c["clause_id"] for c in review["clauses"]},"Review/source inventory mismatch")
    declarations={}
    for c in review["clauses"]:
        source=clauses[c["clause_id"]]
        declarations[c["clause_id"]]=dict(c,source_text=source.original_text,
            source_kind=source.role,source_pointer=source.source_pointer,source_rule_ids=list(source.source_rule_ids))
    action_map={a.record_id:a for a in bundle.source_actions}
    actions={}
    for option in theory.actions:
        for aid in option.action_ids:
            require(aid in action_map,"Missing frozen Action")
            actions.setdefault(aid,{"action_id":aid,"original_action":__import__("json").loads(action_map[aid].original_json),
                "variants":[]})
            for rule in option.source_rule_ids:
                if bundle.knowledge_source.value=="THEORY":
                    matches=[c for c in clauses.values() if c.role=="CONDITION" and
                             c.source_rule_ids==(rule,) and c.source_pointer.startswith("/rules/")]
                    require(len(matches)==1,"Source Action needs exactly its original rule condition; no guessed OR")
                    gate=source_proof_gate(review["applicability"],matches[0].clause_id)
                else:
                    require(rule in theory.source_rule_ids,"Foreign policy Action rule")
                    gate=normalize(review["applicability"])
                if rule not in [v["source_rule_id"] for v in actions[aid]["variants"]]:
                    actions[aid]["variants"].append({"source_rule_id":rule,"gate":gate})
    for a in actions.values():a["gate"]=any_gate([v["gate"] for v in a["variants"]])
    return {"owner_id":theory.theory_id,"method":bundle.knowledge_source.value,
            "source_sha256":bundle.native_source.sha256,"source_text":theory.theory_text,
            "source_rule_ids":list(theory.source_rule_ids),"expression":deepcopy(review["applicability"]),
            "clauses":declarations,"actions":list(actions.values()),
            "result_constraints":[c for c in declarations.values() if c["role"]=="RESULT_REQUIREMENT"]}
