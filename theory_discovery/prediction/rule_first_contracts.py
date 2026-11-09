"""Check hypothetical proposals against complete frozen source rules."""
from pathlib import Path
import json
from theory_discovery.core.schemas import require,stable_id
from theory_discovery.core.storage import write_json
from .rule_first_repairs import canonical_operation
CONTRACT_VERSION="complete-source-action-contract-v1"
PROMPT="""Check each HYPOTHETICAL proposal against its exact frozen source Action and COMPLETE original rule/boundary text.
Use only common Before, pre-edit metadata and supplied source text; no observed After or outside facts exist.
Check positive requirements as well as exclusions, exceptions and desired-result restrictions.
Do not discard obligations embedded after the action in the original rule: e.g. 'while preserving core factual meaning'.
If that source requires preserving meaning, check time, duration, quantity, attribution, identity and causal claims in the proposed result.
Do NOT impose universal meaning preservation: a source rule may explicitly license a quantifier, stance or factual change. Follow that rule's actual permission and limits.
Check that the target and proposed change implement the frozen Action method/type/scope, not a different operation.
Unsupported new facts or unavailable required context give null. A demonstrable source violation gives false.
The hypothetical result, if supplied, is derived from Before and the proposed parameters; it is NOT a human edit.
Return {"results":[{"id":"Q1","value":true,"rationale":"brief source-based reason"}]}.
True means the proposal satisfies all stated source requirements; false means a specific violation; null means insufficient information.
Do not score style, invent constraints, or infer which edit a person actually made."""

def source_contract(owner,action,variant):
    rule=variant["source_rule_id"]
    clauses=[{"source_pointer":c["source_pointer"],"source_kind":c["source_kind"],"text":c["source_text"]}
         for c in owner["clauses"].values() if rule in c["source_rule_ids"]]
    return {"source_sha256":owner["source_sha256"],"owner_id":owner["owner_id"],
        "source_rule_id":rule,"source_action_id":action["action_id"],
        "frozen_action":action["original_action"],"source_theory_or_policy_summary":owner["source_text"],
        "complete_source_clauses":clauses}

def hypothetical_result(case,proposal):
    target=proposal["target"];require(target["side"]=="before" and
        case.before_text[target["start"]:target["end"]]==target["quote"],"Contract target mismatch")
    replacement=proposal.get("replacement");family=proposal["frozen_action"].get("family")
    if family=="deletion" and replacement is None:replacement=""
    if isinstance(replacement,str) and family in ("replacement","deletion"):
        return case.before_text[:target["start"]]+replacement+case.before_text[target["end"]:]
    return None

class SourceContracts:
    def __init__(self,case,requests,directory,freeze_hash):
        self.case=case;self.requests=requests
        self.path=Path(directory)/(case.identity.case_id+".json")
        self.identity=stable_id("source_contract_case",[case.to_dict(),freeze_hash,CONTRACT_VERSION,PROMPT])
        self.saved={}
        if self.path.exists():
            data=json.loads(self.path.read_text());require(data["identity"]==self.identity,"Contract cache source changed")
            self.saved=data["checks"]
    def key(self,owner,action,variant,proposal):
        return stable_id("source_contract_check",[self.case.before_hash,source_contract(owner,action,variant),
             canonical_operation(proposal),CONTRACT_VERSION])
    def check(self,proposals):
        rows=[];keys=[]
        for owner,action,variant,proposal in proposals:
            key=self.key(owner,action,variant,proposal);keys.append(key)
            if key in self.saved:continue
            result=hypothetical_result(self.case,proposal)
            if result==self.case.before_text:
                self.saved[key]={"value":False,"rationale":"The proposed parameters make no text change.",
                   "validator":"DETERMINISTIC_NO_CHANGE","contract_version":CONTRACT_VERSION};continue
            rows.append({"key":key,"source_contract":source_contract(owner,action,variant),
                "hypothetical_proposal":proposal,"hypothetical_result_text":result})
        def parse(item,row):
            value=item.get("value")
            require(value is None or type(value) is bool,"Contract judgement must be Boolean/null")
            require(isinstance(item.get("rationale"),str) and item["rationale"].strip(),"Contract check needs source-based reason")
            return {"value":value,"rationale":item["rationale"],"validator":"MODEL_COMPLETE_SOURCE_CONTRACT",
                "contract_version":CONTRACT_VERSION,"source_contract_hash":stable_id("source_contract",row["source_contract"])}
        if rows:
            found,errors=self.requests.rows("COMPLETE_SOURCE_OPERATION_CONTRACT",PROMPT,rows,
                {"before_text":self.case.before_text,"pre_context":[f.to_dict() for f in self.case.pre_context]},parse)
            for row in rows:
                self.saved[row["key"]]=found.get(row["key"],{"value":None,"rationale":"Invalid/missing contract check; no approval inferred",
                     "contract_version":CONTRACT_VERSION,"errors":errors})
        write_json(self.path,{"identity":self.identity,"checks":self.saved})
        return [self.saved[k] for k in keys]
