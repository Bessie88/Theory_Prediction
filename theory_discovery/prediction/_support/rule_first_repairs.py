"""Explicit v2 measurement fixes, reuse checks and proposal grounding."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import json
from theory_discovery.core.schemas import require,stable_id
from theory_discovery.features._support.measurement_contracts import validate_specification,specification_hash,procedure_for
from theory_discovery.features._support.feature_spec_registry import feature_id_for
from theory_discovery.core.workflow_contracts import MeasurementMethod,FeatureObservation
from .theory_applicability import index_case_features

DATE_FEATURE="feature_ac1e3f00a1741c8f0e08"
PRIOR_VERSION="reviewed-rule-first-v1"

def correct_evidence_contract(specs):
    """Change one contradictory evidence contract; preserve meaning/program."""
    specs=dict(specs);aliases={};changes=[]
    if DATE_FEATURE not in specs:return specs,aliases,changes
    old=specs.pop(DATE_FEATURE)
    require(old.feature_id==feature_id_for(old) and old.measurement_method is MeasurementMethod.CODE
        and old.evidence_requirements==("SPANS",) and old.value_type=="boolean"
        and procedure_for(old).objective_tool=="regex_present","Unexpected date contract; do not generalize repair")
    new=replace(old,evidence_requirements=("SPANS_WHEN_TRUE","FULL_BEFORE"),version="1.1-negative-evidence")
    new=replace(new,feature_id=feature_id_for(new));validate_specification(new)
    require(new.extraction_procedure_json==old.extraction_procedure_json,"Date regex/program changed")
    specs[new.feature_id]=new;aliases[old.feature_id]=new.feature_id
    changes.append({"old_feature_id":old.feature_id,"new_feature_id":new.feature_id,
       "old_specification_hash":specification_hash(old),"new_specification_hash":specification_hash(new),
       "change":"Require spans when true; allow full-Before evidence for exhaustive false. Meaning and regex unchanged."})
    return specs,aliases,changes

def remap_expression(node,aliases):
    node=deepcopy(node)
    if "feature_id" in node:node["feature_id"]=aliases.get(node["feature_id"],node["feature_id"])
    if "feature_ids" in node:node["feature_ids"]=[aliases.get(fid,fid) for fid in node["feature_ids"]]
    if "arg" in node:node["arg"]=remap_expression(node["arg"],aliases)
    if "args" in node:node["args"]=[remap_expression(c,aliases) for c in node["args"]]
    return node

def upgrade_bridge(prior,owners,aliases,changes,version):
    mappings=deepcopy(prior["mappings"])
    current={cid:c for owner in owners for cid,c in owner["clauses"].items()}
    require(set(current)==set(mappings),"Bridge source inventory changed")
    for cid,m in mappings.items():
        require(m["predicate"]==current[cid],"Reviewed source predicate changed")
        if "expression" in m:m["expression"]=remap_expression(m["expression"],aliases)
        if "feature_ids" in m:m["feature_ids"]=[aliases.get(f,f) for f in m["feature_ids"]]
    result={k:deepcopy(v) for k,v in prior.items() if k not in ("freeze_hash","protocol","mappings")}
    result.update(protocol=version,mappings=mappings,reused_source_bridge_hash=prior["freeze_hash"],
       evidence_contract_corrections=changes,
       unknown_recheck_contract={"trigger":"A decision-live BEFORE/PRE_EDIT_METADATA predicate remains Unknown after its original components.",
         "input":"Full original Before and permitted metadata, never generator no-target prose or observed After.",
         "truth":"Independent Boolean judgement of the same reviewed predicate; absent candidate lists do not establish false.",
         "limit":"One bounded recheck measurement per predicate/context; missing context remains Unknown."})
    result["freeze_hash"]=stable_id("reviewed_predicate_bridge",result);return result

def canonical_operation(operation):
    if operation is None:return None
    action={k:v for k,v in operation["frozen_action"].items() if k!="group_id"}
    return {"frozen_action":action,"target":operation["target"],"replacement":operation.get("replacement"),
            "operation":operation.get("operation") if operation.get("replacement") is None else None}

def ground_target(text,quote,start=None,context_quote=None,occurrence=None):
    require(isinstance(quote,str) and quote,"Proposal needs target quote")
    if start is not None:
        require(type(start) is int and start>=0 and text[start:start+len(quote)]==quote,"Target offset mismatch")
        return {"side":"before","start":start,"end":start+len(quote),"quote":quote}
    positions=[];at=text.find(quote)
    while at>=0:positions.append(at);at=text.find(quote,at+1)
    require(positions,"Target absent from Before")
    if len(positions)==1:at=positions[0]
    elif context_quote:
        require(isinstance(context_quote,str) and context_quote in text
                and text.find(context_quote,text.find(context_quote)+1)<0,"Target context must have unique exact match")
        local=context_quote.find(quote)
        require(local>=0 and context_quote.find(quote,local+1)<0,"Target ambiguous inside context")
        at=text.find(context_quote)+local
    else:
        require(type(occurrence) is int and 1<=occurrence<=len(positions),"Repeated target needs context or occurrence")
        at=positions[occurrence-1]
    return {"side":"before","start":at,"end":at+len(quote),"quote":quote}

PROPOSAL_FIELDS=("status","source_action_id","owner_id","method","source_rule_id","frozen_action","target",
    "replacement","operation","reason","is_hypothetical","result_constraints","result_constraints_status")

def import_prior_measurements(measurements,prior_directory,prior_bridge_hash,prior_freeze_hash,proposal_index_file=None):
    """Reuse unchanged primitive observations; never import final decisions."""
    d=Path(prior_directory);p=d/"predictions/measurements"/(measurements.case.identity.case_id+".json")
    if not p.exists():return {"observations":0,"direct":0,"proposals":0}
    saved=json.loads(p.read_text())
    expected=stable_id("lazy_measurements",[measurements.case.to_dict(),prior_bridge_hash,prior_freeze_hash,PRIOR_VERSION])
    require(saved["identity"]==expected,"Prior measurement/case lineage mismatch")
    observed={}
    for fid,record in saved["observed"].items():
        if fid not in measurements.specs:continue
        observation=FeatureObservation.from_dict(record)
        require(observation.specification_hash==specification_hash(measurements.specs[fid]),"Reuse changes feature meaning")
        meta=dict((m.key,m.value) for m in observation.extraction_metadata)
        require(meta.get("actual_measurement_protocol",PRIOR_VERSION)==PRIOR_VERSION,"Prior observation producer changed")
        reason=meta.get("rationale","")
        # A saved Unknown is valid uncertainty, even when an unchanged
        # original procedure failed. Retain it and let the independent
        # predicate recheck close the gap; do not rerun the same failed tool.
        observed[fid]=observation
    valid,witnesses=index_case_features(measurements.case,tuple(observed.values()),measurements.specs)
    require(all(w.valid for w in witnesses.values()),"Prior observation evidence invalid")
    measurements.observed.update(valid);measurements.attempted_features.update(valid)
    direct=0
    allowed_before_keys={measurements.direct_key(cid) for cid,m in measurements.bridge.items()
        if m["predicate"]["truth_scope"] in ("BEFORE","PRE_EDIT_METADATA")}
    for key,record in saved["direct"].items():
        if key not in allowed_before_keys:continue
        if record["value"] is None and any(word in record.get("rationale","") for word in ("Invalid/missing","Unresolved batch row")):continue
        require(record["value"] is None or type(record["value"]) is bool,"Prior predicate type invalid")
        for span in record.get("evidence",[]):
            require(span["side"]=="before" and measurements.case.before_text[span["start"]:span["end"]]==span["quote"],
                    "Prior predicate evidence mismatch")
        # V2 operation keys have a new namespace. Only identical Before keys
        # can be reached; opaque prior operation records cannot override v2.
        measurements.direct[key]=deepcopy(record);measurements.attempted_direct.add(key);direct+=1
        if record["value"] is None:
            measurements.rechecks[key]=dict(deepcopy(record),measurement_origin="REUSED_FULL_BEFORE_UNCERTAINTY")
            measurements.attempted_rechecks.add(key)
    proposals={};unknown_proposals={};scope_reused=0
    source=d/"predictions/cases"/(measurements.case.identity.case_id+".json")
    if source.exists():
        prior=json.loads(source.read_text())
        require(prior["before_hash"]==measurements.case.before_hash and prior["source_logic_freeze_hash"]==prior_freeze_hash,
                "Prior proposal source/case changed")
        measurements.prior_clause_values={t["owner_id"]:t["clause_values"] for t in prior["owner_traces"]}
        if proposal_index_file is not None:
            indexed=json.loads(Path(proposal_index_file).read_text())
            require(indexed["prior_logic_freeze_hash"]==prior_freeze_hash,"Indexed proposal source mismatch")
            require(indexed["index_hash"]==stable_id("prior_proposal_index",{k:v for k,v in indexed.items() if k!="index_hash"}),"Proposal index integrity mismatch")
            recovered=[dict(row["proposal"],_old_gate_truth=row["old_gate_truth"],_cache_source=row["source_cache"])
                for row in indexed["cases"].get(measurements.case.identity.case_id,[])]
            prior["predictions"]["RECOVERED_ORIGINAL_RESPONSES"]={"supported_actions":recovered,"unresolved_operations":[]}
        checked_hashes={t.get("operation_hash") for t in prior["lazy_full_comparisons"] if t.get("operation_hash")}
        for prediction in prior["predictions"].values():
            for answer in prediction["unresolved_operations"]:
                if answer.get("status")=="PROPOSED" or not all(k in answer for k in ("owner_id","source_action_id","source_rule_id","reason")):continue
                if "Invalid/missing" in answer["reason"]:continue
                unknown_proposals[(answer["owner_id"],answer["source_action_id"],answer["source_rule_id"])]=deepcopy(answer)
            for row in prediction["supported_actions"]+prediction["unresolved_operations"]:
                if row.get("status")!="PROPOSED" or not all(k in row for k in PROPOSAL_FIELDS):continue
                candidate={k:deepcopy(row[k]) for k in PROPOSAL_FIELDS}
                # An exact archived proposal fingerprint proves which old scope
                # observations belong to this hypothetical operation.
                if stable_id("operation",candidate) in checked_hashes:
                    for cid,m in measurements.bridge.items():
                        pred=m["predicate"]
                        if pred["truth_scope"]!="PROPOSED_OPERATION_SCOPE":continue
                        oldkey=stable_id("reviewed_predicate_measurement",{"predicate":pred["predicate_text"],
                            "scope":pred["truth_scope"],"pre_context":[f.to_dict() for f in measurements.case.pre_context],
                            "operation":candidate})
                        record=saved["direct"].get(oldkey)
                        if record is None:continue
                        if record["value"] is None and any(word in record.get("rationale","") for word in ("Invalid/missing","Unresolved batch row")):continue
                        require(record["value"] is None or type(record["value"]) is bool,"Prior operation scope type invalid")
                        for span in record.get("evidence",[]):
                            require(span["side"]=="before" and measurements.case.before_text[span["start"]:span["end"]]==span["quote"],
                                    "Prior operation scope evidence mismatch")
                        newkey=measurements.direct_key(cid,candidate)
                        if newkey not in measurements.direct:
                            measurements.direct[newkey]=dict(deepcopy(record),reuse_origin={"prior_file":str(p),"prior_key":oldkey})
                            measurements.attempted_direct.add(newkey);scope_reused+=1
                if row.get("_old_gate_truth") is False:continue
                candidate["proposal_origin"]={"type":"PRIOR_UNVALIDATED_CANDIDATE","prior_case":str(source),"protocol":PRIOR_VERSION,
                    "original_response_cache":row.get("_cache_source")}
                span=candidate["target"]
                require(span["side"]=="before" and measurements.case.before_text[span["start"]:span["end"]]==span["quote"],
                        "Prior candidate target invalid")
                proposals[(candidate["owner_id"],candidate["source_action_id"],candidate["source_rule_id"])]=candidate
    measurements.reusable_candidates=proposals
    measurements.prior_unknown_candidates=unknown_proposals
    measurements.reuse_audit={"prior_directory":str(d),"observations":len(valid),"direct":direct,
       "proposals":len(proposals),"operation_scope_values":scope_reused,"prior_unresolved_answers":len(unknown_proposals),"final_decisions_reused":False,"changed_feature_not_reused":DATE_FEATURE}
    measurements.save();return measurements.reuse_audit

def build_prior_proposal_index(prior_directory):
    """Recover exact accepted hypotheses from original response caches, offline."""
    d=Path(prior_directory)
    from hashlib import sha256
    from theory_discovery.core.workflow_contracts import PredictionCase
    manifest=json.loads((d/"source_logic_freeze.json").read_text())
    raw=json.loads((d/"later_before.json").read_text())["records"][:20]
    cases={c.before_hash:c for c in [PredictionCase.from_dict(row) for row in raw]}
    saved_cases={};accepted={}
    for case in cases.values():
        p=d/"predictions/cases"/(case.identity.case_id+".json")
        if not p.exists():continue
        saved=json.loads(p.read_text())
        require(saved["before_hash"]==case.before_hash and saved["source_logic_freeze_hash"]==manifest["freeze_hash"],
                "Prior index case/source mismatch")
        saved_cases[case.identity.case_id]=saved
        accepted[case.identity.case_id]={t["operation_hash"]:t.get("truth") for t in saved["lazy_full_comparisons"] if t.get("operation_hash")}
    lookup={}
    for owner in manifest["owners"]:
        for action in owner["actions"]:
            for variant in action["variants"]:
                lookup.setdefault(stable_id("legacy_query",[action["original_action"],variant["source_rule_id"]]),[]).append((owner,action,variant))
    index={"prior_logic_freeze_hash":manifest["freeze_hash"],"cases":{},"source_cache_files":{},"model_calls":0}
    for path in (d/"cache/model_transport/llm_responses").glob("*.json"):
        envelope=json.loads(path.read_text());request=envelope["request"]["inputs"]
        messages=request.get("messages",[])
        if not messages or "Propose ONLY" not in messages[0].get("content",""):continue
        require(envelope["key"]==stable_id("cache",envelope["request"]) and
             envelope["value_hash"]==stable_id("value",envelope["value"]),"Prior response-cache integrity mismatch")
        payload=json.loads(messages[1]["content"])
        case=cases.get(sha256(payload["before_text"].encode()).hexdigest())
        if case is None or case.identity.case_id not in accepted:continue
        require(payload.get("pre_context",[])==[f.to_dict() for f in case.pre_context],"Prior cached input context changed")
        queries={row["id"]:row for row in payload["queries"]}
        found=False
        for item in envelope["value"]["result"].get("results",[]):
            if not isinstance(item,dict) or item.get("status")!="PROPOSED" or item.get("id") not in queries:continue
            q=queries[item["id"]];matches=lookup.get(stable_id("legacy_query",[q["frozen_action"],q["source_rule_id"]]),[])
            if len(matches)!=1:continue
            owner,action,variant=matches[0]
            quote=item.get("quote");start=item.get("start")
            try:
                target=ground_target(case.before_text,quote,start=start)
                replacement=item.get("replacement")
                require(replacement is None or isinstance(replacement,str),"Replacement type")
                require(isinstance(item.get("operation"),str) and isinstance(item.get("reason"),str),"Operation text")
            except (ValueError,TypeError,KeyError):continue
            proposal={"status":"PROPOSED","source_action_id":action["action_id"],"owner_id":owner["owner_id"],
              "method":owner["method"],"source_rule_id":variant["source_rule_id"],"frozen_action":action["original_action"],
              "target":target,"replacement":replacement,"operation":item["operation"],"reason":item["reason"],
              "is_hypothetical":True,"result_constraints":[c["predicate_text"] for c in owner["result_constraints"]],
              "result_constraints_status":"GENERATION_CONSTRAINTS_NOT_INDEPENDENTLY_VERIFIED"}
            fingerprint=stable_id("operation",proposal)
            if fingerprint not in accepted[case.identity.case_id]:continue
            index["cases"].setdefault(case.identity.case_id,{})[fingerprint]={"proposal":proposal,
                "old_gate_truth":accepted[case.identity.case_id][fingerprint],"source_cache":str(path)}
            found=True
        if found:index["source_cache_files"][str(path)]=sha256(path.read_bytes()).hexdigest()
    index["cases"]={cid:list(rows.values()) for cid,rows in index["cases"].items()}
    index["index_hash"]=stable_id("prior_proposal_index",index)
    return index
