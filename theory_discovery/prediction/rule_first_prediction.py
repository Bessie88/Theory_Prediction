"""Before-only pilot predictions from reviewed source logic; no gold/evaluation I/O."""
from pathlib import Path
import json,time
from theory_discovery.core.schemas import require,stable_id
from theory_discovery.core.storage import write_json
from ._support.rule_first_logic import VERSION,value,next_clause,comparison,refs,BOOLEAN_LABEL_POLICY
LABEL_PROTOCOL="formalized-action-labels-v3"

def case_values(owner,measurements,operation=None):
    return {cid:measurements.truth(cid,operation) for cid in owner["clauses"]}

def settle_before(owners,measurements):
    """Evaluate live owner/Action proof trees, measuring at most one leaf per tree/round."""
    rounds=0
    while True:
        needed=[]
        for owner in owners:
            vals=case_values(owner,measurements)
            if value(owner["expression"],vals) is False:continue
            trees=[owner["expression"]]+[v["gate"] for a in owner["actions"] for v in a["variants"]]
            eligible={cid for cid in owner["clauses"] if measurements.available(cid)}
            for tree in trees:
                cid=next_clause(tree,vals,eligible)
                if cid is not None:needed.append(cid)
        needed=list(dict.fromkeys(needed))
        if not needed:break
        rounds+=1;require(rounds<=500,"Finite predicate scheduling did not converge")
        measurements.measure(needed)
    return rounds

def predict_case(case,owners,bridge,specs,requests,directory,freeze_hash,registries,reuse=None,measurement_factory=None,output_mode="LABELS"):
    from ._support.rule_first_measurement import Measurements
    directory=Path(directory);checkpoint=directory/"cases"/(case.identity.case_id+".json")
    require(output_mode == "LABELS", "The main workflow returns Action Labels")
    if any(o.get("label_formalization")==BOOLEAN_LABEL_POLICY for o in owners):
        validate_boolean_action_owners(owners)
    identity=stable_id("rule_first_case",[case.to_dict(),freeze_hash,bridge["freeze_hash"],VERSION,LABEL_PROTOCOL,output_mode])
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text());require(saved["input_hash"]==identity,"Prediction resume mismatch")
        return saved
    requests.case_calls=0;stats=requests.stats();start=time.monotonic()
    measurements=(measurement_factory or Measurements)(case,bridge,specs,requests,directory/"measurements",freeze_hash)
    if reuse is not None and not measurements.reuse_audit:
        from ._support.rule_first_repairs import import_prior_measurements
        import_prior_measurements(measurements,reuse["directory"],reuse["bridge_hash"],reuse["logic_freeze_hash"],reuse.get("proposal_index_file"))
        after_import=getattr(measurements,"after_import",None)
        if after_import is not None:after_import()
    rounds=settle_before(owners,measurements);predictions={};comparisons=[];rejections=[];decision_comparisons={}
    for method in ("THEORY","NPOV"):
        selected=[o for o in owners if o["method"]==method]
        prediction=boolean_action_labels(selected,measurements,registries[method])
        predictions[method]=prediction
        # Full and lazy expressions use the same frozen Before observations.
        for owner in selected:
            vals=case_values(owner,measurements)
            for action in owner["actions"]:
                for variant in action["variants"]:
                    comparisons.append(dict(comparison(variant["gate"],vals),owner_id=owner["owner_id"],source_rule_id=variant["source_rule_id"],scope="ACTION_LABEL"))
        full_prediction=boolean_action_labels(selected,measurements,registries[method])
        decision_comparisons[method]=dict(equal=prediction==full_prediction,same_action_labels=prediction["action_labels"]==full_prediction["action_labels"])
    owner_traces=[]
    for owner in owners:
        vals=case_values(owner,measurements)
        result=comparison(owner["expression"],vals);comparisons.append(dict(result,owner_id=owner["owner_id"],scope="BEFORE_PARTIAL"))
        owner_traces.append({"owner_id":owner["owner_id"],"method":owner["method"],
           "truth_without_operation":result["truth"],"clause_values":vals,
           "unknown_clause_ids":[cid for cid in owner["clauses"] if measurements.truth(cid) is None],
           "operation_pending_clause_ids":[cid for cid,c in owner["clauses"].items() if c["truth_scope"]=="PROPOSED_OPERATION_SCOPE"],
           "result_constraints":owner["result_constraints"]})
    after=requests.stats();elapsed=time.monotonic()-start
    saved={"input_hash":identity,"case_id":case.identity.case_id,"before_hash":case.before_hash,
       "protocol":VERSION,"source_logic_freeze_hash":freeze_hash,"predicate_bridge_hash":bridge["freeze_hash"],
       "predictions":predictions,"owner_traces":owner_traces,"candidate_rejections":rejections,
       "lazy_full_comparisons":comparisons,"final_decision_comparisons":decision_comparisons,
       "comparison_passed":all(c["equal"] for c in comparisons) and all(c["equal"] for c in decision_comparisons.values()),
       "seconds":elapsed,"before_rounds":rounds,"joint_requests":requests.case_calls,
       "actual_model_usage":{k:after.get(k,0)-stats.get(k,0) for k in set(stats)|set(after)},
       "original_features_executed":len(measurements.attempted_features)-measurements.reuse_audit.get("observations",0),
       "original_features_available":len(measurements.observed),
       "direct_or_operation_measurements":len(measurements.attempted_direct)-measurements.reuse_audit.get("direct",0)-measurements.reuse_audit.get("operation_scope_values",0),
       "critical_before_rechecks":len(measurements.attempted_rechecks),"measurement_reuse":measurements.reuse_audit,
       "unmeasured_is_unknown":True,"actual_after_opened":False}
    saved["output_mode"]=output_mode
    saved["prediction_protocol"]=LABEL_PROTOCOL
    saved["action_composition_checks"]={"scope":"NONE", "llm_calls":0}
    write_json(checkpoint,saved);return saved


def validate_boolean_action_owners(owners):
    """Preparation contract: complete Before logic and existing source Actions."""
    def check(node,clauses):
        op=node["op"]
        if op=="clause":
            require(node["clause_id"] in clauses,"Missing frozen Feature")
            require(clauses[node["clause_id"]]["truth_scope"] in {"BEFORE","PRE_EDIT_METADATA"},
                    "Action Label logic must use Before Features; concrete edit requirements belong to edit verification")
        elif op=="not":check(node["arg"],clauses)
        else:
            require(op in {"and","or"} and len(node.get("args",[]))>=2,"Complete AND/OR/NOT logic required before prediction")
            for child in node["args"]:check(child,clauses)
    for owner in owners:
        require(owner.get("label_formalization")==BOOLEAN_LABEL_POLICY,"Mixed source formalization protocols")
        require(owner["actions"],"Prepare a frozen Action before predicting")
        check(owner["expression"],owner["clauses"])
        covered=set()
        for action in owner["actions"]:
            require(action["original_action"] and action["variants"],"Incomplete frozen Action")
            for variant in action["variants"]:
                check(variant["gate"],owner["clauses"]);covered.add(variant["source_rule_id"])
        require(set(owner["source_rule_ids"])<=covered,"Every predictive rule must map to an Action")


def boolean_action_labels(owners,measurements,registry=None):
    """Features -> frozen Boolean expressions -> source-backed Action Labels."""
    validate_boolean_action_owners(owners)
    labels={};decisions=[]
    for owner in owners:
        vals=case_values(owner,measurements)
        for action in owner["actions"]:
            frozen=action["original_action"];definition={k:v for k,v in frozen.items() if k!="group_id"}
            label_id=stable_id("action_label",definition)
            for variant in action["variants"]:
                gate=variant["gate"];truth=value(gate,vals);used=refs(gate)
                decision=dict(owner_id=owner["owner_id"],source_rule_id=variant["source_rule_id"],source_action_id=action["action_id"],
                    label_id=label_id,status="APPLICABLE" if truth is True else "NOT_APPLICABLE" if truth is False else "ERROR",
                    before_partial_truth=truth,gate=gate,
                    evidence_references={cid:dict(truth=vals[cid],source_pointer=owner["clauses"][cid]["source_pointer"],
                        feature_ids=getattr(measurements,"bridge",{}).get(cid,{}).get("feature_ids",[])) for cid in sorted(used)},
                    failed_feature_ids=sorted(cid for cid in used if vals[cid] is None) if truth is None else [],
                    result_constraints=variant.get("result_constraints",owner["result_constraints"]))
                for cid,row in decision["evidence_references"].items():
                    row["observation_ids"]=[measurements.observed[f].observation_id for f in row["feature_ids"] if f in getattr(measurements,"observed",{})]
                    if hasattr(measurements,"direct_key") and cid in getattr(measurements,"bridge",{}):
                        key=measurements.direct_key(cid)
                        if key in measurements.direct or key in measurements.rechecks:row["predicate_measurement_key"]=key
                decisions.append(decision)
                if truth is False:continue
                entry=labels.setdefault(label_id,dict(label_id=label_id,label=frozen.get("method",frozen.get("family",action["action_id"])),
                    action_definition=definition,source_action_ids=[],source_rule_ids=[],owner_ids=[],rule_predictions=[]))
                for field,key in (("source_action_ids",action["action_id"]),("source_rule_ids",variant["source_rule_id"]),("owner_ids",owner["owner_id"])):
                    if key not in entry[field]:entry[field].append(key)
                entry["rule_predictions"].append(decision)
    confirmed=sorted(k for k,v in labels.items() if any(d["status"]=="APPLICABLE" for d in v["rule_predictions"]))
    failed=sorted(set(labels)-set(confirmed));errors=[d for d in decisions if d["status"]=="ERROR"]
    from theory_discovery.evaluation.condition_evolution import groups_from_decisions
    return dict(status="PREDICTED" if confirmed else "ERROR" if errors else "NOT_COVERED",output_kind="ACTION_LABEL_SET",
        label_formalization=BOOLEAN_LABEL_POLICY,action_labels=confirmed,label_details=[labels[k] for k in sorted(labels)],
        failed_action_labels=failed,rule_decisions=decisions,condition_groups=groups_from_decisions(owners,decisions,registry),
        prediction_errors=errors,coverage="PARTIAL" if errors else "COMPLETE",
        conditional_action_labels=[],unknown_action_labels=[],underspecified_action_owner_ids=[],
        supported_actions=[],final_actions=[],individually_validated_actions=[],relations=[],
        composition_mode="INDEPENDENT_RULE_LABELS",labels_are_not_validated_concrete_edits=True)
