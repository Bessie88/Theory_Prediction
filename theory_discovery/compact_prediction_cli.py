
"""Binary Feature pipeline. Default is zero-model preflight; --run is explicit."""
import argparse,json,os
from pathlib import Path
from hashlib import sha256
from collections import Counter
from .schemas import require,stable_id
from .storage import write_json
from .workflow_contracts import PredictionCase
from .stage_interfaces import StudySpecification
from .rule_first_repairs import correct_evidence_contract
from .compact_annotations import CompactRequests,instruction_for,pack,FORMAT
from .compact_rule_first import compact_prediction_case,semantic_definition,RUN_PROTOCOL,FEATURE_TASK,DIRECT_TASK,CANDIDATE_TASK
from .rule_first_contracts import source_contract,PROMPT as CONTRACT_TASK
from .config import configured_tokenizer_path

def load_sources(source):
    source=Path(source)
    snapshot=json.loads((source/"code_snapshot.json").read_text())
    for group in ("files","input_hashes"):
        for name,digest in snapshot[group].items():
            require(sha256((source/name).read_bytes()).hexdigest()==digest,"Parent frozen input/code changed: "+name)
    logic=json.loads((source/"source_logic_freeze.json").read_text())
    bridge=json.loads((source/"predicate_bridge_freeze.json").read_text())
    study=StudySpecification.from_dict(json.loads((source/"old_study.json").read_text()))
    specs,_,_=correct_evidence_contract({s.feature_id:s for s in study.measurement_freeze.snapshot.specifications})
    require(logic["predicate_bridge_hash"]==bridge["freeze_hash"],"Source bridge mismatch")
    cases=[PredictionCase.from_dict(r) for r in json.loads((source/"later_before.json").read_text())["records"]]
    manifest={"protocol":RUN_PROTOCOL,"annotation_protocol":FORMAT,"source_logic_freeze_hash":logic["freeze_hash"],
        "source_bridge_hash":bridge["freeze_hash"],"feature_specification_hash":stable_id("features",[s.to_dict() for s in specs.values()]),
        "label_output_tokens":512,"generation_output_tokens":4096,"max_rows":32,"thinking":False,
        "original_logic_and_actions_unchanged":True,"human_labels_are_separate":True,"heldout_after_opened":False}
    manifest["freeze_hash"]=stable_id("compact_prediction_freeze",manifest)
    return logic["owners"],bridge,specs,{"THEORY":study.theory_registry,"NPOV":study.policy_registry},cases,manifest

def parent_completed(source,cases,output_mode=None):
    result={}
    for case in cases:
        path=Path(source)/"predictions/cases"/(case.identity.case_id+".json")
        if path.exists():
            record=json.loads(path.read_text())
            if output_mode is not None and record.get("annotation_protocol")!=FORMAT:continue
            if output_mode is not None and record.get("output_mode","EDITS") != output_mode:continue
            if output_mode=="LABELS":
                from .rule_first_prediction import LABEL_PROTOCOL
                if record.get("prediction_protocol")!=LABEL_PROTOCOL:continue
            require(record["before_hash"]==case.before_hash,"Completed case Before mismatch")
            require(record["comparison_passed"],"Completed parent case failed logical comparison")
            result[case.identity.case_id]={"path":str(path),"sha256":sha256(path.read_bytes()).hexdigest(),
                "protocol":record["protocol"],"status":"REUSED_COMPLETED_PARENT_RESULT_NOT_RELABELLED"}
    return result

def preflight(owners,bridge,specs,cases,output,manifest):
    from .config import LLMConfig
    from .llm import QwenJSONClient,PLAIN_LANGUAGE_INSTRUCTION
    from .storage import Cache
    from .reference_transport import short_reference_payload
    client=QwenJSONClient(LLMConfig(tokenizer_path=configured_tokenizer_path(),max_tokens=512,max_prompt_chars=200000),Cache(output/"preflight_cache"))
    largest=max(cases,key=lambda c:len(c.before_text))
    context={"before_text":largest.before_text,"pre_context":[f.to_dict() for f in largest.pre_context]}
    reports=[]
    def budget(stage,task,rows):
        def check(group):
            payload=dict(context,queries=[dict(id="Q"+str(i+1),**{k:v for k,v in r.items() if k!="key"}) for i,r in enumerate(group)])
            wire,_=short_reference_payload(payload)
            kind="COMPLETE_SOURCE_CONTRACT" if stage=="COMPLETE_SOURCE_CONTRACT" else stage
            msgs=[{"role":"system","content":task+"\n"+instruction_for(kind)+"\n"+PLAIN_LANGUAGE_INSTRUCTION},
                  {"role":"user","content":json.dumps(wire,ensure_ascii=False)}]
            n=client.prompt_tokens(msgs);require(n is not None,"Production tokenizer unavailable")
            if n+512+512>32768 or sum(len(m["content"]) for m in msgs)>200000:
                require(len(group)>1,"Single annotation row exceeds context; no truncation")
                k=len(group)//2;check(group[:k]);check(group[k:]);return
            reports.append({"stage":stage,"rows":len(group),"input_tokens":n,"output_reserve":512,"safety":512})
        for group in pack(rows):check(group)
    budget("BOOLEAN_FEATURE",FEATURE_TASK,[dict(key=fid,**semantic_definition(s)) for fid,s in specs.items()
        if s.measurement_method.value=="SEMANTIC" and s.value_type=="boolean"])
    budget("REVIEWED_PREDICATE",DIRECT_TASK,[dict(key=cid,predicate_text=m["predicate"]["predicate_text"],
        truth_scope=m["predicate"]["truth_scope"]) for cid,m in bridge["mappings"].items()])
    from .measurement_contracts import procedure_for
    candidate_rows=[]
    for fid,spec in specs.items():
        if spec.measurement_method.value!="HYBRID":continue
        mode=procedure_for(spec).semantic_mode
        row={"key":fid,"feature":semantic_definition(spec),"semantic_mode":mode,
            "candidate":{"candidate_id":"budget-only","span":{"side":"before","start":0,
                "end":len(largest.before_text),"quote":largest.before_text}}}
        if mode=="HYPOTHETICAL_REMOVAL":row["hypothetical_before"]=""
        candidate_rows.append(row)
    budget("CANDIDATE_PROPERTIES",CANDIDATE_TASK,candidate_rows)
    rows=[]
    for owner in owners if manifest.get("output_mode","EDITS")=="EDITS" else []:
        for action in owner["actions"]:
            for variant in action["variants"]:
                proposal={"frozen_action":action["original_action"],"target":{"side":"before","start":0,
                    "end":len(largest.before_text),"quote":largest.before_text},"replacement":"Budget-only hypothetical"}
                rows.append({"key":variant["source_rule_id"],"source_contract":source_contract(owner,action,variant),
                    "hypothetical_proposal":proposal,"hypothetical_result_text":largest.before_text})
    budget("COMPLETE_SOURCE_CONTRACT",CONTRACT_TASK.split('Return {"results"')[0],rows)
    return {"models_called":0,"heldout_after_opened":False,"requests_measured":len(reports),
        "max_input_tokens":max(r["input_tokens"] for r in reports),"budgets":reports,
        "real_speed_and_accuracy_measured":False,"annotation_protocol":FORMAT}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=20,help="First N heldout Before cases; completed parent cases are skipped.")
    p.add_argument("--run",action="store_true",help="Call the already running model service only for unfinished cases.")
    p.add_argument("--generate-edits",action="store_true",help="Optional concrete edits; default returns Action Labels.")
    args=p.parse_args();require(args.source_dir.resolve()!=args.output.resolve(),"Preserve parent run")
    args.output.mkdir(parents=True,exist_ok=True)
    package=Path(__file__).resolve().parent
    deployment={str(path):sha256(path.read_bytes()).hexdigest() for path in sorted(package.glob("*.py"))}
    pinned=args.output/"deployment_sources.json"
    if pinned.exists():require(json.loads(pinned.read_text())==deployment,"Prepared source code changed; review before replacing deployment freeze")
    else:write_json(pinned,deployment)
    owners,bridge,specs,registries,all_cases,manifest=load_sources(args.source_dir)
    from .rule_first_prediction import LABEL_PROTOCOL
    manifest["prediction_protocol"]=LABEL_PROTOCOL
    manifest["input_case_ids"]=[c.identity.case_id for c in all_cases]
    manifest["output_mode"]="EDITS" if args.generate_edits else "LABELS"
    manifest["freeze_hash"]=stable_id("compact_prediction_output_mode",manifest)
    require(0<args.limit<=len(all_cases),"Invalid case limit")
    cases=all_cases[:args.limit]
    path=args.output/"protocol.json"
    if path.exists():require(json.loads(path.read_text())==manifest,"Compact configuration changed")
    else:write_json(path,manifest)
    reused=parent_completed(args.source_dir,cases,manifest["output_mode"]);write_json(args.output/"completed_result_references.json",reused)
    if not args.run:
        report=preflight(owners,bridge,specs,all_cases,args.output,manifest)
        report.update(reused_completed_cases=len(reused),new_cases=sum(c.identity.case_id not in reused for c in cases))
        write_json(args.output/"preflight.json",report)
        print(json.dumps({k:v for k,v in report.items() if k!="budgets"}));return
    from .workflow_config import ModelSettings
    from .measurement_contracts import MeasurementRuntime
    from .measurement_runtime import configured_backend
    def backend(tokens,folder):
        b=configured_backend(ModelSettings(True,"qwen3.8-27b",os.environ["SIMULATION_MODEL_BASE_URL"],
            RUN_PROTOCOL,0.0,tokens,2,"Qwen-3.8",None),
            MeasurementRuntime(True,True,RUN_PROTOCOL,str(args.output/"cache"/folder),200000),RUN_PROTOCOL)
        b.client.config.tokenizer_path=configured_tokenizer_path()
        return b
    req=None;results=[]
    for case in cases:
        if case.identity.case_id in reused:continue
        if req is None:req=CompactRequests(backend(4096,"generation"),backend(512,"labels"),args.output/"audit")
        results.append(compact_prediction_case(case,owners,bridge,specs,req,args.output/"predictions",manifest["freeze_hash"],registries,
            reuse={"directory":str(args.source_dir/"predictions"),"bridge_hash":bridge["freeze_hash"],"logic_freeze_hash":manifest["source_logic_freeze_hash"]},output_mode=manifest["output_mode"]))
        write_json(args.output/"progress.json",{"completed_compact":len(results),"completed_reused":len(reused),"requested":len(cases)})
    for item in reused.values():require(sha256(Path(item["path"]).read_bytes()).hexdigest()==item["sha256"],"Parent result changed")
    summary={"protocol":RUN_PROTOCOL,"reused_completed_cases":len(reused),"new_compact_cases":len(results),
        "all_new_lazy_full_equal":all(r["comparison_passed"] for r in results),"parent_outputs_untouched":True,
        "usage":{k:sum(r["actual_model_usage"].get(k,0) for r in results) for k in ("calls","prompt_tokens","completion_tokens")},
        "seconds":sum(r["seconds"] for r in results),"heldout_after_opened":False}
    if results and manifest["output_mode"]=="LABELS":
        from .condition_evolution import freeze_prediction_snapshot
        freeze_prediction_snapshot(args.output,manifest["source_logic_freeze_hash"])
    write_json(args.output/"summary.json",summary);print(json.dumps(summary))
if __name__=="__main__":main()
