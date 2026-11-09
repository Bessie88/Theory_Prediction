
"""Binary Feature pipeline. Default is zero-model preflight; --run is explicit."""
import json
from pathlib import Path
from hashlib import sha256
from theory_discovery.core.schemas import require,stable_id
from theory_discovery.core.workflow_contracts import PredictionCase
from theory_discovery.core.stage_interfaces import StudySpecification
from .rule_first_repairs import correct_evidence_contract
from theory_discovery.annotation._support.compact_annotations import instruction_for,pack,FORMAT
from .compact_rule_first import semantic_definition,RUN_PROTOCOL,FEATURE_TASK,DIRECT_TASK,CANDIDATE_TASK
from theory_discovery.core.config import configured_tokenizer_path

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

def preflight(owners,bridge,specs,cases,output,manifest):
    from theory_discovery.core.config import LLMConfig
    from theory_discovery.core.llm import QwenJSONClient,PLAIN_LANGUAGE_INSTRUCTION
    from theory_discovery.core.storage import Cache
    from theory_discovery.core.reference_transport import short_reference_payload
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
    from theory_discovery.features._support.measurement_contracts import procedure_for
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
    return {"models_called":0,"heldout_after_opened":False,"requests_measured":len(reports),
        "max_input_tokens":max(r["input_tokens"] for r in reports),"budgets":reports,
        "real_speed_and_accuracy_measured":False,"annotation_protocol":FORMAT}
