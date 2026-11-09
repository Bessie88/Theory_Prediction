"""Opt-in compact annotation strategy; detailed v2 remains available."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import json
from theory_discovery.core.schemas import require,stable_id
from theory_discovery.core.storage import write_json
from theory_discovery.annotation.compact_annotations import AnnotationTable,FORMAT,export_review_html
from .rule_first_measurement import Measurements
from .rule_first_logic import comparison,next_clause,value
from theory_discovery.features.condition_normalization import evaluate_expression
from theory_discovery.core.workflow_contracts import MeasurementMethod,ObservationStatus,MetadataItem
from theory_discovery.features.measurement_contracts import unknown,MeasurementResult,specification_hash
from theory_discovery.features.measurement_executor import MeasurementExecutor
from theory_discovery.features.feature_cache import SharedFeatureCache,NoFeatureCache
from theory_discovery.features.feature_evidence import validate_evidence
from .rule_first_contracts import SourceContracts,source_contract,hypothetical_result,PROMPT as CONTRACT_TASK,CONTRACT_VERSION
from theory_discovery.features.open_code_features import definition
RUN_PROTOCOL="reviewed-rule-first-boolean-actions-v5"
DIRECT_TASK="""Label each reviewed predicate from the complete Before and permitted metadata.
For an operation-scope row, use ONLY that row's hypothetical proposed_operation.
Do not infer actual edits or missing facts. Judge the predicate itself; the program applies NOT."""
FEATURE_TASK="""Label each Boolean feature using its full operational definition, positive/negative/ambiguity criteria.
Use only Before and permitted metadata. Contextual meaning is required; keywords alone are insufficient."""
CANDIDATE_TASK="""Label the supplied candidate's property using the common exact feature criteria.
Use the candidate ID/span and that row's hypothetical Before-derived text if present.
Each row describes a distinct candidate. Never share a judgement across different targets."""

def semantic_definition(spec):
    row=definition(spec)
    for field in ("Required Evidence","Required Output Schema"):row.pop(field,None)
    return row

class CompactCandidateBackend:
    def __init__(self,spec,table,queue=None):
        self.model_version=spec.model_version;self.prompt_version=spec.prompt_version;self.table=table;self.queue=queue
    def measure_candidates(self,spec,case,selected,mode):
        rows=[]
        for candidate in selected:
            candidate.span.check_text(case.before_text)
            row={"key":stable_id("candidate_cell",[spec.feature_id,candidate.candidate_id,mode]),
                 "candidate":candidate.to_dict(),"feature":semantic_definition(spec),"semantic_mode":mode}
            if mode=="HYPOTHETICAL_REMOVAL":
                row["hypothetical_before"]=case.before_text[:candidate.span.start]+case.before_text[candidate.span.end:]
            rows.append(row)
        context={"before_text":case.before_text,"pre_context":[f.to_dict() for f in case.pre_context]}
        if self.queue is not None and not self.table.cached("CANDIDATE_PROPERTY",rows,context):
            for row in rows:self.queue[row["key"]]=row
            return tuple(unknown("QUEUED_COMPACT_CANDIDATE",spec.model_version,spec.prompt_version) for _ in selected)
        answers=self.table.annotate("CANDIDATE_PROPERTY",CANDIDATE_TASK,rows,context)
        result=[]
        for candidate in selected:
            answer=answers[stable_id("candidate_cell",[spec.feature_id,candidate.candidate_id,mode])];v=answer["value"]
            if v is None:result.append(unknown("COMPACT_CANDIDATE_ANNOTATION_ERROR",spec.model_version,spec.prompt_version));continue
            # The candidate ID labels an already supplied span, not an invented quote.
            measured=MeasurementResult(v,"KNOWN",(candidate.span,),(),"SPANS",
                "Compact candidate label; input candidate supplies span; no model rationale requested",(),
                spec.model_version,spec.prompt_version)
            prop=replace(spec,value_type="boolean",allowed_values=(True,False),structured_schema_name="",structured_fields=())
            result.append(validate_evidence(prop,case,measured))
        return tuple(result)

class CompactMeasurements(Measurements):
    def __init__(self,case,bridge,specs,requests,directory,freeze_hash):
        self.annotations={}
        path=Path(directory)/(case.identity.case_id+".json")
        if path.exists():
            require(json.loads(path.read_text()).get("annotation_protocol")==FORMAT,
                    "Feature annotation protocol changed; use a new output directory")
        super().__init__(case,bridge,specs,requests,directory,freeze_hash)
        self.table=AnnotationTable(case,requests,Path(directory).parent/"annotation_tables",freeze_hash)
        if self.path.exists():
            saved=json.loads(self.path.read_text());self.annotations=saved.get("compact_feature_annotations",{})
            for fid,record in self.annotations.items():
                require(fid in specs and (record["value"] is None or type(record["value"]) is bool),"Invalid saved Boolean annotation")
        self.candidate_queue={}
        def provider(spec):
            if spec.measurement_method is MeasurementMethod.HYBRID:return CompactCandidateBackend(spec,self.table,self.candidate_queue)
            return SimpleNamespace(model_version=spec.model_version,prompt_version=spec.prompt_version,
                complete=lambda s,p:requests.checked("ORIGINAL_COMPLEX_FEATURE",s,p,lambda x:x))
        candidate_reviews=any(self.table.rows[key]["kind"]=="CANDIDATE_PROPERTY" for key in self.table.review)
        review_suffix=stable_id("review",self.table.review) if candidate_reviews else "model-only"
        if candidate_reviews:
            for fid in list(self.observed):
                if specs[fid].measurement_method is MeasurementMethod.HYBRID:
                    del self.observed[fid];self.attempted_features.discard(fid)
        self.executor=MeasurementExecutor(True,True,provider,SharedFeatureCache(Path(directory)/"compact_program_cache"/review_suffix),RUN_PROTOCOL)
        self.hybrid_executor=MeasurementExecutor(True,True,provider,NoFeatureCache(),RUN_PROTOCOL)
    def save(self):
        super().save()
        data=json.loads(self.path.read_text());data["compact_feature_annotations"]=self.annotations
        data["annotation_protocol"]=FORMAT
        data["annotation_evidence_contract"]="Explicit label-only model annotations; no legacy span-validation claim."
        write_json(self.path,data)
    def views(self):
        result=dict(self.observed)
        for fid,record in self.annotations.items():
            row=self.table.rows.get(record["row_id"])
            label=self.table.effective(row) if row else record["effective_label"]
            v={"Y":True,"N":False,None:None}[label]
            result[fid]=SimpleNamespace(identity=self.case.identity,extracted_value=v,
                status=ObservationStatus.KNOWN if v is not None else ObservationStatus.UNKNOWN)
        return result
    def record_value(self,record):
        row=self.table.rows.get(record.get("row_id"))
        return {"Y":True,"N":False,None:None}[self.table.effective(row)] if row else record.get("value")
    def after_import(self):
        # Keep compatible known values; an old semantic U is not a binary answer.
        for records,attempted in ((self.direct,self.attempted_direct),(self.rechecks,self.attempted_rechecks)):
            for key,record in list(records.items()):
                if record.get("value") is None:
                    del records[key];attempted.discard(key)
    def truth(self,cid,operation=None):
        if not hasattr(self,"table"):return super().truth(cid,operation)
        m=self.bridge[cid];mode=m["mode"]
        if mode in ("UNKNOWN","RESULT_CONSTRAINT"):return None
        if mode in ("DIRECT","OPERATION_SCOPE"):
            if mode=="OPERATION_SCOPE" and operation is None:return None
            return self.record_value(self.direct.get(self.direct_key(cid,operation),{}))
        v=evaluate_expression(m["expression"],self.views())
        if v is not None:return not v if mode=="NEGATE_ORIGINAL" else v
        key=self.direct_key(cid)
        if key in self.rechecks:return self.record_value(self.rechecks[key])
        if key in self.direct:return self.record_value(self.direct[key])
        return None
    def measure(self,clause_ids,operation=None):
        fids=[];direct={};recheck={}
        for cid in dict.fromkeys(clause_ids):
            if not self.available(cid,operation):continue
            m=self.bridge[cid]
            if m["mode"] in ("DIRECT","OPERATION_SCOPE"):
                key=self.direct_key(cid,operation)
                direct[key]={"key":key,"predicate_text":m["predicate"]["predicate_text"],"truth_scope":m["predicate"]["truth_scope"]}
                if operation is not None:direct[key]["proposed_operation"]=operation
            elif m["mode"] in ("ORIGINAL","NEGATE_ORIGINAL"):
                pending=sorted(set(m["feature_ids"])-self.attempted_features,
                    key=lambda fid:(self.specs[fid].measurement_method is not MeasurementMethod.CODE,fid))
                if pending:fids.append(pending[0])
                else:
                    key=self.direct_key(cid);recheck[key]={"key":key,"predicate_text":m["predicate"]["predicate_text"],
                        "truth_scope":m["predicate"]["truth_scope"]}
        scalar=[];hybrid=[]
        for fid in dict.fromkeys(fids):
            spec=self.specs[fid]
            if spec.measurement_method is MeasurementMethod.SEMANTIC and spec.value_type=="boolean":
                scalar.append(dict(key=fid,specification_hash=specification_hash(spec),
                    **semantic_definition(spec)))
            else:
                executor=self.hybrid_executor if spec.measurement_method is MeasurementMethod.HYBRID else self.executor
                obs=executor.execute(spec,self.case)
                if spec.measurement_method is MeasurementMethod.HYBRID:hybrid.append(fid)
                self.observed[fid]=replace(obs,extraction_metadata=obs.extraction_metadata+(
                    MetadataItem("actual_measurement_protocol",FORMAT),MetadataItem("model_rationale_requested",False)))
                self.attempted_features.add(fid)
        context={"before_text":self.case.before_text,"pre_context":[f.to_dict() for f in self.case.pre_context]}
        if self.candidate_queue:
            self.table.annotate("CANDIDATE_PROPERTY",CANDIDATE_TASK,list(self.candidate_queue.values()),context)
            self.candidate_queue.clear()
            for fid in hybrid:
                obs=self.hybrid_executor.execute(self.specs[fid],self.case)
                self.observed[fid]=replace(obs,extraction_metadata=obs.extraction_metadata+(
                    MetadataItem("actual_measurement_protocol",FORMAT),MetadataItem("model_rationale_requested",False)))
        # Incomplete candidate search cannot supply a binary absence decision.
        # Ask the same full feature from Before rather than converting its U to N.
        coverage_gaps={"No candidates in an incomplete domain; semantic absence is not established",
                       "No established positive witness; candidate domain/labels incomplete"}
        for fid in hybrid:
            obs=self.observed[fid];spec=self.specs[fid]
            failed=any(r["kind"]=="CANDIDATE_PROPERTY" and r["definition"]["feature"]["feature_id"]==fid
                       and self.table.effective(r) is None for r in self.table.rows.values())
            if spec.value_type=="boolean" and coverage_gaps.intersection(obs.uncertainty) and not failed:
                scalar.append(dict(key=fid,specification_hash=specification_hash(spec),
                    **semantic_definition(spec),measurement_stage="FULL_BEFORE_BINARY_FALLBACK",
                    candidate_search_is_incomplete=True))
        if scalar:
            answers=self.table.annotate("BOOLEAN_FEATURE",FEATURE_TASK,scalar,context)
            self.annotations.update(answers);self.attempted_features.update(answers)
        if direct:
            answers=self.table.annotate("OPERATION_SCOPE" if operation else "BEFORE_PREDICATE",DIRECT_TASK,list(direct.values()),context)
            self.direct.update(answers);self.attempted_direct.update(answers)
        if recheck:
            answers=self.table.annotate("CRITICAL_BEFORE_PREDICATE",DIRECT_TASK,list(recheck.values()),context)
            self.rechecks.update(answers);self.attempted_rechecks.update(answers)
        self.save()
    def settle_operation_set(self,pairs):
        # Preserve exact generated target/replacement parameters for review/replay.
        for owner,action,variant,proposal in pairs:
            self.reusable_candidates[(owner["owner_id"],action["action_id"],variant["source_rule_id"])]=proposal
        if pairs:self.save()
        while True:
            rows={}
            for owner,action,variant,proposal in pairs:
                vals={cid:self.truth(cid,proposal) for cid in owner["clauses"]}
                eligible={cid for cid,c in owner["clauses"].items() if c["truth_scope"]=="PROPOSED_OPERATION_SCOPE"
                          and self.available(cid,proposal)}
                cid=next_clause(variant["gate"],vals,eligible)
                if cid is not None:
                    key=self.direct_key(cid,proposal);m=self.bridge[cid]
                    rows[key]={"key":key,"predicate_text":m["predicate"]["predicate_text"],
                        "truth_scope":"PROPOSED_OPERATION_SCOPE","proposed_operation":proposal}
            if not rows:break
            answers=self.table.annotate("OPERATION_SCOPE",DIRECT_TASK,list(rows.values()),
                {"before_text":self.case.before_text,"pre_context":[f.to_dict() for f in self.case.pre_context]})
            self.direct.update(answers);self.attempted_direct.update(answers);self.save()
        result={}
        for owner,action,variant,proposal in pairs:
            vals={cid:self.truth(cid,proposal) for cid in owner["clauses"]}
            key=(owner["owner_id"],action["action_id"],variant["source_rule_id"],stable_id("operation",proposal))
            result[key]=(comparison(variant["gate"],vals),vals)
        return result

class CompactSourceContracts(SourceContracts):
    def __init__(self,case,requests,directory,freeze_hash,table):
        super().__init__(case,requests,directory,freeze_hash);self.table=table
    def check(self,proposals):
        rows=[];keys=[]
        for owner,action,variant,proposal in proposals:
            key=self.key(owner,action,variant,proposal);keys.append(key)
            if key in self.saved:continue
            result=hypothetical_result(self.case,proposal)
            if result==self.case.before_text:
                self.saved[key]={"value":False,"rationale":"DETERMINISTIC_NO_CHANGE","contract_version":CONTRACT_VERSION};continue
            rows.append({"key":key,"source_contract":source_contract(owner,action,variant),
                "hypothetical_proposal":proposal,"hypothetical_result_text":result})
        task=CONTRACT_TASK.split('Return {"results"')[0]+"\nY means all source requirements hold; N means a source violation; U means undecidable."
        if rows:
            answers=self.table.annotate("COMPLETE_SOURCE_CONTRACT",task,rows,
                {"before_text":self.case.before_text,"pre_context":[f.to_dict() for f in self.case.pre_context]})
            for key,answer in answers.items():
                self.saved[key]=dict(answer,validator="COMPACT_MODEL_SOURCE_CONTRACT",contract_version=CONTRACT_VERSION)
        write_json(self.path,{"identity":self.identity,"checks":self.saved,"annotation_protocol":FORMAT})
        result=[]
        for key in keys:
            record=dict(self.saved[key]);row=self.table.rows.get(record.get("row_id"))
            if row:
                label=self.table.effective(row)
                record.update(value={"Y":True,"N":False,"U":None}[label],effective_label=label,
                    label_origin="HUMAN_REVIEW" if row["row_id"] in self.table.review else "MODEL_UNREVIEWED")
            result.append(record)
        return result

def compact_prediction_case(case,owners,bridge,specs,requests,directory,freeze_hash,registries,reuse=None,output_mode="LABELS"):
    from .rule_first_prediction import predict_case
    from hashlib import sha256
    directory=Path(directory)
    human=directory/"annotation_tables"/(case.identity.case_id+".human.json")
    review_hash=sha256(human.read_bytes()).hexdigest() if human.exists() else None
    checkpoint=directory/"cases"/(case.identity.case_id+".json")
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        require(saved.get("annotation_protocol")==FORMAT,"Feature annotation protocol changed; use a new output directory")
        require(saved.get("annotation_review_hash")==review_hash,
            "Human labels changed: import the review into a separate output directory to preserve model-only decisions")
    box={}
    def measurements(*args):
        m=CompactMeasurements(*args);box["table"]=m.table;box["measurements"]=m;return m
    def contracts(*args):return CompactSourceContracts(*args,table=box["table"])
    result=predict_case(case,owners,bridge,specs,requests,directory,freeze_hash,registries,
        reuse=reuse,measurement_factory=measurements,contract_factory=contracts,output_mode=output_mode)
    table=box.get("table")
    result.update(run_protocol=RUN_PROTOCOL,annotation_protocol=FORMAT,feature_labels=["Y","N"],
        feature_annotation_errors=[dict(row_id=r["row_id"],kind=r["kind"],state=r["model_state"])
            for r in table.rows.values() if r["kind"]!="COMPLETE_SOURCE_CONTRACT" and table.effective(r) is None] if table else result.get("feature_annotation_errors",[]),
        annotation_review_state="MODEL_LABELS_UNREVIEWED_EXCEPT_EXPLICIT_HUMAN_LABELS",
        annotation_review_hash=review_hash)
    if table is not None and output_mode == "LABELS":
        views = box["measurements"].views()
        feature_ids = {fid for decision in result["predictions"]["THEORY"]["rule_decisions"]
                       for ref in decision.get("evidence_references", {}).values()
                       for fid in ref.get("feature_ids", ())}
        result["theory_feature_truths"] = {
            fid: views[fid].extracted_value if fid in views and views[fid].status is ObservationStatus.KNOWN
            and type(views[fid].extracted_value) is bool else None for fid in sorted(feature_ids)}
    path=Path(directory)/"cases"/(case.identity.case_id+".json");write_json(path,result)
    if table is not None:export_review_html(table.bundle(),table.directory/(case.identity.case_id+".html"))
    return result
