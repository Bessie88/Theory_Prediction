"""Before-only open coding, exhaustive definition alignment and lazy critical checks.

Feature definitions/source logic stay frozen. Actual model prompts use VERSION;
observation prompt_version remains the frozen per-feature evidence-contract version,
with the actual execution protocol explicitly recorded in separate metadata.
No old per-feature observations are silently reused in this new measurement arm.
"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
from theory_discovery.core.schemas import require, stable_id
from theory_discovery.core.storage import write_json
from theory_discovery.core.workflow_contracts import FeatureObservation, ObservationStatus, MetadataItem, feature_observation_id_for, MeasurementMethod
from .measurement_contracts import specification_hash, unknown
from .semantic_features import _semantic_response, feature_payload, label_before_conditions
from .feature_evidence import validate_evidence
from .condition_normalization import evaluate_expression, referenced_features
from theory_discovery.induction.theory_logic import evaluate_logic
from .measurement_executor import MeasurementExecutor
from .measurement_runtime import configured_backend
from .feature_cache import SharedFeatureCache

VERSION="open-code-align-critical-v3"
OPEN_PROMPT="""Open-code ONLY the supplied Before text, without any Theory, policy or feature list.
Describe all observable content, wording, attribution, stance, structure and uncertainty in plain language.
Each code must have a short label, definition and an exact quote from Before.
Do not infer an edit, After, motives or unavailable article context. Return {"codes":[{"code_id":"C1","label":"...","definition":"...","quote":"..."}]}.
An empty list is allowed; absence from the list does not mean false."""
MAP_PROMPT="""Match supplied open codes to frozen definitions by meaning and scope.
Use Before only as context. Return candidate associations, NOT feature truth values.
Return {"results":[{"feature_id":"ID","code_ids":["C1"],"rationale":"short reason"}]}.
Use supplied IDs only. Empty results are allowed. Missing codes never prove absence."""
CHECK_PROMPT="""Verify the supplied frozen features directly against this Before, using open codes only as auxiliary evidence.
Missing open codes are never proof of absence. Check all criteria and scope from the original Before.
Return each requested feature_id once, value (typed scalar or null), quote (exact Before text or empty if absent), rationale (one short sentence).
Use null when evidence or context is insufficient. Return {"results":[...]}. Do not infer After or actions."""
MAX_GROUP=12
MAX_GROUP_CHARS=18000

def definition(spec):
    row=feature_payload(spec,SimpleNamespace(before_text="",pre_context=()))
    row.pop("before_text");row.pop("Pre-edit Metadata")
    row.pop("Required Output Schema",None)
    return dict(feature_id=spec.feature_id,value_type=spec.value_type,required_input_fields=list(spec.required_input_fields),**row)

def groups(specs):
    chunk=[];size=0
    for spec in specs:
        cost=len(json.dumps(definition(spec),ensure_ascii=False))
        require(cost <= MAX_GROUP_CHARS,"Single feature exceeds alignment budget; no truncation")
        if chunk and (len(chunk)>=MAX_GROUP or size+cost>MAX_GROUP_CHARS):
            yield chunk;chunk=[];size=0
        chunk.append(spec);size+=cost
    if chunk:yield chunk

def pending_logic(expr, values):
    if evaluate_logic(expr,values)[0] is not None:return []
    op=expr["op"]
    if op in ("clause","boundary_clear"):return [expr["clause_id"]]
    if op in ("and","or"):
        return [k for child in expr["args"] for k in pending_logic(child,values)]
    return pending_logic(expr["arg"],values) if op=="not" else []

def critical_features(study, observed, attempted):
    needed=[]
    for registry,freeze in ((study.theory_registry,study.measurement_freeze),
                            (study.policy_registry,study.policy_measurement_freeze)):
        bindings={b.clause_id:b for b in freeze.bindings}
        values={key:(evaluate_expression(json.loads(b.expression_json),observed)
                     if b.resolution in ("RESOLVED","NO_ADDITIONAL_BOUNDARY") else None) for key,b in bindings.items()}
        for rule in registry.theories:
            expr=json.loads(rule.applicability_expression_json)
            value=evaluate_logic(expr,values)[0]
            clauses=pending_logic(expr,values) if value is None else []
            if value is True:
                for action in rule.directives:
                    gate=json.loads(action.when_expression_json)
                    clauses+=pending_logic(gate,values)
                    if evaluate_logic(gate,values)[0] is True and action.target_feature_id:
                        if action.target_feature_id not in attempted:needed.append(action.target_feature_id)
            # Check one unresolved feature per clause first, then re-evaluate short circuits.
            for key in clauses:
                expr=json.loads(bindings[key].expression_json)
                choices=sorted(referenced_features(expr)-attempted)
                if choices:needed.append(choices[0])
    return list(dict.fromkeys(needed))

def execute_case(inputs,case,provider):
    study=inputs.study;specs={s.feature_id:s for s in study.measurement_freeze.snapshot.specifications}
    setting=replace(inputs.settings.semantic_model,prompt_version=VERSION,max_tokens=4096)
    backend=configured_backend(setting,inputs.runtime,VERSION)
    calls=0
    def checked(system,payload,decode):
        nonlocal calls
        calls+=1
        require(calls<=160,"Per-case open-coding call budget exhausted")
        return backend.complete_checked(system,payload,decode)
    def decode_codes(raw):
        require(isinstance(raw,dict) and set(raw)=={"codes"} and isinstance(raw["codes"],list),"Invalid open codes")
        ids=set()
        for c in raw["codes"]:
            require(set(c)=={"code_id","label","definition","quote"} and
                    all(isinstance(v,str) and v.strip() for v in c.values()),"Invalid open code record")
            require(c["code_id"] not in ids and c["quote"] in case.before_text,"Open code needs unique ID and Before evidence")
            ids.add(c["code_id"])
        return raw["codes"]
    codes=checked(OPEN_PROMPT,{"before_text":case.before_text},decode_codes)
    observed={};attempted=set();failures=[];matches=[];candidate_batches=[];hybrid_executed=[]
    from .batched_candidate_features import CandidateBatchBackend
    def lazy_provider(spec):
        return CandidateBatchBackend(spec,checked,definition,candidate_batches) if spec.measurement_method is MeasurementMethod.HYBRID else provider(spec)
    executor=MeasurementExecutor(True,inputs.runtime.models_authorized,lazy_provider,
        SharedFeatureCache(inputs.runtime.cache_directory),inputs.runtime.configuration_id,inputs.runtime.max_text_characters)
    input_hash=stable_id("feature_input",case.to_dict())
    def observation(spec,result):
        return FeatureObservation(feature_observation_id_for(case.identity,spec.feature_id,input_hash,
            specification_hash(spec),spec.measurement_method),case.identity,spec.feature_id,result.value,
            result.evidence,result.metadata_names,spec.measurement_method,ObservationStatus(result.status),result.uncertainty,
            (MetadataItem("model_version",result.model_version),MetadataItem("prompt_version",result.prompt_version),
             MetadataItem("feature_contract_prompt_version",spec.prompt_version),
             MetadataItem("actual_measurement_protocol",VERSION),MetadataItem("evidence_scope",result.scope),
             MetadataItem("rationale",result.rationale)),input_hash,specification_hash(spec))
    def parse(row,spec):
        value=row["value"];quote=row["quote"]
        require(value is None or type(value) in (bool,int,float,str),"Unsupported joint typed value")
        require(isinstance(quote,str) and isinstance(row["rationale"],str) and row["rationale"].strip(),"Invalid compact evidence")
        evidence=[{"side":"before","start":0,"end":len(quote),"quote":quote}] if quote else []
        raw={"value":value,"status":"UNKNOWN" if value is None else "KNOWN","evidence":evidence,
             "metadata_names":[],"scope":"UNKNOWN" if value is None else "SPANS" if quote else "FULL_BEFORE",
             "rationale":row["rationale"],"uncertainty":["Insufficient Before evidence"] if value is None else []}
        contract=SimpleNamespace(model_version=spec.model_version,prompt_version=spec.prompt_version)
        result=_semantic_response(raw,spec,case,contract,None)
        return observation(spec,validate_evidence(spec,case,result))
    code_ids={c["code_id"] for c in codes}
    def request(group,critical):
        allowed={s.feature_id:s for s in group}
        def decode(raw):
            require(isinstance(raw,dict) and set(raw)=={"results"} and isinstance(raw["results"],list),"Invalid alignment")
            found={};errors=[];seen=set()
            for row in raw["results"]:
                if not isinstance(row,dict) or row.get("feature_id") not in allowed or row["feature_id"] in seen:
                    errors.append({"feature_id":row.get("feature_id") if isinstance(row,dict) else None,
                                   "error":"Unknown/duplicate mapped feature; ignored, not false"})
                    continue
                seen.add(row["feature_id"])
                if not critical:
                    if not (isinstance(row.get("code_ids"),list) and row["code_ids"] and
                            all(isinstance(k,str) and k in code_ids for k in row["code_ids"])):
                        errors.append({"feature_id":row["feature_id"],"error":"Unverified code association; direct check still required"})
                        continue
                    matches.append({"feature_id":row["feature_id"],"code_ids":row["code_ids"],
                                    "rationale":row.get("rationale",""),"candidate_only":True})
                    continue
                try:
                    found[row["feature_id"]]=parse(row,allowed[row["feature_id"]])
                except (ValueError,TypeError,KeyError) as exc:errors.append({"feature_id":row["feature_id"],"error":str(exc)})
            # Invalid/missing rows are not false and are retried only if decision-critical.
            return found,errors
        payload={"before_text":case.before_text,"open_codes":codes,
                 "pre_context":[f.to_dict() for f in case.pre_context],
                 "features":[definition(s) for s in group]}
        try:
            found,errors=checked(CHECK_PROMPT if critical else MAP_PROMPT,payload,decode)
        except (ValueError, RuntimeError) as exc:
            if len(group)>1 and any(word in str(exc) for word in ("output_length_limit","context budget","Prompt character budget")):
                middle=len(group)//2
                request(group[:middle],critical);request(group[middle:],critical)
                return
            raise
        failures.extend(errors);observed.update(found)
        if critical:attempted.update(allowed)
    # Objective/hybrid programs keep their exact original execution semantics.
    for spec in specs.values():
        if spec.measurement_method is MeasurementMethod.CODE:
            observed[spec.feature_id]=executor.execute(spec,case);attempted.add(spec.feature_id)
    semantic=[s for s in specs.values() if s.measurement_method is MeasurementMethod.SEMANTIC and s.value_type!="structured"]
    if codes:
        for group in groups(semantic):request(group,False)
    attempted.update(k for k,v in observed.items() if v.status is ObservationStatus.KNOWN)
    rounds=0
    while True:
        keys=critical_features(study,observed,attempted)
        if not keys:break
        rounds+=1
        for group in groups([specs[k] for k in keys]):
            scalar=[s for s in group if s.measurement_method is MeasurementMethod.SEMANTIC and s.value_type!="structured"]
            if scalar:request(scalar,True)
            for spec in group:
                if spec.measurement_method is not MeasurementMethod.SEMANTIC or spec.value_type=="structured":
                    observed[spec.feature_id]=executor.execute(spec,case);attempted.add(spec.feature_id)
                    if spec.measurement_method is MeasurementMethod.HYBRID:
                        hybrid_executed.append(spec.feature_id)
                        observed[spec.feature_id]=replace(observed[spec.feature_id],extraction_metadata=
                            observed[spec.feature_id].extraction_metadata+(
                                MetadataItem("actual_measurement_protocol",VERSION),
                                MetadataItem("feature_contract_prompt_version",spec.prompt_version)))
    for key,spec in specs.items():
        if key not in observed:
            observed[key]=observation(spec,unknown("NOT_MEASURED_OR_UNRESOLVED; open-code absence is not false",
                                                  spec.model_version,spec.prompt_version))
    directory=Path(inputs.runtime.cache_directory)/"open_coding_cases"
    write_json(directory/(case.identity.case_id+".json"),
        {"protocol":VERSION,"case_id":case.identity.case_id,"before_hash":case.before_hash,
         "study_hash":study.freeze_hash,"open_codes":codes,"joint_model_calls":calls,"critical_rounds":rounds,
         "candidate_batches":candidate_batches,"hybrid_features_executed":len(hybrid_executed),"hybrid_feature_ids":hybrid_executed,
         "feature_code_matches":matches,"row_validation_errors":failures,"known_features":sum(o.status is ObservationStatus.KNOWN for o in observed.values()),
         "labels_do_not_update_frozen_rules":True})
    return tuple(observed[k] for k in specs)
