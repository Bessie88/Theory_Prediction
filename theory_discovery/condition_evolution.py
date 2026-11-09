"""Direct Condition/Action counts and old/new proportions from existing labels."""
from collections import defaultdict
from copy import deepcopy
from .schemas import require,stable_id
from .action_comparison import compare_condition_actions

VERSION="condition-action-rates-v2"
RATE_REPORT_VERSION="condition-action-rates-v3"
MIN_TEXTS_PER_PERIOD=5


def condition_identity(gate,clauses):
    """Same full predicate expression, not one feature name or one Action."""
    def source(node):
        op=node["op"]
        if op=="clause":
            c=clauses[node["clause_id"]]
            return dict(op=op,predicate=c["predicate_text"],scope=c["truth_scope"])
        if op=="not":return dict(op=op,arg=source(node["arg"]))
        if op in ("and","or"):
            parts=[source(n) for n in node["args"]]
            return dict(op=op,args=sorted(parts,key=lambda n:stable_id("term",n)))
        return deepcopy(node)
    return stable_id("complete_condition",source(gate))


def rate_difference(old_k,old_n,new_k,new_n):
    """Compare proportions only when each period has enough usable texts."""
    if min(old_n,new_n)<MIN_TEXTS_PER_PERIOD:
        return dict(old_rate=None,new_rate=None,delta=None,percentage_point_difference=None)
    old_rate=old_k/old_n
    new_rate=new_k/new_n
    delta=new_rate-old_rate
    return dict(old_rate=old_rate,new_rate=new_rate,delta=delta,percentage_point_difference=delta*100)


def validate_bundle(bundle,period):
    require(isinstance(bundle,dict) and set(bundle)>={"metadata","records"},"Use a frozen Condition record bundle")
    meta=bundle["metadata"]
    for key in ("condition_freeze_hash","action_protocol_hash"):
        require(isinstance(meta.get(key),str) and meta[key],"Missing frozen identity: "+key)
    if period=="NEW":require(meta.get("prediction_frozen_before_outcomes") is True,"New predictions must be frozen before outcomes are joined")
    require(meta.get("condition_labels_before_only") is True,"Condition groups cannot be derived from After")
    require(meta.get("population_complete") is True,"Supply all cases, including hits, misses and unmatched cases; not mismatches alone")
    require(isinstance(meta.get("case_ids"),list) and len(set(meta["case_ids"]))==len(meta["case_ids"]),"Freeze the complete case manifest")
    require(set(meta["case_ids"])=={r["case_id"] for r in bundle["records"]},"Cases are missing from the frozen population")
    seen=set()
    for row in bundle["records"]:
        require(row.get("period")==period,"Wrong period")
        for key in ("case_id","condition_id","before_hash","evidence_ref","annotation_ref"):
            require(isinstance(row.get(key),str) and row[key],"Missing row provenance: "+key)
        key=row["case_id"],row["condition_id"]
        require(key not in seen,"Duplicate Condition/case; consolidate frozen rule predictions before analysis")
        seen.add(key)
        require(row["condition"] is None or type(row["condition"]) is bool,"Condition truth must be Boolean, or null for an unavailable label")
        require(type(row["observed_complete"]) is bool,"Declare whether observed Action labels are complete")
        require(row.get("attribution_scope") in ("CONDITION_TARGET","CASE_ASSOCIATION"),"Declare target-level attribution or case-level association")
        for field in ("predicted_action_ids","observed_action_ids"):
            require(isinstance(row[field],list) and len(set(row[field]))==len(row[field]) and all(isinstance(v,str) and v for v in row[field]),"Action labels must be comparable ID sets")
        require(all(v is None or type(v) is bool for v in row.get("context_labels",{}).values()),"Subconditions must be frozen Before labels")
    return seen


def confirmed_uncovered(record):
    branch = record["predictions"]["THEORY"]
    groups = branch["condition_groups"]
    return (branch.get("status") == "NOT_COVERED" and bool(groups)
            and all(group["condition"] is False and not group.get("condition_measurement_conflict")
                    for group in groups)
            and not branch.get("prediction_errors") and not record.get("feature_annotation_errors"))


def complete(row):
    return row["condition"] is True and row["observed_complete"] and "OTHER" not in row["observed_action_ids"]


def analyze(old,new):
    """Count every labelled case in its Condition group, without text backtracking."""
    validate_bundle(old,"OLD");validate_bundle(new,"NEW")
    om,nm=old["metadata"],new["metadata"]
    require(om.get("method","THEORY")==nm.get("method","THEORY"),"Do not mix Theory and policy labels across periods")
    for key in ("condition_freeze_hash","action_protocol_hash"):
        require(om[key]==nm[key],"Old/new label definitions differ: "+key)
    old_cases={r["case_id"] for r in old["records"]};new_cases={r["case_id"] for r in new["records"]}
    require(not old_cases&new_cases,"Old/new case manifests must be disjoint")
    require(not ({r["before_hash"] for r in old["records"]}&{r["before_hash"] for r in new["records"]}),"Old/new Before texts overlap")
    condition_ids={r["condition_id"] for r in old["records"]}
    require(condition_ids=={r["condition_id"] for r in new["records"]},"Old/new labels must cover the same frozen Conditions")
    for bundle,cases in ((old,old_cases),(new,new_cases)):
        require(len(bundle["records"])==len(cases)*len(condition_ids),"Missing existing Condition/case labels")
    grouped=defaultdict(lambda:dict(OLD=[],NEW=[]))
    for bundle,period in ((old,"OLD"),(new,"NEW")):
        for row in bundle["records"]:grouped[row["condition_id"]][period].append(row)
    conditions=[];comparisons=[]
    for cid,periods in sorted(grouped.items()):
        usable={p:[r for r in rows if complete(r)] for p,rows in periods.items()}
        scores=[compare_condition_actions(r["predicted_action_ids"],r["observed_action_ids"],
            condition=r["condition"],observed_complete=r["observed_complete"],predicted_complete=r.get("predicted_complete",True))
            for r in periods["NEW"] if r["condition"] is True]
        yes=sum(s["passed"] is True for s in scores);no=sum(s["passed"] is False for s in scores)
        enough_texts=min(len(usable["OLD"]),len(usable["NEW"]))>=MIN_TEXTS_PER_PERIOD
        conditions.append(dict(condition_id=cid,action_rate_status="COMPARABLE" if enough_texts else "INSUFFICIENT_TEXTS",
            any_hit_passes=yes,any_hit_failures=no,
            unscored_predictions=len(scores)-yes-no,any_hit_rate=yes/(yes+no) if yes+no else None,
            counts={p:dict(total=len(rows),condition_true=sum(r["condition"] is True for r in rows),
                condition_false=sum(r["condition"] is False for r in rows),
                condition_unavailable=sum(r["condition"] is None for r in rows),
                usable_observed=len(usable[p]),
                incomplete_observed=sum(r["condition"] is True and not complete(r) for r in rows)) for p,rows in periods.items()}))
        actions=sorted({a for rows in periods.values() for r in rows for a in r["predicted_action_ids"]+r["observed_action_ids"] if a!="OTHER"})
        for action in actions:
            a,b=usable["OLD"],usable["NEW"]
            counts=dict(old_success=sum(action in r["observed_action_ids"] for r in a),old_n=len(a),
                new_success=sum(action in r["observed_action_ids"] for r in b),new_n=len(b))
            comparisons.append(dict(condition_id=cid,action_id=action,**counts,
                **rate_difference(counts["old_success"],len(a),counts["new_success"],len(b))))
    label_coverage={}
    for period,bundle in (("OLD",old),("NEW",new)):
        rows=bundle["records"]
        matched={r["case_id"] for r in rows if r["condition"] is True}
        incomplete={r["case_id"] for r in rows if r["condition"] is True and not complete(r)}
        unavailable={r["case_id"] for r in rows if r["condition"] is None}
        label_coverage[period]=dict(matched=len(matched),complete=len(matched-incomplete),
            incomplete_case_ids=sorted(incomplete),condition_unavailable_case_ids=sorted(unavailable))
    coverage_status=("PARTIAL" if any(v["incomplete_case_ids"] or v["condition_unavailable_case_ids"]
        for v in label_coverage.values()) else "COMPLETE")
    return dict(protocol=RATE_REPORT_VERSION,method=om.get("method","THEORY"),primary_criterion="ANY_ACTION_HIT_WITHIN_ONE_CONDITION",
        label_coverage=label_coverage,coverage_status=coverage_status,
        minimum_texts_per_period=MIN_TEXTS_PER_PERIOD,conditions=conditions,action_distributions=comparisons,
        case_counts=dict(OLD=len(old_cases),NEW=len(new_cases)),
        rate_denominator="Cases satisfying this Condition with complete observed Action labels; each Action counted once per case",
        condition_freeze_hash=om["condition_freeze_hash"],action_protocol_hash=om["action_protocol_hash"],
        old_prediction_snapshot_hash=om.get("prediction_snapshot_hash"),new_prediction_snapshot_hash=nm.get("prediction_snapshot_hash"),
        no_model_calls=True,existing_labels_reused=True,old_source_unchanged=True)


def groups_from_decisions(owners,decisions,registry):
    """Freeze same-Condition Action alternatives before any actual edit is read."""
    from .action_comparison import action_meaning_key
    owner_map={o['owner_id']:o for o in owners}
    definitions={a.source_action_id:a.definition for a in getattr(registry,'source_actions',()) if a.definition is not None}
    groups={}
    for d in decisions:
        cid=condition_identity(d['gate'],owner_map[d['owner_id']]['clauses'])
        d['condition_id']=cid
        entry=groups.setdefault(cid,dict(condition_id=cid,condition=d['before_partial_truth'],predicted_action_ids=[],
            action_mapping_complete=True,source_rule_ids=[],source_action_ids=[],label_ids=[],rule_predictions=[]))
        entry.setdefault('measurement_truths',[]).append(d['before_partial_truth'])
        definition=definitions.get(d['source_action_id'])
        try:key=stable_id('condition_action',action_meaning_key(definition)) if definition is not None else None
        except AttributeError:key=None
        if key is None:entry['action_mapping_complete']=False
        elif key not in entry['predicted_action_ids']:entry['predicted_action_ids'].append(key)
        for field,value in (('source_rule_ids',d['source_rule_id']),('source_action_ids',d['source_action_id']),('label_ids',d['label_id'])):
            if value not in entry[field]:entry[field].append(value)
        entry['rule_predictions'].append(deepcopy(d))
    for entry in groups.values():
        known={v for v in entry['measurement_truths'] if v is not None}
        entry['condition_measurement_conflict']=len(known)>1
        entry['condition']=next(iter(known)) if len(known)==1 else None
    return [groups[k] for k in sorted(groups)]


def freeze_prediction_snapshot(directory,source_freeze_hash):
    from pathlib import Path
    from hashlib import sha256
    from .storage import write_json
    directory=Path(directory);files={str(p.relative_to(directory)):sha256(p.read_bytes()).hexdigest() for p in sorted((directory/'predictions/cases').glob('*.json'))}
    require(files,'No completed prediction records to freeze')
    for name in files:
        record=__import__('json').loads((directory/name).read_text())
        require(record.get('output_mode')=='LABELS','Freeze the independent label output before outcome comparison')
        require(record.get('source_logic_freeze_hash')==source_freeze_hash,'Prediction source differs from frozen Conditions')
        require(all('condition_groups' in branch for branch in record['predictions'].values()),'Old predictions need Condition grouping before this evaluation')
    run_protocol=directory/'protocol.json'
    intended=__import__('json').loads(run_protocol.read_text()).get('input_case_ids') if run_protocol.exists() else None
    completed=[__import__('json').loads((directory/name).read_text())['case_id'] for name in files]
    result=dict(protocol=VERSION,source_freeze_hash=source_freeze_hash,prediction_files_sha256=files,
        input_population_complete=set(completed)==set(intended) if intended is not None else True,
        case_ids=sorted(completed),
        next_stage='INDEPENDENT_OBSERVED_ACTION_JOIN',actual_outcomes_read=False)
    result['snapshot_hash']=stable_id('condition_prediction_snapshot',result)
    path=directory/'condition_prediction_snapshot.json'
    if path.exists():require(__import__('json').loads(path.read_text())==result,'Prediction snapshot changed; use a new version directory')
    else:write_json(path,result)
    return result


def records_from_annotations(predictions,annotations,protocol,*,period,condition_freeze_hash,prediction_snapshot_hash,method='THEORY'):
    """Join existing independent annotations after prediction freeze; no new annotator/LLM.

    Case-level operations are explicitly associations, not proven Condition-target attribution.
    """
    from .action_comparison import validate_protocol,action_meaning_key,novel_action_definition
    validate_protocol(protocol)
    definitions={a.action_id:a for a in protocol.actions}
    catalog_semantics={stable_id('condition_action',action_meaning_key(a)) for a in protocol.actions if a.action_id!='OTHER'}
    annotations=tuple(annotations)
    actual={a.identity.case_id:a for a in annotations}
    case_ids={p['case_id'] for p in predictions}
    missing=case_ids-set(actual)
    skippable={p['case_id'] for p in predictions if p['case_id'] in missing
               and period=='NEW' and method=='THEORY' and confirmed_uncovered(p)}
    require(len(actual)==len(annotations) and set(actual)<=case_ids and missing==skippable,
            'Prediction/annotation cases differ')
    rows=[]
    for p in predictions:
        annotation=actual.get(p['case_id'])
        require(annotation is None or annotation.protocol_hash==protocol.protocol_hash,'Observed Action protocol differs')
        ids=[];unresolved=False
        for action in annotation.observed_actions if annotation is not None else ():
            if action.action_id=='OTHER':unresolved=True;continue
            definition=definitions.get(action.action_id) or novel_action_definition(action)
            require(definition is not None,'Observed action has no independent canonical definition')
            key=stable_id('condition_action',action_meaning_key(definition))
            if key not in ids:ids.append(key)
        known=annotation is not None and annotation.annotation_status.value in ('ANNOTATED','NO_CHANGE') and not unresolved
        if known and annotation.annotation_status.value=='NO_CHANGE':
            unchanged=[a for a in protocol.actions if a.family=='no_change']
            require(len(unchanged)<=1,'No-change Action definition is ambiguous')
            meaning=(action_meaning_key(unchanged[0]) if unchanged else
                     ('no change','keep existing wording','text','no change','entire text'))
            ids=[stable_id('condition_action',meaning)]
        for group in p['predictions'][method]['condition_groups']:
            rows.append(dict(period=period,case_id=p['case_id'],condition_id=group['condition_id'],before_hash=p['before_hash'],
                condition=group['condition'],predicted_action_ids=group['predicted_action_ids'],predicted_complete=group['action_mapping_complete'] and set(group['predicted_action_ids'])<=catalog_semantics,
                observed_action_ids=sorted(ids)+(['OTHER'] if unresolved else []),observed_complete=known,
                attribution_scope='CASE_ASSOCIATION',evidence_ref='frozen_prediction:'+p['case_id']+':'+group['condition_id'],
                annotation_ref=annotation.annotation_id if annotation else 'skipped_uncovered:'+p['case_id'],
                editor_id=annotation.identity.editor.editor_id if annotation else None,
                sampling_unit_id=annotation.identity.revision_id if annotation else None))
    return dict(metadata=dict(method=method,condition_freeze_hash=condition_freeze_hash,action_protocol_hash=protocol.protocol_hash,
        prediction_snapshot_hash=prediction_snapshot_hash,prediction_frozen_before_outcomes=period=="NEW",
        existing_labels_reused=True,case_ids=sorted(p["case_id"] for p in predictions),
        condition_labels_before_only=True,population_complete=True),records=rows)
