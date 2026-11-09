"""Count existing old/new Condition/Action labels and compare their proportions."""
import argparse,json
from pathlib import Path
from hashlib import sha256
from .condition_evolution import analyze, validate_bundle
from theory_discovery.core.schemas import require,stable_id
from theory_discovery.core.storage import write_json


def frozen_predictions(directory,method):
    """Read an existing prediction freeze when an annotation join is requested."""
    snapshot=json.loads((directory/'condition_prediction_snapshot.json').read_text())
    identity=dict(snapshot);identity.pop('snapshot_hash')
    require(stable_id('condition_prediction_snapshot',identity)==snapshot['snapshot_hash'],'Prediction snapshot identity changed')
    require(snapshot.get('input_population_complete') is True,'Prediction cohort is incomplete')
    predictions=[]
    for name,digest in snapshot['prediction_files_sha256'].items():
        path=directory/name
        require(sha256(path.read_bytes()).hexdigest()==digest,'Prediction changed before outcome comparison')
        record=json.loads(path.read_text());require(method in record['predictions'],'Missing prediction branch')
        predictions.append(record)
    return snapshot,predictions


def validate_join(new,snapshot,predictions,method):
    require(new['metadata']['prediction_snapshot_hash']==snapshot['snapshot_hash'],'Joined outcomes refer to different predictions')
    require(new['metadata']['condition_freeze_hash']==snapshot['source_freeze_hash'],'Condition freeze differs from predictions')
    frozen_rows={(r['case_id'],g['condition_id']):(r,g) for r in predictions for g in r['predictions'][method]['condition_groups']}
    require(len(new['records'])==len(frozen_rows),'The outcome join must retain every frozen Condition/case row')
    for row in new['records']:
        key=row['case_id'],row['condition_id'];require(key in frozen_rows,'Unknown joined Condition/case')
        record,group=frozen_rows[key]
        require(row['before_hash']==record['before_hash'] and row['condition'] is group['condition'],'Outcome join changed frozen Before labels')
        require(set(row['predicted_action_ids'])==set(group['predicted_action_ids']),'Outcome join changed predicted Actions')
        require(not row.get('predicted_complete',True) or group['action_mapping_complete'],'Cannot upgrade unresolved Action mappings')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    old_input=p.add_mutually_exclusive_group(required=True)
    old_input.add_argument('--old-records',type=Path,help='Existing old Condition/Action labels; used directly')
    old_input.add_argument('--prepare-old-baseline',type=Path,metavar='CONFIG',
        help='Explicitly prepare and resume OLD Condition/actual Action labels before statistics')
    new_input=p.add_mutually_exclusive_group(required=True)
    new_input.add_argument('--new-records',type=Path,help='Existing new Condition/Action labels; used directly')
    new_input.add_argument('--new-annotations',type=Path,help='Existing independent annotations to join to existing predictions')
    p.add_argument('--action-protocol',type=Path,help='Frozen Action protocol for annotation joins or baseline preparation')
    p.add_argument('--prediction-dir',type=Path,help='Existing prediction freeze; required only with --new-annotations')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=('THEORY','NPOV'),default='THEORY')
    p.add_argument('--suggest-theory',action='store_true',help='Optionally ask the LLM for unapplied high-level directions after a large rate difference')
    p.add_argument('--theory-context',type=Path,help='Existing compiled_sources.json; required only with --suggest-theory')
    p.add_argument('--suggestion-threshold-pp',type=float,default=10,help='Absolute percentage-point difference for advice, e.g. 5 or 10 (default 10)')
    args=p.parse_args()
    require(not args.output.exists(),'Use a fresh output directory; preserve prior results')
    require(not args.new_annotations or (args.action_protocol and args.prediction_dir),
            'Existing annotations need their Action protocol and prediction directory')
    require(not args.suggest_theory or (args.theory_context and args.action_protocol and args.method=='THEORY'),
            'Theory advice needs compiled Theory context and the Action protocol; official policy is not rewritten')
    require(not args.prepare_old_baseline or args.action_protocol, 'Baseline preparation needs the frozen Action protocol')
    paths=[args.new_records or args.new_annotations]
    if args.old_records: paths.append(args.old_records)
    if args.prepare_old_baseline: paths.append(args.prepare_old_baseline)
    if args.action_protocol:paths.append(args.action_protocol)
    if args.suggest_theory:paths.append(args.theory_context)
    before={str(path):sha256(path.read_bytes()).hexdigest() for path in paths}
    snapshot,predictions=frozen_predictions(args.prediction_dir,args.method) if args.prediction_dir else (None,None)
    old=json.loads(args.old_records.read_text()) if args.old_records else None
    if args.new_records:new=json.loads(args.new_records.read_text())
    else:
        from theory_discovery.annotation.annotation_contracts import FrozenActionProtocol
        from theory_discovery.core.stage_interfaces import AnnotationBatch
        from .condition_evolution import records_from_annotations
        protocol=FrozenActionProtocol.from_dict(json.loads(args.action_protocol.read_text()))
        annotations=AnnotationBatch.from_dict(json.loads(args.new_annotations.read_text())).annotations
        new=records_from_annotations(predictions,annotations,protocol,period='NEW',
            condition_freeze_hash=snapshot['source_freeze_hash'],prediction_snapshot_hash=snapshot['snapshot_hash'],method=args.method)
    require(new['metadata'].get('method','THEORY')==args.method,'Wrong method in labels')
    if snapshot:validate_join(new,snapshot,predictions,args.method)
    validate_bundle(new,'NEW')
    if args.prepare_old_baseline:
        from theory_discovery.annotation.action_comparison import FrozenActionProtocol
        from .baseline_preparation import prepare_baseline
        protocol=FrozenActionProtocol.from_dict(json.loads(args.action_protocol.read_text()))
        require(protocol.protocol_hash==new['metadata']['action_protocol_hash'],
                'Baseline protocol differs from the NEW labels')
        args.old_records=prepare_baseline(args.prepare_old_baseline,protocol,args.method,args.output,
                                          new['metadata']['condition_freeze_hash'])
        before[str(args.old_records)]=sha256(args.old_records.read_bytes()).hexdigest()
        old=json.loads(args.old_records.read_text())
    report=analyze(old,new)
    for path,digest in before.items():require(sha256(Path(path).read_bytes()).hexdigest()==digest,'Labels changed during analysis')
    args.output.mkdir(parents=True)
    write_json(args.output/'condition_evaluation.json',report)
    write_json(args.output/'input_manifest.json',dict(input_sha256=before,models_called=0 if not args.prepare_old_baseline else None,gpu_jobs_submitted=0,
        existing_labels_reused=not bool(args.prepare_old_baseline), baseline_prepared=bool(args.prepare_old_baseline)))
    usage={};suggestions=None
    if args.suggest_theory:
        from theory_discovery.annotation.action_comparison import FrozenActionProtocol
        from .theory_change_directions import suggest_directions
        from theory_discovery.features.source_preparation_cli import backend
        compiled=json.loads(args.theory_context.read_text())
        protocol=FrozenActionProtocol.from_dict(json.loads(args.action_protocol.read_text()))
        advice_backend=None
        def create_backend():
            nonlocal advice_backend
            advice_backend=backend(args.output/'suggestion_transport',4096)
            return advice_backend
        try:
            suggestions=suggest_directions(report,compiled,protocol,create_backend,args.suggestion_threshold_pp)
            write_json(args.output/'theory_change_suggestions.json',suggestions)
        finally:
            usage=dict(advice_backend.client.stats) if advice_backend is not None else {}
            write_json(args.output/'input_manifest.json',dict(input_sha256=before,models_called=usage.get('calls',0),
                gpu_jobs_submitted=0,existing_labels_reused=True,original_theory_modified=False,
                theory_advice_completed=suggestions is not None,
                baseline_prepared=bool(args.prepare_old_baseline)))
    for path,digest in before.items():require(sha256(Path(path).read_bytes()).hexdigest()==digest,'Inputs changed during analysis or advice')
    print(json.dumps(dict(conditions=len(report['conditions']),action_comparisons=len(report['action_distributions']),
        coverage_status=report['coverage_status'],
        theory_advice_requests=suggestions['requests'] if suggestions else 0,
        models_called=usage.get('calls',0) if not args.prepare_old_baseline else None)))


if __name__=='__main__':main()
