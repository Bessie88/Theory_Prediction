"""Run the existing compact predictor against frozen shared sources, in a new directory."""
import argparse, json
from pathlib import Path
from theory_discovery.core.schemas import require, stable_id
from theory_discovery.features.source_preparation import VERSION
from .rule_first_prediction import LABEL_PROTOCOL
from theory_discovery.core.storage import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--before-input", type=Path, help="OLD/NEW Before-only PredictionCase records; same frozen Theory.")
    p.add_argument("--cohort", choices=("OLD","NEW"), default="NEW")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--run", action="store_true")
    p.add_argument("--generate-edits", action="store_true", help="Optional concrete edits; default returns independent Action Labels.")
    args = p.parse_args()
    compiled = json.loads((args.prepared_dir/"compiled_sources.json").read_text())
    expected = compiled["freeze_hash"]; unhashed = dict(compiled); unhashed.pop("freeze_hash")
    require(compiled["protocol"]==VERSION,"Source preparation protocol changed; use the current source preparation entry")
    from .rule_first_prediction import validate_boolean_action_owners
    validate_boolean_action_owners(compiled["owners"])
    require(stable_id("shared_source_deployment", unhashed) == expected, "Compiled source hash mismatch")
    from hashlib import sha256
    from .compact_prediction_cli import load_sources, preflight
    from theory_discovery.core.workflow_contracts import FeatureSpecification
    old_owners, old_bridge, old_specs, registries, cases, _ = load_sources(compiled["source_dir"])
    del old_owners, old_bridge, old_specs
    before_input_hash=None
    if args.before_input:
        from theory_discovery.core.workflow_contracts import PredictionCase
        before_input_hash=sha256(args.before_input.read_bytes()).hexdigest()
        raw=json.loads(args.before_input.read_text())
        require(isinstance(raw,dict) and set(raw)=={"records"},"Before input contains only PredictionCase records")
        cases=[PredictionCase.from_dict(row) for row in raw["records"]]
        require(len({c.identity.case_id for c in cases})==len(cases),"Duplicate Before cases")
    require(0 < args.limit <= len(cases), "Invalid case count")
    require(args.output.resolve() not in {args.prepared_dir.resolve(), Path(compiled["source_dir"]).resolve()}, "Preserve sources/old results")
    args.output.mkdir(parents=True, exist_ok=True)
    specs = {s["feature_id"]:FeatureSpecification.from_dict(s) for s in compiled["specifications"]}
    from theory_discovery.annotation.compact_annotations import FORMAT
    manifest = dict(protocol=VERSION, annotation_protocol=FORMAT, prediction_protocol=LABEL_PROTOCOL, compiled_sha256=sha256((args.prepared_dir/"compiled_sources.json").read_bytes()).hexdigest(),
        source_freeze_hash=expected, cohort=args.cohort, before_input_sha256=before_input_hash,
        input_case_ids=[c.identity.case_id for c in cases],
        output_mode="EDITS" if args.generate_edits else "LABELS", max_rows=32, feature_labels="Y/N", source_contract_labels="Y/N/U", limit=args.limit, heldout_after_opened=False)
    frozen = args.output/"protocol.json"
    if frozen.exists(): require(json.loads(frozen.read_text()) == manifest, "Prediction protocol changed")
    else: write_json(frozen, manifest)
    if not args.run:
        report = preflight(compiled["owners"], compiled["bridge"], specs, cases[:args.limit], args.output, manifest)
        write_json(args.output/"preflight.json", report); print(json.dumps({k:v for k,v in report.items() if k!="budgets"})); return
    from theory_discovery.features.source_preparation_cli import backend
    from theory_discovery.annotation.compact_annotations import CompactRequests
    from .compact_rule_first import compact_prediction_case
    requests = CompactRequests(backend(args.output/"generation", 4096), backend(args.output/"labels", 512), args.output/"audit")
    results = []
    for case in cases[:args.limit]:
        result = compact_prediction_case(case, compiled["owners"], compiled["bridge"], specs,
            requests, args.output/"predictions", expected, registries, output_mode=manifest["output_mode"])
        results.append(result)
        write_json(args.output/"progress.json", dict(completed=len(results), requested=args.limit))
    from theory_discovery.evaluation.condition_evolution import freeze_prediction_snapshot
    if manifest["output_mode"]=="LABELS":freeze_prediction_snapshot(args.output,expected)
    write_json(args.output/"summary.json", dict(completed=len(results), heldout_after_opened=False,
        all_lazy_full_equal=all(r["comparison_passed"] for r in results)))


if __name__ == "__main__": main()
