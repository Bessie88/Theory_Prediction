"""Prepare old-cohort statistics after Theory freeze, with no old Action prediction."""
from collections import Counter, defaultdict
import fcntl
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from .action_annotator import COMPLETE_ACTION_STATUSES, annotate_cases
from .action_comparison import validate_protocol
from .annotation_data import prepare_annotation_cases
from .annotation_prompts import PROMPT_VERSION
from .compact_action_annotator import CompactActionBackend
from .condition_evolution import condition_identity, records_from_annotations, validate_bundle
from .schemas import require, stable_id
from .stage_interfaces import AnnotationBatch, LaterRecords, OutcomeRecords
from .storage import read_json, write_json
from .workflow_config import AnnotationSettings, ModelSettings

VERSION = "old-cohort-statistics-baseline-v1"


def measure_conditions(compiled, before, method, workspace):
    """Reuse the predictor's Before measurements, stopping before Action selection."""
    from .compact_annotations import CompactRequests
    from .compact_rule_first import CompactMeasurements
    from .rule_first_logic import value
    from .rule_first_prediction import case_values, settle_before, validate_boolean_action_owners
    from .source_preparation_cli import backend
    from .workflow_contracts import FeatureSpecification

    owners = [owner for owner in compiled["owners"] if owner["method"] == method]
    require(owners, "Frozen Theory has no baseline Conditions for this method")
    validate_boolean_action_owners(owners)
    specs = {s["feature_id"]: FeatureSpecification.from_dict(s) for s in compiled["specifications"]}
    requests = CompactRequests(backend(workspace / "measurement_transport", 4096),
                               backend(workspace / "label_transport", 512), workspace / "measurement_audit")
    records = []
    for case in before.records:
        requests.case_calls = 0
        measurements = CompactMeasurements(case, compiled["bridge"], specs, requests,
                                           workspace / "measurements", compiled["freeze_hash"])
        settle_before(owners, measurements)
        truths = defaultdict(set)
        for owner in owners:
            values = case_values(owner, measurements)
            for action in owner["actions"]:
                for variant in action["variants"]:
                    gate = variant["gate"]
                    cid = condition_identity(gate, owner["clauses"])
                    truth = value(gate, values)
                    known = truths[cid]
                    if truth is not None:
                        known.add(truth)
        groups = [dict(condition_id=cid, condition=next(iter(known)) if len(known) == 1 else None,
                       predicted_action_ids=[], action_mapping_complete=False)
                  for cid, known in sorted(truths.items())]
        # This is only the common record shape consumed by the join.
        records.append(dict(case_id=case.identity.case_id, before_hash=case.before_hash,
                            predictions={method: {"condition_groups": groups}}))
        write_json(workspace / "condition_progress.json", dict(completed=len(records), total=len(before.records)))
    return records


def prepare_baseline(config_path, protocol, method, output, expected_condition_freeze):
    """With explicit execution, prepare all OLD labels before the comparison stage."""
    from .source_preparation import VERSION as SOURCE_VERSION
    from .source_preparation_cli import backend

    validate_protocol(protocol)
    config = read_json(config_path)
    require(set(config) == {"compiled_sources", "before", "outcomes", "workspace"},
            "Baseline config needs compiled_sources, before, outcomes and workspace")
    paths = {k: (config_path.parent / v).resolve() for k, v in config.items()}
    workspace = paths.pop("workspace")
    require(workspace != output.resolve() and all(p.is_file() for p in paths.values())
            and all(not p.is_relative_to(workspace) for p in paths.values()),
            "Use a separate baseline workspace and preserve source files")
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compiled = read_json(paths["compiled_sources"])
        freeze = compiled["freeze_hash"]
        require(freeze == expected_condition_freeze, "Baseline and NEW Conditions must use the same Theory freeze")
        require(compiled["protocol"] == SOURCE_VERSION and freeze ==
                stable_id("shared_source_deployment", {k:v for k,v in compiled.items() if k != "freeze_hash"}),
                "Baseline must use verified frozen Conditions")
        before = LaterRecords.from_dict(read_json(paths["before"]))
        ids = [c.identity.case_id for c in before.records]
        require(ids and len(set(ids)) == len(ids), "Missing or duplicate OLD Before cases")
        # Pin the Before inputs and catalogue before opening any old After text.
        manifest = dict(version=VERSION, condition_freeze_hash=freeze, method=method,
                        action_protocol_hash=protocol.protocol_hash, case_ids=ids,
                        before_sha256=sha256(paths["before"].read_bytes()).hexdigest(),
                        outcomes_path=str(paths["outcomes"]))
        manifest_path = workspace / "manifest.json"
        if manifest_path.exists():
            require(read_json(manifest_path) == manifest, "Baseline identity changed; use a fresh workspace")
        else:
            write_json(manifest_path, manifest)
        condition_path = workspace / "before_conditions.json"
        if condition_path.exists():
            condition_freeze = read_json(condition_path)
            require(condition_freeze["hash"] == stable_id("baseline_before_conditions", condition_freeze["records"]),
                    "OLD Condition snapshot changed")
        else:
            conditions = measure_conditions(compiled, before, method, workspace)
            condition_freeze = dict(records=conditions, hash=stable_id("baseline_before_conditions", conditions))
            write_json(condition_path, condition_freeze)
        require([r["case_id"] for r in condition_freeze["records"]] == ids and
                [r["before_hash"] for r in condition_freeze["records"]] == [c.before_hash for c in before.records],
                "Old Condition snapshot does not cover the frozen Before cohort")
        outcomes_bytes = paths["outcomes"].read_bytes()
        outcomes_hash = sha256(outcomes_bytes).hexdigest()
        outcomes_manifest = workspace / "outcomes_manifest.json"
        if outcomes_manifest.exists():
            require(read_json(outcomes_manifest)["sha256"] == outcomes_hash, "OLD outcomes changed on resume")
        else:
            write_json(outcomes_manifest, {"sha256":outcomes_hash})
        cases = prepare_annotation_cases(OutcomeRecords.from_dict(json.loads(outcomes_bytes)), before)
        settings = AnnotationSettings(
            ModelSettings(enabled=True, model="qwen3.8-27b", endpoint="http://127.0.0.1/v1",
                          prompt_version=PROMPT_VERSION, temperature=0.0, max_tokens=4096, retries=2,
                          family="Qwen", revision="qwen3.8-27b"),
            VERSION, PROMPT_VERSION, same_family_rationale="Separate blind pass with the same model family.")
        runtime = SimpleNamespace(execution_authorized=True, models_authorized=True, configuration_id=VERSION)
        for round_no in (1, 2):
            annotations, audit = annotate_cases(cases, protocol, settings, runtime, ("Qwen",),
                lambda: CompactActionBackend(backend(workspace / f"action_transport_{round_no}", 4096)),
                checkpoint_dir=workspace / "action_checkpoints")
            write_json(workspace / f"action_audit_{round_no}.json", audit.to_dict())
            if all(a.annotation_status in COMPLETE_ACTION_STATUSES for a in annotations):
                break
        write_json(workspace / "old_annotations.json", AnnotationBatch(annotations).to_dict())
        result = records_from_annotations(condition_freeze["records"], annotations, protocol, period="OLD",
            condition_freeze_hash=freeze, prediction_snapshot_hash=None, method=method)
        result["metadata"].update(condition_measurement_hash=condition_freeze["hash"], old_actions_predicted=False)
        validate_bundle(result, "OLD")
        path = workspace / "old_records.json"
        write_json(path, result)
        incomplete = [a.identity.case_id for a in annotations if a.annotation_status not in COMPLETE_ACTION_STATUSES]
        write_json(workspace / "baseline_coverage.json", dict(total=len(ids), complete=len(ids)-len(incomplete),
                   incomplete_case_ids=incomplete,
                   statuses=dict(Counter(a.annotation_status.value for a in annotations))))
        return path
