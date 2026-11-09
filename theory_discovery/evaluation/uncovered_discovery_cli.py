"""Induce candidate Theory directions from confirmed-uncovered NEW pairs."""

import argparse
from collections import Counter
from dataclasses import replace
import fcntl
from hashlib import sha256
import json
from pathlib import Path

from .condition_evolution import confirmed_uncovered
from .condition_evolution_cli import frozen_predictions
from theory_discovery.core.config import load_config
from theory_discovery.core.schemas import EditingPair, require
from theory_discovery.core.stage_interfaces import OutcomeRecords
from theory_discovery.core.storage import Cache, read_json, write_json


def partition(predictions):
    """Use only frozen Before decisions to route; short-circuited Features stay unknown."""
    routes = {}
    for record in predictions:
        branch = record["predictions"]["THEORY"]
        groups = branch["condition_groups"]
        if any(group["condition"] is True for group in groups):
            route = "COVERED"
        elif confirmed_uncovered(record):
            truths = record.get("theory_feature_truths", {})
            route = ("UNCOVERED_ALL_FEATURES_NO" if truths and
                     all(value is False for value in truths.values()) else "UNCOVERED")
        else:
            route = "ERROR_OR_UNRESOLVED"
        routes[record["case_id"]] = route
    return routes


def discovery_pairs(predictions, routes, outcomes):
    by_id = {item.identity.case_id: item for item in outcomes}
    require(len(by_id) == len(outcomes), "Duplicate outcome case IDs")
    require(set(by_id) <= set(routes), "Outcome case outside the frozen NEW population")
    pairs, unavailable = [], []
    for record in sorted(predictions, key=lambda r: routes[r["case_id"]] != "UNCOVERED_ALL_FEATURES_NO"):
        cid = record["case_id"]
        if not routes[cid].startswith("UNCOVERED"):
            continue
        outcome = by_id.get(cid)
        if outcome is None:
            unavailable.append({"case_id": cid, "reason": "MISSING_OUTCOME"})
            continue
        require(sha256(outcome.before_text.encode()).hexdigest() == record["before_hash"],
                "Outcome Before differs from frozen prediction: " + cid)
        if outcome.alignment_status != "ALIGNED" or outcome.after_text is None or not outcome.before_text.strip():
            unavailable.append({"case_id": cid, "reason": "NO_ALIGNED_BEFORE_AFTER"})
            continue
        pairs.append(EditingPair(cid, outcome.before_text, outcome.after_text))
    return pairs, unavailable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True, help="Existing NEW OutcomeRecords with aligned Before/After")
    parser.add_argument("--config", type=Path, required=True, help="Existing discovery Config, with an authorized endpoint for --run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Run fresh open coding through Theory synthesis")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    prediction_dir = args.prediction_dir.resolve()
    require(out != prediction_dir and prediction_dir not in out.parents and out not in prediction_dir.parents,
            "Use a separate output directory outside the prediction tree")
    protocol = read_json(prediction_dir / "protocol.json")
    require(protocol.get("cohort") == "NEW", "Uncovered induction requires the frozen NEW cohort")
    snapshot, predictions = frozen_predictions(prediction_dir, "THEORY")
    require(snapshot.get("actual_outcomes_read") is False, "Prediction snapshot must be frozen before reading After")
    require(len(predictions) == len({r["case_id"] for r in predictions})
            and set(snapshot["case_ids"]) == {r["case_id"] for r in predictions},
            "Prediction case IDs differ from the frozen manifest")
    require(all(r.get("actual_after_opened") is False and r.get("output_mode") == "LABELS"
                and r.get("source_logic_freeze_hash") == snapshot["source_freeze_hash"] for r in predictions),
            "Use Before-only predictions from the same frozen Theory")
    routes = partition(predictions)
    outcome_bytes = args.outcomes.read_bytes()
    outcomes = OutcomeRecords.from_dict(json.loads(outcome_bytes))
    pairs, unavailable = discovery_pairs(predictions, routes, outcomes.records)
    config = replace(load_config(args.config), input_path="", input_format="jsonl",
                     output_dir=str(out / "induction"), cache_dir=str(out / "cache"),
                     pre_context_keys=[])
    require(len(pairs) <= config.max_observations, "Discovery exceeds configured observation limit")
    if args.run and pairs:
        require(config.allow_real_data and config.llm.enabled, "Enable real data and a discovery model in the supplied config")
    identity = dict(protocol="uncovered-discovery-v1", source_freeze_hash=snapshot["source_freeze_hash"],
                    prediction_snapshot_hash=snapshot["snapshot_hash"],
                    outcomes_sha256=sha256(outcome_bytes).hexdigest(), config=config.to_dict())
    report = dict(routes=routes, counts=dict(Counter(routes.values())),
                  discovery_case_ids=[pair.case_id for pair in pairs], unavailable_outcomes=unavailable,
                  deletions=sum(pair.after_text == "" for pair in pairs),
                  unchanged=sum(pair.after_text == pair.before_text for pair in pairs))

    out.mkdir(parents=True, exist_ok=True)
    with (out / ".discovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = out / "input_manifest.json"
        if manifest.exists():
            require(read_json(manifest) == identity, "Discovery inputs changed; use a fresh output directory")
        else:
            require(all(path == out / ".discovery.lock" for path in out.iterdir()),
                    "Use a fresh discovery output directory")
            write_json(manifest, identity)
        write_json(out / "routing.json", report)
        result = out / "candidate_theory_directions.json"
        if args.run and not result.exists():
            directions = []
            if pairs:
                from theory_discovery.induction.parallel_discovery import run_framework
                from theory_discovery.core.response_recovery import RecoverableQwenJSONClient
                backend = RecoverableQwenJSONClient(config.llm, Cache(out / "transport"))
                backend.verify_model()
                theory = run_framework(pairs, config, backend, retain_unchanged=True)["theory"]
                directions = [
                    {key: row[key] for key in ("branch_id", "situation", "source_rules", "action_options",
                                              "evidence_citations", "uncertainty")}
                    for row in theory["branches"]
                ]
            write_json(result, dict(state="CANDIDATE_DIRECTIONS" if directions else
                                    "NO_SUPPORTED_DIRECTIONS" if pairs else "NO_ELIGIBLE_PAIRS",
                                    theory_path="induction/theory.json" if pairs else None,
                                    original_theory_freeze_hash=snapshot["source_freeze_hash"],
                                    directions=directions, discovered_from_new_cohort=True))
        print(json.dumps(dict(eligible=len(pairs), routes=report["counts"],
                              stage="INDUCTION_COMPLETE" if result.exists() else "PREFLIGHT",
                              output=str(out))))


if __name__ == "__main__":
    main()
