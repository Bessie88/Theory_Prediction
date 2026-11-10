"""Preflight or run independent review of large NEW prediction-rate gaps."""
import argparse
from dataclasses import replace
import fcntl
from hashlib import sha256
import json
import re
from pathlib import Path

from theory_discovery.annotation._support.action_comparison import action_meaning_key
from .condition_evolution import analyze
from .condition_evolution_cli import frozen_predictions, validate_join
from .condition_shift_audit import audit_condition_shifts, audit_manifest, condition_catalog, prediction_gaps
from theory_discovery.core.config import LLMConfig, load_config
from theory_discovery.core.schemas import require, stable_id
from theory_discovery.core.stage_interfaces import OutcomeRecords
from theory_discovery.core.storage import Cache, read_json, write_json
from theory_discovery.core.wire_budget_client import WireBudgetQwenClient
from theory_discovery.core.workflow_contracts import ActionDefinition


class AuditJSONClient(WireBudgetQwenClient):
    """Use the generic chat-completions payload for an independently served model."""
    def _request(self, suffix, payload=None):
        if payload is not None:
            payload = {k:v for k,v in payload.items() if k != "chat_template_kwargs"}
        return super()._request(suffix, payload)


def _definitions(raw):
    definitions = {}
    for key, value in raw.items():
        action = ActionDefinition.from_dict(value)
        require(key == stable_id("condition_action", action_meaning_key(action)),
                "Observed Action definition does not match its canonical identity")
        definitions[key] = {k:value[k] for k in ("family","method","target_feature","direction","scope","description")}
    return definitions


def run_from_config(old, new, predictions, snapshot, config_path, output, *, run):
    """Run after evaluation with a pinned alternate model and a separate output directory."""
    config_path = Path(config_path).resolve()
    cfg = read_json(config_path)
    required = {"compiled_sources","outcomes","action_definitions","primary_model","review_llm","induction_config"}
    require(required <= set(cfg) <= required | {"threshold_pp"}, "Invalid condition shift audit configuration")
    path = lambda key: (config_path.parent / cfg[key]).resolve()
    compiled = read_json(path("compiled_sources"))
    freeze = stable_id("shared_source_deployment", {k:v for k,v in compiled.items() if k != "freeze_hash"})
    require(freeze == compiled["freeze_hash"] == new["metadata"]["condition_freeze_hash"] == snapshot["source_freeze_hash"],
            "Audit must use the same frozen Conditions")
    require(snapshot.get("actual_outcomes_read") is False
            and all(r.get("actual_after_opened") is False for r in predictions),
            "Audit must follow Before-only prediction freeze")
    validate_join(new, snapshot, predictions, "THEORY")
    analyze(old, new)
    catalogue = condition_catalog(compiled)
    require(set(catalogue) == {r["condition_id"] for r in new["records"]}, "Frozen Condition catalogue mismatch")
    raw_definitions = read_json(path("action_definitions"))
    definitions = _definitions(raw_definitions)
    outcomes = OutcomeRecords.from_dict(read_json(path("outcomes")))
    review_llm = LLMConfig(**cfg["review_llm"])
    existing = {cfg["primary_model"], *(s["model_version"] for s in compiled.get("specifications", []) if s.get("model_version"))}
    model_key = lambda value: re.sub(r"[^a-z0-9]", "", value.strip().casefold().rsplit("/",1)[-1])
    require(all(isinstance(m,str) and model_key(m) for m in existing) and isinstance(review_llm.model,str)
            and model_key(review_llm.model) and model_key(review_llm.model) not in {model_key(m) for m in existing},
            "Choose a different audit model from the original prediction model")
    require(0 <= review_llm.retries <= 5 and review_llm.max_tokens >= 1 and review_llm.max_prompt_chars >= 1
            and review_llm.context_length > review_llm.max_tokens + review_llm.context_safety_margin,
            "Invalid audit model limits")
    threshold = cfg.get("threshold_pp",10)
    gaps = prediction_gaps(new, threshold)
    output = Path(output).resolve()
    inputs = [config_path, *(path(k) for k in ("compiled_sources","outcomes","action_definitions","induction_config"))]
    require(all(not p.is_relative_to(output) for p in inputs), "Audit output must be separate from source inputs")
    input_hashes = {str(p):sha256(p.read_bytes()).hexdigest() for p in inputs}
    identity = audit_manifest(compiled,new,old,outcomes,raw_definitions,review_llm.model,threshold,
                              input_hashes[str(path("induction_config"))])
    identity.update(input_sha256=input_hashes, prediction_snapshot_hash=snapshot["snapshot_hash"])
    output.mkdir(parents=True, exist_ok=True)
    with (output/".audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = output/"input_manifest.json"
        if manifest.exists():
            require(read_json(manifest) == identity, "Audit inputs changed; use a fresh directory")
        else:
            require(all(p == output/".audit.lock" for p in output.iterdir()), "Use a separate audit output directory")
            write_json(manifest, identity)
        preflight = dict(state="PREFLIGHT", prediction_gaps=gaps, review_model=review_llm.model,
                         original_model=cfg["primary_model"], calls=0)
        write_json(output/"preflight.json",preflight)
        if not run:
            return preflight

        if gaps:
            require(review_llm.enabled and review_llm.base_url, "Enable the independent reviewer explicitly")
        client = None
        checked_cache = Cache(output/"checked_reviews")
        def request(task, payload, accept):
            def compute():
                nonlocal client
                if client is None:
                    client = AuditJSONClient(review_llm, Cache(output/"review_transport"),
                                             endpoint=review_llm.base_url)
                require(client.prompt_fits(task,payload,repair_reserve=True), "Audit request exceeds context budget")
                if not review_llm.tokenizer_path:
                    limit = review_llm.context_length - review_llm.max_tokens - review_llm.context_safety_margin - 1024
                    require(len((task+json.dumps(payload,ensure_ascii=False)).encode()) <= limit,
                            "Audit request exceeds conservative context budget; configure a tokenizer for longer input")
                client.reserve_work_calls((review_llm.retries+1)**2, "independent_condition_audit")
                return client.infer_checked(task,payload,accept)
            return checked_cache.get_or_compute("condition_review", dict(task=task,payload=payload,identity=identity),compute)

        report = audit_condition_shifts(old,new,catalogue,outcomes.records,definitions,request,threshold)
        report["review_model"] = review_llm.model
        report["new_review_calls"] = 0 if client is None else client.stats["calls"]
        report["induction_state"] = "PENDING" if report["uncovered_case_ids"] else "NO_CONFIRMED_MISSES"
        write_json(output/"condition_shift_audit.json", report)
        if report["uncovered_case_ids"]:
            from .uncovered_discovery_cli import discovery_pairs, induce_uncovered
            selected = set(report["uncovered_case_ids"])
            routes = {r["case_id"]:("UNCOVERED" if r["case_id"] in selected else "OUTSIDE_AUDIT_INDUCTION") for r in predictions}
            pairs, unavailable = discovery_pairs(predictions,routes,outcomes.records)
            require(not unavailable and len(pairs)==len(selected), "An audited uncovered case lacks an aligned pair")
            # A changed routing set gets a separate induction cache and output.
            workspace = output/"induction"/stable_id("audited_uncovered",sorted(selected))
            induction = replace(load_config(path("induction_config")), input_path="", input_format="jsonl",
                                output_dir=str(workspace/"theory"), cache_dir=str(workspace/"cache"), pre_context_keys=[])
            require(len(pairs) <= induction.max_observations and induction.allow_real_data and induction.llm.enabled,
                    "Enable and configure the induction model for audited uncovered cases")
            completion = workspace/"candidate_theory_directions.json"
            if not completion.exists():
                write_json(completion,dict(case_ids=sorted(selected),directions=induce_uncovered(pairs,induction,workspace),
                                           original_theory_freeze_hash=freeze))
            report["candidate_theory_directions"] = str(completion)
            report["induction_state"] = "CANDIDATES_RECORDED"
            write_json(output/"condition_shift_audit.json",report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-records",type=Path,required=True)
    parser.add_argument("--new-records",type=Path,required=True)
    parser.add_argument("--prediction-dir",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True,
        help="JSON paths for compiled_sources, outcomes, action_definitions, and induction_config, "
             "plus primary_model, review_llm, and optional threshold_pp.")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run",action="store_true",help="Call the configured review model and run induction for confirmed misses")
    args=parser.parse_args(argv)
    snapshot,predictions=frozen_predictions(args.prediction_dir,"THEORY")
    output=args.output.resolve()
    require(output != args.prediction_dir.resolve() and args.prediction_dir.resolve() not in output.parents
            and all(not p.resolve().is_relative_to(output) for p in (args.old_records,args.new_records)),
            "Preserve the frozen predictions and evaluation inputs")
    report=run_from_config(read_json(args.old_records),read_json(args.new_records),predictions,snapshot,args.config,output,run=args.run)
    print(json.dumps(dict(state=report["state"],prediction_gaps=len(report["prediction_gaps"]),
                         uncovered=len(report.get("uncovered_case_ids",[])))))


if __name__=="__main__":
    main()
