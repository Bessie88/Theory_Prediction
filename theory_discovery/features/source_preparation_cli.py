"""Prepare compact sources; no model calls without --run. Prediction is a separate command."""
from pathlib import Path
from hashlib import sha256
import argparse, json, os
from theory_discovery.core.schemas import require, stable_id
from theory_discovery.core.storage import Cache, write_json
from ._support.source_preparation import VERSION, source_packets, prepare, compile_sources, plan_shared_measurements


def read_sources(source):
    source = Path(source)
    names = ("theories.json", "policy_plan.json")
    records = {n:json.loads((source/n).read_text()) for n in names}
    bundles = [records["theories.json"], records["policy_plan.json"]["canonical_bundle"]]
    hashes = {n:sha256((source/n).read_bytes()).hexdigest() for n in names}
    return source_packets(bundles), hashes


def backend(output, tokens=8192):
    from theory_discovery.core.config import LLMConfig, configured_tokenizer_path
    from theory_discovery.core.wire_budget_client import WireBudgetQwenClient
    from ._support.semantic_features import LegacyJSONFeatureBackend, ModelPermit
    endpoint = os.environ["SIMULATION_MODEL_BASE_URL"]
    config = LLMConfig(enabled=True, base_url=endpoint, model="qwen3.8-27b", temperature=0,
        max_tokens=tokens, max_prompt_chars=180000, timeout_seconds=900, retries=2,
        tokenizer_path=configured_tokenizer_path())
    client = WireBudgetQwenClient(config, Cache(output/"transport"))
    client.verify_model()
    return LegacyJSONFeatureBackend(client, "qwen3.8-27b", VERSION, ModelPermit(True, True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cached-only", action="store_true", help="Compile using existing checked responses only")
    args = parser.parse_args()
    require(not args.cached_only or args.run, "Cached-only mode requires --run")
    require(args.output.resolve() != args.source_dir.resolve(), "Preserve old results")
    args.output.mkdir(parents=True, exist_ok=True)
    packets, hashes = read_sources(args.source_dir)
    # No heldout Before/After, prior labels or predictions are loaded by this preparation stage.
    package = Path(__file__).resolve().parents[1]
    identity = dict(protocol=VERSION, source=str(args.source_dir.resolve()), input_hashes=hashes,
        code_sha256={str(p.relative_to(package)):sha256(p.read_bytes()).hexdigest()
                     for p in sorted(package.rglob("*.py"))})
    pinned = args.output/"source_manifest.json"
    if pinned.exists(): require(json.loads(pinned.read_text()) == identity, "Prepared source/code changed; use a new output")
    else: write_json(pinned, identity)
    write_json(args.output/"source_packets.json", packets)
    report = dict(owners=len(packets), source_rules=sum(len(p["rule_ids"]) for p in packets),
        paragraphs=sum(len(p["paragraphs"]) for p in packets), source_only=True,
        heldout_inputs_opened=False, models_called=0, generalization_stage="JOINT_THEORY_BOUNDARY",
        added_llm_stages=0, feature_input_source="JOINT_THEORY_BOUNDARY_RULES", state="PREPARED_NOT_EXECUTED")
    report["policy_context_owners"]=sum(p["method"]=="NPOV" and not p["actions_read_only"] for p in packets)
    report["predictive_owners"]=len(packets)-report["policy_context_owners"]
    if not args.run:
        write_json(args.output/"preflight.json", report)
        print(json.dumps(report)); return
    b = None if args.cached_only else backend(args.output)
    cache = Cache(args.output/"checkpoints"); calls = []
    def request(stage, prompt, payload, accept):
        key = dict(identity=identity, stage=stage, prompt=prompt, payload=payload)
        def compute():
            require(b is not None, "Checked response missing in cached-only mode: " + stage)
            require(b.client.prompt_fits(prompt, payload, repair_reserve=True),
                    "Source request exceeds full context; no truncation: " + stage)
            return b.complete_checked(prompt, payload, accept)
        result = cache.get_or_compute(stage, key, compute)
        calls.append(dict(stage=stage, request_hash=stable_id("source_request", key)))
        write_json(args.output/"progress.json", dict(completed_requests=len(calls), stages=calls,
            cache_hits=cache.hits, cache_misses=cache.misses, source_only=True))
        return result
    request.fits = lambda prompt,payload: b is None or b.client.prompt_fits(prompt, payload, repair_reserve=True)
    prepared = prepare(packets, request)
    write_json(args.output/"shared_sources.json", prepared)
    write_json(args.output/"joint_theory_boundary.json", dict(protocol=VERSION, source_freeze_hash=prepared["freeze_hash"],
        generalization_stage=prepared["generalization_stage"], integrated_theories=prepared["summaries"], source_only=True))
    measurements = plan_shared_measurements(packets, prepared, request)
    write_json(args.output/"shared_measurements.json", measurements)
    compiled = compile_sources(packets, prepared, measurements["mappings"])
    compiled["source_dir"] = str(args.source_dir.resolve())
    compiled["specifications"] = measurements["specifications"]
    compiled.pop("freeze_hash")
    compiled["freeze_hash"] = stable_id("shared_source_deployment", compiled)
    write_json(args.output/"compiled_sources.json", compiled)
    report.update(state="SOURCE_PREPARATION_COMPLETE_REVIEW_REQUIRED", model_usage=dict(b.client.stats) if b else {},
        questions_before_dedup=len(prepared["aliases"]), questions_after_dedup=len(prepared["catalog"]),
        measurement_programs=len(measurements["specifications"]),
        measurement_methods={m:sum(s["measurement_method"]==m for s in measurements["specifications"])
                             for m in ("CODE", "HYBRID", "SEMANTIC")},
        predictive_owners=len(compiled["owners"]),policy_context_owners=len(compiled["context_sources"]),
        semantic_quality="MODEL_UNREVIEWED", speedup_measured=False, predictions_started=False)
    report.pop("models_called")
    write_json(args.output/"summary.json", report)
    _, after = read_sources(args.source_dir)
    require(after == hashes, "Source changed during preparation")
    print(json.dumps(report))


if __name__ == "__main__": main()
