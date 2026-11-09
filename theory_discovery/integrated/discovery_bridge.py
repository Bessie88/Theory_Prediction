"""Narrow EARLY-only compatibility bridge to the EXISTING natural workflow.

No run-specific launcher, alternative coder, induction or constructor is created.
All execution/I/O below is inside explicitly called future APIs, never import time.
"""
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
import fcntl
import os
from ..workflow_contracts import SourceReference, Period
from ..stage_interfaces import DiscoveryInput, DiscoveryOutput, NativeTheoryBundle, LegacyCaseAlias
from ..config import Config, LLMConfig
from ..schemas import EditingPair, stable_id, require
from ..storage import Cache, write_json
from ..artifact_registry import file_hash
from ..chronological_split import early_manifest_projection

# Original parallel_discovery reads this environment variable. The wrapper pins
# the central value temporarily and restores the prior value, without changing
# the original analytical implementation.


def prepare_discovery(inputs: DiscoveryInput):
    manifest = inputs.split.manifest
    require(inputs.workspace.execution_authorized and inputs.workspace.models_authorized,
            "Explicit workflow/model authorization required before discovery")
    require(inputs.settings.model.enabled, "Discovery model is disabled")
    cases = {c.case_id: c for c in manifest.cases if c.period is Period.EARLY}
    require({p.identity.case_id: p.identity for p in inputs.early.records} == cases,
            "Discovery input must be the exact EARLY case set")
    revisions = {r.revision_id: r for r in inputs.early.revisions}
    require(set(revisions) == set(manifest.early_revision_ids), "EARLY revision metadata mismatch")
    require(len(revisions) == len(inputs.early.revisions), "Duplicate EARLY revision metadata")
    for r in revisions.values():
        require(r.editor == manifest.editor and r.editor_verified_human and not r.missing_fields,
                "Invalid EARLY editor/revision metadata")
        require(r.timestamp == dict((m.key, m.value) for m in manifest.revision_timestamps)[r.revision_id],
                "Revision timestamp differs from frozen manifest")
    supported, excluded = [], []
    require(tuple(p.identity.case_id for p in inputs.early.records) ==
            tuple(c.case_id for c in manifest.cases if c.period is Period.EARLY),
            "Preserve the frozen EARLY input order")
    for pair in inputs.early.records:
        require(pair.identity.editor == manifest.editor and pair.identity.period is Period.EARLY and
                pair.identity.revision_id in manifest.early_revision_ids,
                "Future/editor-mismatched revision cannot enter induction")
        metadata = {m.key: m.value for m in pair.unit_metadata}
        require(metadata.get("text_unit") in ("BEFORE_TARGETED_SENTENCE", "PROVIDED_EDITED_SENTENCE_PAIR"),
                "Original workflow expects sentence pairs; full pages require an explicit adapter")
        if pair.alignment_status != "ALIGNED" or pair.after_text is None:
            excluded.append({"case_id": pair.identity.case_id, "revision_id": pair.identity.revision_id,
                             "reason": pair.alignment_status})
        else:
            supported.append(pair)
    require(bool(supported), "No compatible EARLY sentence pairs; no empty/fake discovery")
    cohort = stable_id("discovery_cohort", {
        "manifest": early_manifest_projection(manifest),
        "eligible": [p.to_dict() for p in supported], "excluded": excluded})
    aliases = tuple(LegacyCaseAlias(p.identity.case_id, f"case-{i + 1:04d}", cohort)
                    for i, p in enumerate(supported))
    # Original short-ID citation conventions work unchanged. Identity stays in a sidecar.
    pairs = tuple(EditingPair(alias.native_case_id, pair.before_text, pair.after_text,
                              pre_context={}, metadata={}, split="discovery")
                  for alias, pair in zip(aliases, supported))
    params = {p.key: p.value for p in inputs.settings.parameters}
    require(set(params) <= {f.name for f in fields(Config)}, "Unknown original Config parameter")
    llm_params = {p.key: p.value for p in inputs.settings.llm_parameters}
    require(set(llm_params) <= {f.name for f in fields(LLMConfig)}, "Unknown original LLMConfig parameter")
    protected = {"enabled", "base_url", "model", "temperature", "max_tokens", "retries"}
    require(not set(llm_params) & protected, "Model identity/budgets come from central ModelSettings")
    model = inputs.settings.model
    legacy_llm = LLMConfig(**dict(llm_params, enabled=model.enabled, base_url=model.endpoint,
                                 model=model.model, temperature=model.temperature,
                                 max_tokens=model.max_tokens, retries=model.retries))
    root = Path(inputs.workspace.root).resolve() / cohort
    cfg = Config(**dict(params, allow_real_data=True, input_path="", input_format="jsonl",
                        split="discovery", output_dir=str(root / "discovery"),
                        cache_dir=str(root / "discovery_cache"), pre_context_keys=[],
                        llm=legacy_llm)).validate()
    require(len(pairs) <= cfg.max_observations, "Configure the original observation limit explicitly")
    return pairs, cfg, aliases, supported, excluded, root, cohort


@contextmanager
def _legacy_environment(workers):
    previous = os.environ.get("SIMULATION_BOUNDARY_WORKERS")
    os.environ["SIMULATION_BOUNDARY_WORKERS"] = str(workers)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SIMULATION_BOUNDARY_WORKERS", None)
        else:
            os.environ["SIMULATION_BOUNDARY_WORKERS"] = previous


def run(inputs: DiscoveryInput) -> DiscoveryOutput:
    pairs, cfg, aliases, supported, excluded, root, cohort = prepare_discovery(inputs)
    # No GPU service startup, model downloads, canary, preflight, CLI or scheduler.
    # Future execution uses an explicitly configured already available endpoint.
    root.mkdir(parents=True, exist_ok=True)
    with (root / "bridge.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            identity = {"experiment_id": inputs.workspace.experiment_id,
                        "configuration_id": inputs.workspace.configuration_id, "cohort_id": cohort}
            owner_path = root / "bridge_identity.json"
            from ..storage import read_json
            if owner_path.exists():
                require(read_json(owner_path) == identity, "Refuse unrelated discovery workspace")
            else:
                write_json(owner_path, identity)
            write_json(root / "legacy_configuration.json", cfg.to_dict())
            write_json(root / "case_identity.json", {
                "aliases": [a.to_dict() for a in aliases],
                "cases": [p.to_dict() for p in supported],
                "revision_metadata": [r.to_dict() for r in inputs.early.revisions],
                "excluded_early_cases": excluded,
                "early_manifest": early_manifest_projection(inputs.split.manifest)})
            # Lazy imports avoid importing historical top-level execution scripts.
            from ..parallel_discovery import run_framework
            from ..hierarchical_theory import synthesize_hierarchical
            from ..response_recovery import RecoverableQwenJSONClient
            from ..wire_budget_client import WireBudgetQwenClient
            backend = RecoverableQwenJSONClient(cfg.llm, Cache(cfg.cache_dir))
            backend.verify_model()
            with _legacy_environment(inputs.settings.boundary_workers):
                artifacts = run_framework(pairs, cfg, backend)
            constructor_cache = Cache(root / "integration_cache")
            constructor = WireBudgetQwenClient(cfg.llm, constructor_cache)
            constructor.verify_model()
            final = synthesize_hierarchical(
                artifacts["theory"]["rules"], artifacts["theory"]["construction_case_ids"],
                constructor, constructor_cache, output=root / "construction",
                workers=inputs.settings.integration_workers)
            def ref(name, path):
                return SourceReference(name, str(path), file_hash(path), "")
            names = ("observations", "conditions", "action_groups", "condition_comparisons",
                     "boundary_discovery", "evidence_ledger", "semantic_alignment")
            sidecars = tuple(ref(name, root / "discovery" / (name + ".json")) for name in names)
            sidecars += (ref("native_discovery_theory", root / "discovery/theory.json"),
                         ref("case_identity", root / "case_identity.json"),
                         ref("legacy_configuration", root / "legacy_configuration.json"),
                         ref("bridge_identity", owner_path),
                         ref("group_theories", root / "construction/group_theories.json"))
            return DiscoveryOutput(NativeTheoryBundle(
                ref("native_theory", root / "construction/theory.json"), sidecars, aliases,
                final["generation_mode"], final["rule_representation"],
                final["integration_method"], final["version"]))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
