"""Lossless metadata normalization after EXISTING final theory construction."""
from dataclasses import replace
import json
from pathlib import Path
from ..workflow_contracts import (
    TheoryRecord, ProseClause, ActionOption, SourceReference, Period, TheoryEvidenceLink,
    NativeRepresentation, EditorRevision, theory_id_for,
)
from ..stage_interfaces import TheoryAdapterInput, TheoryAdapterOutput, TheoryBundle
from ..artifact_registry import file_hash
from ..schemas import require, stable_id
from ..chronological_split import early_manifest_projection


def _read(ref):
    require(file_hash(ref.location) == ref.sha256, "Native source/sidecar hash mismatch")
    # Bridge-produced JSON only. No original dataset is opened by this adapter.
    return json.loads(Path(ref.location).read_text(encoding="utf-8"))


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def normalize_theory(inputs: TheoryAdapterInput, native, sidecar_values) -> TheoryAdapterOutput:
    """Pure normalization API; source bytes/locators remain unchanged."""
    expected = ("natural_language_predictive_theory", "natural_language",
                "within_method_group_then_all_groups", "method-group-then-global-prose-v1")
    actual = tuple(native.get(k) for k in
                   ("generation_mode", "rule_representation", "integration_method", "version"))
    require(actual == expected, "Unknown native theory schema; no executable-rule fallback")
    # Pure original validator; no model call is performed by it.
    from ..natural_pipeline import validate_theory
    validate_theory(native)
    manifest = inputs.split.manifest
    cases = {c.case_id: c for c in manifest.cases if c.period is Period.EARLY}
    aliases = {a.native_case_id: a.canonical_case_id for a in inputs.native.case_aliases}
    require(len(aliases) == len(inputs.native.case_aliases) and
            len(set(aliases.values())) == len(aliases), "Case alias map must be bijective")
    require(set(aliases.values()) <= set(cases), "Native theory cites non-EARLY cases")
    require(set(native["construction_case_ids"]) == set(aliases),
            "Construction IDs do not match the accepted EARLY cohort")
    identity = sidecar_values["case_identity"]
    require(identity["aliases"] == [a.to_dict() for a in inputs.native.case_aliases] and
            identity["early_manifest"] == early_manifest_projection(manifest), "Identity sidecar/manifest mismatch")
    supplied_cases = {row["identity"]["case_id"]: row for row in identity["cases"]}
    require(set(supplied_cases) == set(aliases.values()), "Missing historical case provenance")
    for canonical, row in supplied_cases.items():
        require(row["identity"] == cases[canonical].to_dict(), "Historical identity mismatch")
    observations = sidecar_values["observations"]
    require({o["case_id"] for o in observations} == set(aliases), "Observation cohort mismatch")
    require(len(observations) == len(aliases), "Duplicate observations")
    for observation in observations:
        original = supplied_cases[aliases[observation["case_id"]]]
        require(observation["before_text"] == original["before_text"] and
                observation["after_text"] == original["after_text"], "Native case text changed")

    rules = {r["rule_id"]: r for r in native["rules"]}
    source_refs = {ref.source_id: ref for ref in inputs.native.sidecars}
    require(len(source_refs) == len(inputs.native.sidecars), "Duplicate sidecar names")
    native_ref = inputs.native.theory
    diagnostics = ["Conditions remain source prose; no executable conversion performed.",
                   "Evidence counts are inherited construction citations, not prediction probabilities.",
                   "Action alternatives remain UNSPECIFIED; catalog mapping is deferred to Measurement Planner."]
    representations, original_conditions = [], []
    for index, condition in enumerate(sidecar_values["conditions"]):
        original_conditions.append(NativeRepresentation(
            condition["condition_id"], "CONDITION", _json(condition),
            replace(source_refs["conditions"], json_pointer="/" + str(index))))
    rule_positions = {r["rule_id"]: index for index, r in enumerate(native["rules"])}
    for rid, rule in rules.items():
        representations.append(NativeRepresentation(
            rid, "RULE", _json(rule), replace(native_ref, json_pointer="/rules/" + str(rule_positions[rid]))))
    action_records, seen_actions, theories = [], set(), []
    for branch_index, branch in enumerate(native["branches"]):
        refs = tuple(branch["source_rule_ids"])
        compared = sorted(set().union(*(set(rules[r]["construction_case_ids"]) for r in refs)))
        require(set(compared) <= set(aliases), "Unknown historical evidence case")
        canonical_ids = tuple(sorted(aliases[c] for c in compared))
        links = tuple(TheoryEvidenceLink(
            aliases[c], c, cases[aliases[c]].editor, cases[aliases[c]].revision_id,
            cases[aliases[c]].prediction_unit_id) for c in compared)
        pointer = "/branches/" + str(branch_index)
        clauses = [ProseClause(branch["situation"], refs, pointer + "/situation", "UNPLANNED")]
        boundaries = []
        if "boundary_text" in branch:
            boundaries.append(ProseClause(branch["boundary_text"], refs,
                                          pointer + "/boundary_text", "UNPLANNED"))
        else:
            diagnostics.append(branch["branch_id"] + ": no branch-level boundary field; source boundaries retained")
        for rid in refs:
            rp = "/rules/" + str(rule_positions[rid])
            clauses.append(ProseClause(rules[rid]["natural_rule_text"], (rid,),
                                       rp + "/natural_rule_text", "UNPLANNED"))
            boundaries.append(ProseClause(rules[rid]["boundary_text"], (rid,),
                                           rp + "/boundary_text", "UNPLANNED"))
        options = []
        # Native validator proves these options match frozen source rules exactly.
        for option_index, option in enumerate(branch["action_options"]):
            action_id = stable_id("native_action", option["action"])
            prov_id = action_id  # Resolves to a source_actions NativeRepresentation.
            options.append(ActionOption((action_id,), tuple(option["source_rule_ids"]), (prov_id,),
                                        "UNSPECIFIED"))
            if action_id not in seen_actions:
                action_records.append(NativeRepresentation(
                    action_id, "ACTION", _json(option["action"]),
                    replace(native_ref, json_pointer=pointer + "/action_options/" +
                            str(option_index) + "/action")))
                seen_actions.add(action_id)
        source_revisions = {link.revision_id for link in links}
        revision_rows = tuple(EditorRevision.from_dict(row) for row in identity["revision_metadata"])
        revision_metadata = {row.revision_id: row for row in revision_rows}
        require(len(revision_metadata) == len(revision_rows), "Duplicate historical revision metadata")
        require(source_revisions <= set(revision_metadata), "Historical revision provenance missing")
        timestamps = {item.key: item.value for item in manifest.revision_timestamps}
        require(all(row.editor == manifest.editor and row.editor_verified_human and
                    not row.missing_fields and row.revision_id in manifest.early_revision_ids and
                    row.timestamp == timestamps[row.revision_id] for row in revision_rows),
                "Historical revision metadata differs from EARLY manifest")
        historical_refs = tuple(ref for rid in sorted(source_revisions)
                                for ref in revision_metadata[rid].provenance)
        original_refs = (native_ref,) + inputs.native.sidecars + historical_refs
        uncertainty = tuple(branch.get("uncertainty", []))
        theories.append(TheoryRecord(
            theory_id_for(manifest.editor, branch["branch_id"], native_ref.sha256),
            manifest.editor, branch["situation"] + "\n" + branch["interpretation"], branch["branch_id"],
            tuple(clauses), tuple(boundaries), tuple(options), canonical_ids, refs, original_refs,
            "VERIFIED_SOURCE_LINKS",
            uncertainty + ("Semantic applicability and action composition are not operationalized.",),
            tuple(sorted({link.revision_id for link in links})), links, _json(branch)))
    require(len({t.theory_id for t in theories}) == len(theories), "Duplicate canonical theory IDs")
    return TheoryAdapterOutput(TheoryBundle(
        manifest.editor, tuple(theories), native_ref,
        tuple(row["rule_id"] for row in native["excluded_theory_rules"]), tuple(diagnostics),
        tuple(representations), tuple(original_conditions), tuple(action_records),
        inputs.native.case_aliases, _json(native["excluded_theory_rules"])))


def run(inputs: TheoryAdapterInput) -> TheoryAdapterOutput:
    """Registry supplies the native bundle; planner never supplies original file paths."""
    native = _read(inputs.native.theory)
    values = {ref.source_id: _read(ref) for ref in inputs.native.sidecars}
    require({"case_identity", "observations", "conditions"} <= set(values), "Missing provenance sidecars")
    return normalize_theory(inputs, native, values)
