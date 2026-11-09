"""Registered-artifact report templates. No invented findings or model narrative."""
from collections import Counter
import json
from .evaluation_contracts import ResearchReport, CohortAnalysis
from .schemas import require, stable_id
SECTIONS = ("Dataset description", "Existing Theory Discovery workflow", "Feature operationalization",
    "Objective/Semantic/Hybrid extraction statistics", "Prediction performance", "NPOV comparison",
    "Boundary ablation", "Human annotation quality", "Disagreement analysis", "Editor-specific patterns", "Limitations")

def report_template():
    return "# Wikipedia workflow report\n\n" + "\n\n".join("## " + s + "\n\n{{" + str(i) + "}}" for i, s in enumerate(SECTIONS))

def encode(x):
    return json.dumps(x.to_dict() if hasattr(x, "to_dict") else x, ensure_ascii=False, indent=2)

def render_report(inputs):
    e, a, b, context = inputs.evaluation_details, inputs.editor_analysis, inputs.boundary_analysis, inputs.context
    require(e.editor == a.editor == inputs.theories.editor and e.study_hash == b.original_study_hash ==
            inputs.study.freeze_hash, "Report editor/freeze mismatch")
    from .action_comparison import novel_action_definition
    new_actions = {}
    known_ids = {action.action_id for action in inputs.study.catalog.actions}
    for gold in e.gold:
        if gold.scorable:
            for action in gold.actions:
                if action.action_id not in known_ids:
                    definition = novel_action_definition(action)
                    if definition is not None:
                        new_actions[definition.action_id] = definition.to_dict()
    content = [
        encode({"editor": e.editor.to_dict(), "early_revisions": context.early_revision_count,
            "later_revisions": context.later_revision_count, "cutoff": context.cutoff_utc,
            "unit_version": context.unit_version, "later_cases": len(e.gold),
            "sources": [r.to_dict() for r in context.source_references]}),
        "Existing parallel_discovery -> hierarchical_theory -> registered native Theory output -> "
        "integrated.theory_adapter -> TheoryBundle -> shared measurement planner. Induction is unchanged.",
        encode({"features": [s.to_dict() for s in inputs.study.measurement_freeze.snapshot.specifications],
                "unresolved": list(inputs.study.measurement_freeze.unresolved_clause_ids),
                "policy_snapshot": inputs.study.policy_registry.source_snapshot_id}),
        encode({"feature_methods": dict(Counter(s.measurement_method.value for s in inputs.study.measurement_freeze.snapshot.specifications)),
                "observation_statuses": dict(Counter(o.status.value for o in inputs.features.observations))}),
        encode({"primary": [m.to_dict() for m in e.methods],
                "confirmed_new_action_definitions": list(new_actions.values()),
                "model_only": [m.to_dict() for m in e.model_reference_methods],
                "human_only": [m.to_dict() for m in e.human_only_methods]}),
        "Identical cases, Action vocabulary and gold labels for both methods. Protocol: " + e.protocol_hash,
        encode(b),
        encode({"reliability": [m.to_dict() for m in e.annotation_reliability],
                "review_modes": [m.to_dict() for m in e.reliability_by_mode],
                "feature_review": [m.to_dict() for m in e.feature_review_counts]}),
        encode([r.to_dict() for r in a.disagreements]),
        encode({"patterns": [r.to_dict() for r in a.recurring_patterns],
                "temporal": [r.to_dict() for r in a.temporal_bins],
                "confounding": [r.to_dict() for r in a.confounding_counts], "interpretation": a.interpretation}),
        "\n".join(e.limitations + b.limitations + (
            "Targeted human-review metrics are subset descriptions, not population estimates.",
            "No automatic style or causal conclusion. Topic control, transfer production, replication and intervals need additional evidence."))]
    text = "# Wikipedia workflow report\n\n" + "\n\n".join(
        "## " + section + "\n\n" + value for section, value in zip(SECTIONS, content))
    return ResearchReport(e.editor, text, "registered-report-v1", stable_id("evaluation_details", e.to_dict()),
                          tuple(g.identity.case_id for g in e.gold), e.protocol_hash, e.evaluation_protocol_hash,
                          inputs.study.theory_registry.registry_hash, encode([m.to_dict() for m in e.methods]))

def compare_editors(reports, transfers=()):
    require(len({r.editor for r in reports}) == len(reports), "Duplicate editor report")
    require(len({(r.action_protocol_hash, r.evaluation_protocol_hash) for r in reports}) <= 1,
            "Editors must share Action/evaluation protocols")
    known = {r.editor for r in reports}
    for t in transfers:
        require(t.source_editor in known and t.target_editor in known and t.source_editor != t.target_editor and
                not t.target_in_source_training and t.source_registry_hash and t.protocol_id and t.target_case_ids,
                "Transfer evidence needs source/target isolation and frozen protocol")
        source = next(r for r in reports if r.editor == t.source_editor)
        require(source.theory_registry_hash == t.source_registry_hash, "Transfer theory source differs")
        target = next(r for r in reports if r.editor == t.target_editor)
        require(set(t.target_case_ids) <= set(target.case_ids), "Transfer target cases not registered")
    text = {"editors": [r.editor.to_dict() for r in reports], "transfers": [t.to_dict() for t in transfers],
            "editor_performance": {r.editor.wiki_id + ":" + r.editor.editor_id: json.loads(r.performance_json) for r in reports},
            "multiple_editors": len(reports) >= 2, "topic_controlled_comparisons": sum(t.matched_topic_control for t in transfers),
            "interpretation": "Descriptive evidence only; multiple editors and appropriate comparisons are necessary, "
                              "not sufficient. No automatic editor-style or causal claim."}
    return CohortAnalysis(tuple(reports), tuple(transfers), False, "# Cross-editor inventory\n\n" + encode(text))


def materialize_report(report, directory, artifact_id):
    """Future derived export from an already committed registry object."""
    from pathlib import Path
    from hashlib import sha256
    from .storage import write_json
    import os
    import tempfile
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "report.md"
    raw = report.markdown.encode("utf-8")
    if path.exists():
        require(path.read_bytes() == raw, "Refuse to overwrite a changed report export")
    else:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            handle.write(raw)
            temporary = handle.name
        os.replace(temporary, path)
    write_json(root / "provenance.json", {"artifact_id": artifact_id,
        "markdown_sha256": sha256(raw).hexdigest(), "template_source": "registered result, no model narrative"})
    return str(path)
