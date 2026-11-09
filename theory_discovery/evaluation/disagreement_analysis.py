"""Case retrieval and descriptive pattern inventory; no model-generated interpretation."""
from collections import defaultdict, Counter
import json
from .evaluation_contracts import DisagreementCase, QualitativeCase, PatternCount, EditorAnalysis
from theory_discovery.core.workflow_contracts import Method, MetadataItem
from theory_discovery.core.schemas import require

def dump(v):
    return json.dumps(v.to_dict() if hasattr(v, "to_dict") else v, ensure_ascii=False, sort_keys=True)

def analyze_editor(inputs):
    e = inputs.evaluation_details
    t = {r.identity.case_id: r for r in e.scores if r.method is Method.THEORY}
    n = {r.identity.case_id: r for r in e.scores if r.method is Method.NPOV}
    pairs = {r.identity.case_id: r for r in inputs.annotation_cases.cases}
    raw = {r.identity.case_id: r for r in inputs.annotations.annotations}
    human = {r.identity.case_id: r for r in inputs.reviews.reviews}
    tp = {r.identity.case_id: r for r in inputs.theory_predictions.predictions}
    np = {r.identity.case_id: r for r in inputs.npov_predictions.predictions}
    traces = {r.identity.case_id: r for r in inputs.prediction_provenance.records}
    nt = {r.identity.case_id: r for r in inputs.npov_prediction_provenance.records}
    theories = {r.theory_id: r for r in inputs.theories.theories}
    rules = {r.rule_id: r for r in inputs.policy_rules.rules}
    revisions = {r.revision_id: r for r in inputs.context.revisions}
    require(set(t) == set(n) == set(pairs) == set(raw) == set(human) == set(traces) == set(nt),
            "Qualitative case/provenance identities differ")
    matrix, cases, patterns, temporal, confounds = [], [], defaultdict(list), Counter(), Counter()
    for cid in sorted(t):
        a, b = t[cid], n[cid]
        if not a.gold_scorable or not b.gold_scorable:
            category = "GOLD_UNRESOLVED"
        elif not a.usable or not b.usable:
            category = "ABSTENTION_OR_UNRESOLVED_PREDICTION"
        else:
            category = ("BOTH_CORRECT" if a.match == b.match == "EXACT" else
                        "THEORY_ONLY_CORRECT" if a.match == "EXACT" else
                        "NPOV_ONLY_CORRECT" if b.match == "EXACT" else "BOTH_INCORRECT")
        statuses = tuple(sorted({r.prediction_status for r in (a, b) if r.prediction_status != "PREDICTED"}))
        matrix.append(DisagreementCase(a.identity, category, a.prediction_status, b.prediction_status, a.match, b.match, statuses))
        case, trace, policy_trace, meta = pairs[cid], traces[cid], nt[cid], revisions[a.identity.revision_id]
        require(case.identity == a.identity == b.identity == raw[cid].identity == human[cid].identity,
                "Qualitative canonical identity mismatch")
        require(set(tp[cid].applicable_theory_ids) <= set(theories) and
                set(np[cid].applicable_policy_rule_ids) <= set(rules), "Matched source representation missing")
        cases.append(QualitativeCase(a.identity, category, dump(meta), case.before_text, case.after_text,
            dump(case.actual_diff) if case.actual_diff else "null",
            dump([theories[k].to_dict() for k in tp[cid].applicable_theory_ids if k in theories]),
            dump([v.to_dict() for v in trace.clause_evaluations]), dump(tp[cid]),
            dump([rules[k].to_dict() for k in np[cid].applicable_policy_rule_ids if k in rules]),
            dump(np[cid]), dump(raw[cid]), dump(human[cid]),
            dump({"theory": trace.to_dict(), "npov": policy_trace.to_dict()})))
        if category in ("THEORY_ONLY_CORRECT", "NPOV_ONLY_CORRECT", "BOTH_INCORRECT"):
            pattern = dump({"conditions": sorted((w.feature_id, dump(w.value)) for w in trace.condition_features if w.valid),
                            "actions": list(a.predicted_ids)})
            patterns[(category, pattern)].append((a.identity, meta))
        temporal[(meta.timestamp[:7] if meta.timestamp else "MISSING") + ":" + category] += 1
        confounds["PAGE:" + meta.page_id if meta.page_id else "MISSING_PAGE_ID"] += 1
    temporal_summary = []
    for month in sorted({r.timestamp[:7] if r.timestamp else "MISSING" for r in revisions.values()}):
        for method in (Method.THEORY, Method.NPOV):
            selected = [r for r in e.scores if r.method is method and
                (revisions[r.identity.revision_id].timestamp[:7] if revisions[r.identity.revision_id].timestamp else "MISSING") == month]
            if selected:
                known = [r for r in selected if r.gold_scorable]
                temporal_summary.append(MetadataItem(month + ":" + method.value, dump({
                    "cases": len(selected), "usable": sum(r.usable for r in selected),
                    "scorable": len(known), "exact": sum(r.match == "EXACT" for r in known),
                    "coverage": sum(r.usable for r in selected) / len(selected),
                    "full_set_exact": sum(r.match == "EXACT" for r in known) / len(known) if known else None})))
    grouped = tuple(PatternCount(c, p, tuple(i.case_id for i, _ in rows),
        tuple(sorted({i.revision_id for i, _ in rows})), tuple(sorted({m.page_id for _, m in rows if m.page_id})),
        sum(m.page_id is None for _, m in rows)) for (c, p), rows in sorted(patterns.items()))
    return EditorAnalysis(e.editor, tuple(matrix), tuple(cases), grouped,
        tuple(temporal_summary),
        tuple(MetadataItem(k, v) for k, v in sorted(confounds.items())),
        "Patterns and monthly counts are descriptive retrieval candidates, not editor-style or causal conclusions. "
        "Page IDs do not supply topic labels or adjusted confounding estimates.")

def retrieve_cases(analysis, case_ids=None, categories=("THEORY_ONLY_CORRECT", "NPOV_ONLY_CORRECT", "BOTH_INCORRECT")):
    wanted = set(case_ids) if case_ids is not None else None
    require(wanted is None or wanted <= {r.identity.case_id for r in analysis.qualitative_cases}, "Unknown case")
    rank = {name: i for i, name in enumerate(categories)}
    return tuple(sorted((r for r in analysis.qualitative_cases if
        (r.identity.case_id in wanted if wanted is not None else r.category in categories)),
        key=lambda r: (rank.get(r.category, len(rank)), r.identity.case_id)))
