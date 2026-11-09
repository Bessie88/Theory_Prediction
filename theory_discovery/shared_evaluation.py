"""Central label-set evaluation; callers must supply the SAME gold for both methods."""
from collections import Counter, defaultdict
from .workflow_contracts import Method, Metric, MetadataItem, AnnotationStatus, PredictionStatus, require_same_cases
from .action_comparison import compare_action_sets, validate_protocol, permits_novel_actions, novel_action_definition
from .evaluation_contracts import (
    GoldLabel, CaseScore, LabelPerformance, OwnerPerformance, MethodPerformance, EvaluationDetails,
)
from .schemas import require, stable_id

def ratio(n, d):
    return n / d if d else None

def metric(name, n, d):
    return Metric(name, ratio(n, d), n, d)

def choose_gold(annotations, reviews, protocol, policy):
    require(policy in ("MODEL_ONLY", "HUMAN_PREFERRED", "HUMAN_ONLY"), "Unknown gold policy")
    require_same_cases(annotations.annotations, reviews.reviews)
    review = {r.identity.case_id: r for r in reviews.reviews}
    allowed = {a.action_id for a in protocol.actions}
    rows = []
    for raw in annotations.annotations:
        r = review[raw.identity.case_id]
        require(r.model_annotation == raw and raw.protocol_hash == protocol.protocol_hash,
                "Human/model annotation lineage mismatch")
        actions, source, status, reason = raw.observed_actions, "MODEL", raw.annotation_status.value, ""
        scorable = raw.annotation_status in (AnnotationStatus.ANNOTATED, AnnotationStatus.NO_CHANGE)
        if policy != "MODEL_ONLY" and r.review_status in ("CONFIRMED", "CORRECTED", "UNRESOLVED"):
            require(r.response_id and r.reviewer_id and r.completed_at_utc, "Completed human provenance required")
            source, status = "HUMAN", r.review_status
            if r.review_status == "CORRECTED":
                require(r.corrected_label is not None, "Missing human correction")
                actions = r.corrected_label
            scorable = r.human_verified
            if not scorable:
                actions, reason = (), "Human reviewer marked uncertain; no model fallback."
        elif policy == "HUMAN_ONLY":
            actions, scorable, reason = (), False, "No completed known human label."
            source = "HUMAN_UNAVAILABLE"
        require(all(a.action_id in allowed or (permits_novel_actions(protocol) and novel_action_definition(a) is not None)
                    for a in actions), "Gold uses an unexplained private action vocabulary")
        if any(a.action_id == "OTHER" for a in actions):
            scorable, reason = False, "OTHER preserves an unmodeled operation; identity unresolved."
        if not scorable and not reason:
            reason = "Annotation is unknown, uncertain or otherwise unscorable."
        rows.append(GoldLabel(raw.identity, tuple(actions), scorable, source, status, r.review_status,
                             raw.annotation_id, r.response_id, reason))
    return tuple(rows)

def score_predictions(batch, gold, protocol):
    require_same_cases(batch.predictions, gold)
    require(batch.action_protocol_hash == protocol.protocol_hash, "Prediction/gold protocol differs")
    allowed = {a.action_id for a in protocol.actions}
    actual = {r.identity.case_id: r for r in gold}
    rows = []
    for p in batch.predictions:
        g = actual[p.identity.case_id]
        ids = {a.action_id for a in p.predicted_actions}
        require(ids <= allowed, "Prediction uses private Action IDs")
        usable = p.prediction_status is PredictionStatus.PREDICTED and bool(ids) and "OTHER" not in ids
        if not g.scorable:
            match, reason = "GOLD_UNRESOLVED", g.reason
        elif not usable:
            match, reason = "ABSTAIN_" + p.prediction_status.value, "No exact-match credit; preserved separately from incorrect predictions."
            if p.prediction_status is PredictionStatus.PREDICTED:
                match = "PREDICTED_OTHER_UNRESOLVED"
        else:
            comparison = compare_action_sets(p.predicted_actions, g.actions, protocol)
            match, reason = comparison.category, comparison.reason
        owners = p.applicable_theory_ids if batch.method is Method.THEORY else p.applicable_policy_rule_ids
        rows.append(CaseScore(p.identity, batch.method, p.prediction_status.value, usable, g.scorable, match,
            tuple(sorted(ids)), tuple(sorted({a.action_id for a in g.actions})), owners, g.source, reason))
    return tuple(rows)

def summarize(method, rows, protocol):
    rows = tuple(rows)
    n, covered = len(rows), sum(r.usable for r in rows)
    scored = tuple(r for r in rows if r.gold_scorable)
    covered_scored = tuple(r for r in scored if r.usable)
    exact = sum(r.match == "EXACT" for r in scored)
    metrics = [metric("coverage", covered, n),
        metric("covered_case_accuracy", exact, len(covered_scored)),
        metric("full_set_exact_over_scorable_gold_including_abstentions", exact, len(scored)),
        metric("all_case_confirmed_exact_yield_lower_bound", exact, n),
        metric("all_case_exact_yield_upper_bound", exact + n - len(scored), n),
        metric("gold_scorability", len(scored), n)]
    labels = []
    known_ids = {a.action_id for a in protocol.actions}
    new_ids = sorted({key for row in scored for key in row.actual_ids if key not in known_ids})
    for action_id in [a.action_id for a in protocol.actions] + new_ids:
        if action_id == "OTHER":
            labels.append(LabelPerformance("OTHER", 0, 0, 0, sum("OTHER" in r.actual_ids for r in rows), None, None, None))
            continue
        tp = fp = fn = support = 0
        for row in scored:
            predicted = set(row.predicted_ids) if row.usable else set()
            actual = set(row.actual_ids)
            tp += action_id in predicted & actual
            fp += action_id in predicted - actual
            fn += action_id in actual - predicted
            support += action_id in actual
        labels.append(LabelPerformance(action_id, tp, fp, fn, support,
                      ratio(tp, tp + fp), ratio(tp, tp + fn), ratio(2 * tp, 2 * tp + fp + fn)))
    active = tuple(r for r in labels if r.f1 is not None)
    owners = sorted({owner for r in rows for owner in r.matched_owner_ids})
    owner_rows = []
    for owner in owners:
        matched = [r for r in rows if owner in r.matched_owner_ids]
        known = [r for r in matched if r.gold_scorable]
        correct = sum(r.match == "EXACT" for r in known)
        owner_rows.append(OwnerPerformance(owner, len(matched), len(known), sum(r.usable for r in known),
            correct, ratio(correct, len(known)), "Case-level ensemble result among owner matches, not isolated owner causation."))
    revisions = defaultdict(list)
    for row in rows:
        revisions[row.identity.revision_id].append(row)
    known_revisions = [group for group in revisions.values() if all(r.gold_scorable for r in group)]
    metrics.extend((metric("revision_all_units_exact", sum(all(r.match == "EXACT" for r in group)
                    for group in known_revisions), len(known_revisions)),
                    metric("revision_any_usable_coverage", sum(any(r.usable for r in g) for g in revisions.values()), len(revisions))))
    counts = Counter(r.prediction_status for r in rows)
    counts.update({"GOLD_UNRESOLVED": n - len(scored), "PARTIAL": sum(r.match == "PARTIAL" for r in rows),
        "CONFIRMED_NEW_ACTION_CASES": sum(any(key not in known_ids for key in r.actual_ids) for r in scored),
        "CONFIRMED_NEW_ACTION_LABELS": len(new_ids),
        "PREDICTED_OTHER": sum("OTHER" in r.predicted_ids for r in rows),
        "ALL_REVISIONS": len(revisions), "UNSCORABLE_REVISIONS": len(revisions) - len(known_revisions)})
    return MethodPerformance(method, tuple(r.identity.case_id for r in rows), tuple(metrics), tuple(labels),
        sum(r.f1 for r in active) / len(active) if active else None,
        tuple(r.action_id for r in active), tuple(MetadataItem(k, v) for k, v in sorted(counts.items())), tuple(owner_rows))

def evaluate(inputs):
    validate_protocol(inputs.action_protocol)
    from .theory_application_registry import validate_study
    validate_study(inputs.study)
    require(inputs.theory_predictions.study_spec_hash == inputs.study.freeze_hash, "Evaluation study mismatch")
    require(inputs.settings.protocol_frozen, "Freeze evaluation policy before Later outcomes")
    require_same_cases(inputs.later.records, inputs.annotations.annotations, inputs.reviews.reviews,
                       inputs.theory_predictions.predictions, inputs.npov_predictions.predictions)
    editor = inputs.context.revisions[0].editor if inputs.context.revisions else None
    require(editor is not None and all(r.editor == editor for r in inputs.context.revisions) and
            len({r.revision_id for r in inputs.context.revisions}) == len(inputs.context.revisions) and
            all(c.identity.editor == editor for c in inputs.later.records),
            "Evaluation metadata/editor missing")
    require({r.revision_id for r in inputs.context.revisions} >= {c.identity.revision_id for c in inputs.later.records},
            "Evaluation revision metadata incomplete")
    gold = choose_gold(inputs.annotations, inputs.reviews, inputs.action_protocol, inputs.settings.gold_policy)
    scores = tuple(row for batch in (inputs.theory_predictions, inputs.npov_predictions)
                   for row in score_predictions(batch, gold, inputs.action_protocol))
    def methods(reference):
        return tuple(summarize(batch.method, score_predictions(batch, reference, inputs.action_protocol), inputs.action_protocol)
                     for batch in (inputs.theory_predictions, inputs.npov_predictions))
    model_gold = choose_gold(inputs.annotations, inputs.reviews, inputs.action_protocol, "MODEL_ONLY")
    human_gold = choose_gold(inputs.annotations, inputs.reviews, inputs.action_protocol, "HUMAN_ONLY")
    completed = [r for r in inputs.reviews.reviews if r.response_id]
    verified = [r for r in completed if r.human_verified]
    agreement = [compare_action_sets(r.model_annotation.observed_actions,
        r.corrected_label if r.corrected_label is not None else r.model_annotation.observed_actions,
        inputs.action_protocol).category for r in verified
        if r.model_annotation.annotation_status in (AnnotationStatus.ANNOTATED, AnnotationStatus.NO_CHANGE)]
    agreement = [a for a in agreement if a != "UNRESOLVED"]
    reliability = (metric("review_completion_among_selected", len(completed),
                         sum(r.review_status != "NOT_SELECTED" for r in inputs.reviews.reviews)),
                   metric("known_human_review_fraction_all_cases", len(verified), len(gold)),
                   metric("model_human_exact_agreement_scorable_verified_subset", agreement.count("EXACT"), len(agreement)))
    feature_verified = [r for r in inputs.feature_reviews.records if r.human_verified and r.model_label.status == "KNOWN"]
    comparable_features = [r for r in feature_verified if (r.corrected_label or r.model_label).status == "KNOWN"]
    reliability += (metric("feature_model_human_value_agreement_verified_subset",
        sum(r.model_label.value == (r.corrected_label or r.model_label).value for r in comparable_features),
        len(comparable_features)),)
    mode_counts = Counter(r.review_mode + ":" + r.review_status for r in inputs.reviews.reviews)
    feature_counts = Counter(r.mode + ":" + r.review_status for r in inputs.feature_reviews.records)
    return EvaluationDetails(editor, inputs.action_protocol.protocol_hash,
        stable_id("evaluation_protocol", inputs.settings.to_dict()), inputs.theory_predictions.study_spec_hash,
        gold, scores, methods(gold), methods(model_gold), methods(human_gold), reliability,
        tuple(MetadataItem(k, v) for k, v in sorted(mode_counts.items())),
        tuple(MetadataItem(k, v) for k, v in sorted(feature_counts.items())),
        ("Human-preferred is a mixed reference on a targeted subset; model-only and human-only views remain separate.",
         "Abstentions receive no exact credit in full-set metrics, but are not ordinary incorrect predictions.",
         "Confirmed new operations remain scorable misses for the frozen prediction vocabulary; uncertain annotations stay separate.",
         "Macro-F1 includes confirmed new labels and excludes OTHER and zero TP/FP/FN labels; denominators are reported.",
         "Multiple units share revisions; revision metrics are descriptive, not confidence intervals.",
         "No causal/editor-style inference follows from these metrics."))
