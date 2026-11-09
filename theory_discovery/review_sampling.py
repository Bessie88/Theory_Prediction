"""Deterministic selection only; selectors know strata, reviewer packets do not."""
from collections import Counter
from .workflow_contracts import PredictionStatus, Method, require_same_cases, MetadataItem
from .action_comparison import provisional_agreement, validate_protocol
from .review_contracts import (
    ReviewAssignment, ReviewAssignments, SamplingAuditRow, ReviewSamplingAudit,
)
from .schemas import require, stable_id


def select_reviews(theory, npov, annotations, feature_source, protocol, settings):
    validate_protocol(protocol)
    require(theory.method is Method.THEORY and npov.method is Method.NPOV, "Sampling source methods differ")
    require_same_cases(theory.predictions, npov.predictions, annotations.annotations)
    require(theory.study_spec_hash == npov.study_spec_hash and
            theory.action_protocol_hash == npov.action_protocol_hash == protocol.protocol_hash,
            "Sampling methods use different frozen studies/Action protocols")
    t = {p.identity.case_id: p for p in theory.predictions}
    p = {row.identity.case_id: row for row in npov.predictions}
    a = {row.identity.case_id: row for row in annotations.annotations}
    require(len({row.annotation_id for row in a.values()}) == len(a) and
            all(row.annotation_id and row.protocol_hash == protocol.protocol_hash for row in a.values()),
            "Sampling needs registered, protocol-bound annotations")
    label_counts = Counter(label for row in a.values() for label in {v.action_id for v in row.observed_actions})
    membership = {}
    for cid, row in a.items():
        ta, pa = provisional_agreement(t[cid], row, protocol), provisional_agreement(p[cid], row, protocol)
        success_t, success_p = ta == "EXACT", pa == "EXACT"
        groups = {"RANDOM"}
        if success_t and not success_p:
            groups.add("THEORY_ONLY_SUCCESS")
        if success_p and not success_t:
            groups.add("NPOV_ONLY_SUCCESS")
        if success_t and success_p:
            groups.add("BOTH_SUCCESS")
        if ta in ("PARTIAL", "NO_MATCH") and pa in ("PARTIAL", "NO_MATCH"):
            groups.add("BOTH_FAILURE")
        groups.update(status for status in ("NOT_COVERED", "UNKNOWN", "CONFLICT")
                      if any(v.prediction_status.value == status for v in (t[cid], p[cid])))
        if any(label_counts[v.action_id] <= settings.rare_action_max_cases for v in row.observed_actions):
            groups.add("RARE_ACTION")
        membership[cid] = groups
    selected, counts = set(), []
    for quota in settings.quotas:
        candidates = [cid for cid, groups in membership.items() if quota.stratum in groups]
        counts.append(MetadataItem(quota.stratum, len(candidates)))
        # Rank independently per stratum, union duplicates without replacement.
        ranked = sorted(candidates, key=lambda cid: stable_id("review_rank", [settings.seed, quota.stratum, cid]))
        selected.update(ranked[:quota.limit])
    action_assignments = ReviewAssignments(tuple(
        ReviewAssignment(a[cid].identity, a[cid].annotation_id, settings.action_mode) for cid in sorted(selected)))
    feature_candidates = feature_source.records
    require(all(r.identity.case_id in a and r.identity == a[r.identity.case_id].identity for r in feature_candidates),
            "Feature sampling case identity differs from annotation manifest")
    require(len({r.subject_id for r in feature_candidates}) == len(feature_candidates), "Duplicate feature sampling unit")
    feature_ranked = sorted(feature_candidates,
        key=lambda r: stable_id("feature_review_rank", [settings.seed, r.identity.to_dict(), r.subject_id]))
    feature_assignments = ReviewAssignments(tuple(ReviewAssignment(r.identity, r.subject_id, settings.feature_mode)
                                                   for r in feature_ranked[:settings.feature_limit]))
    audit = ReviewSamplingAudit(settings.seed, tuple(SamplingAuditRow(a[cid].identity,
        tuple(sorted(membership[cid])), cid in selected) for cid in sorted(a)), tuple(counts),
        "Success means provisional exact label agreement with an UNVERIFIED model annotation, never human correctness. "
        "Overlapping quotas are unioned; no resampling/top-up or population inference is implied. Features sampled independently.")
    return action_assignments, feature_assignments, audit
