"""Code-only source-action prediction. No annotation/evaluation access or LLM."""
from dataclasses import replace
import itertools
from theory_discovery.core.workflow_contracts import ActionInstance, ActionPrediction, Method, PredictionStatus
from theory_discovery.core.application_contracts import (
    PredictedActionTrace, ActionConflict, PredictionProvenance, PredictionProvenanceArtifact,
)
from .theory_applicability import apply_knowledge
from theory_discovery.core.schemas import require, stable_id

VERSION = "deterministic-source-action-prediction-v1"


def _action_candidate(case, theory, directive, trace, feature_map, source, method):
    if source.mapping_status == "UNMAPPED" or source.definition is None:
        return None, "UNMAPPED_SOURCE_ACTION:" + source.source_action_id
    spans, observation_ids = (), ()
    if directive.target_mode == "ALL_EVIDENCE_SPANS":
        observation = feature_map.get(directive.target_feature_id)
        if observation is None or observation.status.value != "KNOWN" or not observation.evidence:
            return None, "UNRESOLVED_ACTION_TARGET:" + directive.directive_id
        spans = tuple(sorted(set(observation.evidence), key=lambda s: (s.start, s.end, s.quote)))
        for span in spans:
            require(span.side == "before", "Action targets must be Before spans")
            span.check_text(case.before_text)
        observation_ids = (observation.observation_id,)
    action = ActionInstance(source.definition.action_id, spans[0].quote if len(spans) == 1 else None,
                            source.original_json, spans)
    key = stable_id("action_trace", [case.identity.to_dict(), theory.theory_id,
                                   directive.directive_id, source.source_action_id,
                                   [s.to_dict() for s in spans]])
    return PredictedActionTrace(key, action, (source.source_action_id,), ((theory.theory_id,) if method is Method.THEORY else ()),
                                directive.source_rule_ids, (directive.directive_id,),
                                observation_ids, spans, (theory.theory_id,), method), None


def _merge_duplicates(candidates, actions):
    buckets = {}
    for row in candidates:
        # Same original action semantics at the same targeting granularity only.
        key = (actions[row.source_action_ids[0]].semantic_signature, bool(row.target_spans))
        buckets.setdefault(key, []).append(row)
    merged = []
    for rows in buckets.values():
        source_ids = tuple(sorted({key for row in rows for key in row.source_action_ids}))
        representative = source_ids[0]  # Representation of EXACT duplicates, never a winning theory.
        spans = tuple(sorted({span for row in rows for span in row.target_spans},
                             key=lambda s: (s.start, s.end, s.quote)))
        action = ActionInstance(actions[representative].definition.action_id, spans[0].quote if len(spans) == 1 else None,
                                actions[representative].original_json, spans)
        merged.append(PredictedActionTrace(
            stable_id("merged_action_trace", [source_ids, sorted(row.action_trace_id for row in rows)]),
            action, source_ids, tuple(sorted({t for row in rows for t in row.theory_ids})),
            tuple(sorted({r for row in rows for r in row.source_rule_ids})),
            tuple(sorted({d for row in rows for d in row.directive_ids})),
            tuple(sorted({o for row in rows for o in row.target_observation_ids})), spans,
            tuple(sorted({owner for row in rows for owner in row.owner_ids})), rows[0].knowledge_source))
    return tuple(sorted(merged, key=lambda row: row.action_trace_id))


def _pair_relation(left, right, registry):
    relations = {frozenset((r.left_action_id, r.right_action_id)): r for r in registry.compatibility}
    declared = [relations[frozenset((a, b))] for a in left.source_action_ids
                for b in right.source_action_ids if frozenset((a, b)) in relations]
    incompatible = next((r for r in declared if r.relation == "INCOMPATIBLE"), None)
    if incompatible:
        return ActionConflict(left.action.action_id, right.action.action_id, "INCOMPATIBLE",
                              incompatible.authority_id, incompatible.rationale)
    compatible = next((r for r in declared if r.relation == "COMPATIBLE"), None)
    if compatible:
        return ActionConflict(left.action.action_id, right.action.action_id, "COMPATIBLE",
                              compatible.authority_id, compatible.rationale)
    theories = {t.theory_id: t for t in registry.theories}
    joint = next((theories[key] for key in set(left.owner_ids or left.theory_ids) & set(right.owner_ids or right.theory_ids)
                  if theories[key].composition == "ALL_SUPPORTED"), None)
    if joint:
        return ActionConflict(left.action.action_id, right.action.action_id, "COMPATIBLE",
                              joint.authority_id, "Frozen source-backed ALL_SUPPORTED composition.")
    return ActionConflict(left.action.action_id, right.action.action_id, "UNKNOWN", None,
                          "No frozen source/protocol establishes compatibility; no heuristic applied.")


def predict_knowledge(features, applicability, study, method=Method.THEORY, boundary_mode="WITH_BOUNDARIES"):
    # Cheap deterministic replay verifies the supplied decision artifact. This
    # does not extract a feature, call a model, or read any observed outcome.
    expected, _ = apply_knowledge(features, study, method, boundary_mode)
    require(applicability == expected, "Applicability artifact differs from frozen feature/source logic")
    registry = study.theory_registry if method is Method.THEORY else study.policy_registry
    action_map = {a.source_action_id: a for a in registry.source_actions}
    by_decision = {(d.identity.case_id, d.owner_id): d for d in applicability.decisions}
    by_trace = {(t.identity.case_id, t.theory_id): t for t in applicability.traces}
    observations = {(o.identity.case_id, o.feature_id): o for o in features.observations}
    predictions, provenance = [], []
    for case in features.cases:
        applicable_ids, rejected_ids, unknown_ids = [], [], []
        case_traces, decision_ids, candidates, relations, reasons = [], [], [], [], []
        reasons.extend("SOURCE_SCOPE_UNRESOLVED:" + section for section in registry.unresolved_source_sections)
        for theory in registry.theories:
            decision = by_decision[(case.identity.case_id, theory.theory_id)]
            trace = by_trace[(case.identity.case_id, theory.theory_id)]
            case_traces.append(trace)
            decision_ids.append(decision.decision_id)
            if decision.status == "FALSE":
                rejected_ids.append(theory.theory_id)
                continue
            if decision.status != "TRUE":
                unknown_ids.append(theory.theory_id)
                continue
            applicable_ids.append(theory.theory_id)
            gate_map = {gate.directive_id: gate for gate in trace.action_gates}
            local_candidates = []
            for directive in theory.directives:
                gate = gate_map[directive.directive_id]
                if gate.truth is None:
                    reasons.append("ACTION_GATE_UNKNOWN:" + directive.directive_id)
                    continue
                if gate.truth is False:
                    continue
                # Only an observation accepted by applicability evidence validation
                # may be used for deterministic target localization.
                feature_map = {key: value for (cid, key), value in observations.items()
                               if cid == case.identity.case_id}
                if directive.target_feature_id is not None:
                    target = feature_map.get(directive.target_feature_id)
                    if target is not None:
                        # Target-only features were validated in apply_theories too;
                        # recompute their structural validity before localization.
                        from .theory_applicability import index_case_features
                        specs = {s.feature_id: s for s in study.measurement_freeze.snapshot.specifications}
                        valid, _ = index_case_features(case, (target,), specs)
                        feature_map = {**feature_map, directive.target_feature_id: valid.get(directive.target_feature_id)}
                candidate, error = _action_candidate(case, theory, directive, trace, feature_map,
                                                     action_map[directive.source_action_id], method)
                if error:
                    reasons.append(error)
                else:
                    local_candidates.append(candidate)
            if not local_candidates:
                reasons.append(("NO_SOURCE_ACTION_FOR_APPLICABLE_THEORY:" if method is Method.THEORY else "POLICY_OPERATION_UNDERSPECIFIED:") + theory.theory_id)
            if theory.composition == "ALLOWABLE_ALTERNATIVES":
                signatures = {action_map[r.source_action_ids[0]].semantic_signature for r in local_candidates}
                if len(signatures) > 1:
                    reasons.append("ALLOWABLE_ACTION_SET_UNDERSPECIFIED:" + theory.theory_id)
            if theory.composition == "EXCLUSIVE_ALTERNATIVES":
                groups = {}
                for row in local_candidates:
                    signature = action_map[row.source_action_ids[0]].semantic_signature
                    groups.setdefault(signature, row)
                if len(groups) > 1:
                    for left, right in itertools.combinations(groups.values(), 2):
                        relations.append(ActionConflict(left.action.action_id, right.action.action_id,
                            "INCOMPATIBLE", theory.authority_id,
                            "Several frozen exclusive action alternatives matched."))
            candidates.extend(local_candidates)
        merged = _merge_duplicates(candidates, action_map)
        if study.require_shared_action_ids:
            shared_ids = {a.action_id for a in study.catalog.actions}
            reasons.extend("UNMAPPED_SHARED_ACTION:" + row.action.action_id
                           for row in merged if row.action.action_id not in shared_ids)
        relations.extend(_pair_relation(a, b, registry) for a, b in itertools.combinations(merged, 2))
        definite_conflict = any(r.relation == "INCOMPATIBLE" for r in relations)
        unresolved_compatibility = any(r.relation == "UNKNOWN" for r in relations)
        if definite_conflict:
            status = PredictionStatus.CONFLICT
            reasons.append("INCOMPATIBLE_SOURCE_ACTIONS")
        elif unknown_ids or reasons or unresolved_compatibility:
            status = PredictionStatus.UNKNOWN
            if unknown_ids:
                reasons.append("INCOMPLETE_APPLICABILITY_SET")
            if unresolved_compatibility:
                reasons.append("ACTION_COMPATIBILITY_UNRESOLVED")
        elif applicable_ids and merged:
            status = PredictionStatus.PREDICTED
            reasons.append("FROZEN_SOURCE_ACTIONS_SUPPORTED")
        else:
            status = PredictionStatus.NOT_COVERED
            reasons.append(("NO_APPLICABLE_THEORY" if registry.theories else "EMPTY_FROZEN_THEORY_REGISTRY")
                           if method is Method.THEORY else
                           ("NO_APPLICABLE_POLICY_RULE" if registry.theories else "EMPTY_FROZEN_POLICY_RULE_REGISTRY"))
        conditions, boundaries = {}, {}
        all_clauses = tuple(c for trace in case_traces for c in trace.clauses)
        for clause in all_clauses:
            destination = conditions if clause.logical_role == "CONDITION" else boundaries
            for witness in clause.features:
                destination[witness.feature_id] = witness
        observation_ids = {w.observation_id for w in (*conditions.values(), *boundaries.values()) if w.observation_id}
        observation_ids.update(o for row in merged for o in row.target_observation_ids)
        matched_features = {key for (cid, key), obs in observations.items()
                            if cid == case.identity.case_id and obs.observation_id in observation_ids}
        missing = tuple(sorted({key for trace in case_traces for key in trace.missing_feature_ids}))
        record_id = stable_id("prediction_provenance", {
            "version": VERSION, "case": case.identity.to_dict(), "study": study.freeze_hash, "method": method.value, "boundary_mode": boundary_mode,
            "applicability": sorted(decision_ids), "features": applicability.feature_input_artifact_hash})
        candidate_rules = tuple(sorted({r for row in merged for r in row.source_rule_ids}))
        final_actions = tuple(row.action for row in merged) if status is PredictionStatus.PREDICTED else ()
        predictions.append(ActionPrediction(
            case.identity, method, tuple(sorted(applicable_ids)) if method is Method.THEORY else (), candidate_rules,
            final_actions, status, tuple(sorted(matched_features)), tuple(sorted(observation_ids)),
            study.freeze_hash, case.before_hash, tuple(sorted(set(reasons))),
            tuple(sorted(decision_ids)), record_id, tuple(sorted(rejected_ids)) if method is Method.THEORY else (),
            tuple(sorted(unknown_ids)) if method is Method.THEORY else (), tuple(row.action_trace_id for row in merged),
            tuple(sorted(conditions)), tuple(sorted(boundaries)), missing,
            tuple(sorted(applicable_ids)) if method is Method.NPOV else (),
            tuple(sorted(rejected_ids)) if method is Method.NPOV else (),
            tuple(sorted(unknown_ids)) if method is Method.NPOV else ()))
        provenance.append(PredictionProvenance(
            record_id, case.identity, study.freeze_hash, registry.registry_hash if method is Method.THEORY else "",
            applicability.feature_input_artifact_hash, tuple(sorted(decision_ids)),
            tuple(sorted(applicable_ids)) if method is Method.THEORY else (),
            tuple(sorted(rejected_ids)) if method is Method.THEORY else (),
            tuple(sorted(unknown_ids)) if method is Method.THEORY else (),
            tuple(conditions[key] for key in sorted(conditions)),
            tuple(boundaries[key] for key in sorted(boundaries)), all_clauses,
            tuple(reason for trace in case_traces for reason in trace.exclusion_reasons),
            merged, tuple(relations), tuple(sorted(set(reasons))), method, registry.registry_hash,
            tuple(sorted(applicable_ids)) if method is Method.NPOV else (),
            tuple(sorted(rejected_ids)) if method is Method.NPOV else (),
            tuple(sorted(unknown_ids)) if method is Method.NPOV else (), registry.source_snapshot_id))
    from theory_discovery.core.stage_interfaces import PredictionBatch
    return (PredictionBatch(method, tuple(predictions), study.freeze_hash, study.action_protocol_hash, boundary_mode),
            PredictionProvenanceArtifact(tuple(provenance), study.freeze_hash, method))


def predict_actions(features, applicability, study):
    """Stage 04 compatibility wrapper around the common source-action predictor."""
    return predict_knowledge(features, applicability, study, Method.THEORY)
