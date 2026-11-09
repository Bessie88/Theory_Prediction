"""Deterministic three-valued application of frozen original-theory declarations."""
import json
from theory_discovery.core.workflow_contracts import (
    ApplicabilityDecision, Method, Period, ObservationStatus, stable_id, require,
)
from theory_discovery.core.application_contracts import (
    FeatureWitness, ClauseEvaluation, ActionGateTrace, TheoryApplicabilityTrace,
    TheoryFeatureObservations,
)
from theory_discovery.features._support.condition_normalization import evaluate_expression, referenced_features
from theory_discovery.features._support.measurement_contracts import specification_hash, MeasurementResult
from theory_discovery.features._support.feature_evidence import validate_evidence
from theory_discovery.induction._support.theory_logic import evaluate_logic
from .theory_application_registry import validate_study

VERSION = "deterministic-theory-applicability-v1"


def index_case_features(case, observations, specifications):
    valid, witnesses = {}, {}
    expected_input = stable_id("feature_input", case.to_dict())
    for observation in observations:
        require(observation.identity == case.identity, "Cross-case observation supplied")
        require(observation.feature_id in specifications, "Unknown observed feature specification")
        spec = specifications[observation.feature_id]
        require(observation.input_hash == expected_input and
                observation.specification_hash == specification_hash(spec) and
                observation.extraction_method is spec.measurement_method,
                "Stale/context-mismatched feature observation")
        reasons = []
        accepted = True
        try:
            for span in observation.evidence:
                span.check_text(case.before_text)
            require(set(observation.context_evidence_names) <= {field.name for field in case.pre_context},
                    "Unknown pre-edit metadata reference")
        except (ValueError, TypeError, KeyError, OverflowError):
            accepted = False
            reasons.append("INVALID_FEATURE_EVIDENCE_OR_VALUE")
        if observation.status is ObservationStatus.KNOWN:
            try:
                meta = {m.key: m.value for m in observation.extraction_metadata}
                require(len(meta) == len(observation.extraction_metadata), "Duplicate feature metadata")
                validate_evidence(spec, case, MeasurementResult(
                    observation.extracted_value, "KNOWN", observation.evidence,
                    observation.context_evidence_names, meta["evidence_scope"], meta["rationale"],
                    observation.uncertainty, meta["model_version"], meta["prompt_version"]))
            except (ValueError, TypeError, KeyError, OverflowError):
                accepted = False
                reasons.append("INVALID_FEATURE_EVIDENCE_OR_VALUE")
        else:
            reasons.append("FEATURE_" + observation.status.value)
        if accepted:
            valid[observation.feature_id] = observation
        witnesses[observation.feature_id] = FeatureWitness(
            observation.feature_id, observation.observation_id,
            observation.extracted_value if accepted and observation.status is ObservationStatus.KNOWN else None,
            observation.status.value if accepted else "INVALID",
            observation.evidence if accepted else (), observation.context_evidence_names if accepted else (),
            accepted, tuple(sorted(set(reasons))))
    return valid, witnesses


def missing_witness(feature_id):
    return FeatureWitness(feature_id, None, None, "MISSING", (), (), False, ("MISSING_FEATURE_OBSERVATION",))


def apply_knowledge(features, study, method=Method.THEORY, boundary_mode="WITH_BOUNDARIES"):
    from theory_discovery.induction._support.boundary_ablation import evaluate_application, WITH, WITHOUT
    require(boundary_mode in (WITH, WITHOUT) and (method is Method.THEORY or boundary_mode == WITH),
            "Boundary variant is supported for Theory only")
    validate_study(study)
    registry = study.theory_registry if method is Method.THEORY else study.policy_registry
    if registry is None:
        from .policy_contracts import PolicySourceUnavailable
        raise PolicySourceUnavailable("POLICY_SOURCE_UNAVAILABLE: no frozen policy registry")
    require(registry.knowledge_source is method, "Applicability knowledge source mismatch")
    source_measurements = (study.measurement_freeze if method is Method.THEORY
                           else study.policy_measurement_freeze)
    require(features.study_spec_hash == study.freeze_hash, "Feature/study freeze mismatch")
    require(all(c.identity.period is Period.LATER and c.identity.editor == registry.editor for c in features.cases),
            "Only this editor's Later Before cases are eligible")
    specifications = {s.feature_id: s for s in study.measurement_freeze.snapshot.specifications}
    require(len(specifications) == len(study.measurement_freeze.snapshot.specifications), "Duplicate feature definition")
    binding_map = {b.clause_id: b for b in source_measurements.bindings}
    theory_feature_ids = set()
    for theory in registry.theories:
        for clause_id in theory.clause_ids:
            theory_feature_ids.update(binding_map[clause_id].component_feature_ids)
        for directive in theory.directives:
            if directive.target_feature_id is not None:
                theory_feature_ids.add(directive.target_feature_id)
    by_case = {c.identity.case_id: [] for c in features.cases}
    require(len(by_case) == len(features.cases), "Duplicate Later case")
    for obs in features.observations:
        require(obs.identity.case_id in by_case, "Observation outside Later manifest")
        by_case[obs.identity.case_id].append(obs)
    artifact_hash = stable_id("feature_batch", features.to_dict())
    decisions, traces = [], []
    for case in features.cases:
        observed = by_case[case.identity.case_id]
        require(len({o.feature_id for o in observed}) == len(observed), "Duplicate feature observations")
        valid, witnesses = index_case_features(case, observed, specifications)
        for theory in registry.theories:
            clauses, values, missing, observed_ids, used = [], {}, set(), set(), set()
            for clause_id in theory.clause_ids:
                binding = binding_map[clause_id]
                expression = json.loads(binding.expression_json)
                needed = referenced_features(expression)
                selected = tuple(witnesses.get(key, missing_witness(key)) for key in sorted(needed))
                used |= needed
                observed_ids.update(w.observation_id for w in selected if w.observation_id)
                missing.update(w.feature_id for w in selected if
                               w.observation_status != "KNOWN" or not w.valid)
                truth = evaluate_expression(expression, valid)
                # Unresolved meaning remains UNKNOWN even if a malformed imported
                # binding attempts to encode a usable-looking value.
                if binding.resolution in ("UNRESOLVED", "NON_OBSERVABLE"):
                    truth = None
                values[clause_id] = truth
                satisfied = (None if truth is None else not truth) if binding.logical_role == "BOUNDARY_EXCLUDES" else truth
                clauses.append(ClauseEvaluation(clause_id, binding.logical_role, truth,
                                                satisfied, binding.resolution, selected))
            if theory.logic_resolution == "UNRESOLVED":
                truth, nodes = None, ()
                reasons = ("UNRESOLVED_SOURCE_THEORY_LOGIC",)
            else:
                boundary_ids = {c.clause_id for c in clauses if c.logical_role.startswith("BOUNDARY_")}
                truth, nodes = evaluate_application(json.loads(theory.applicability_expression_json), values, boundary_ids, boundary_mode)
                reasons = ("APPLICABLE" if truth is True else
                           "NOT_APPLICABLE" if truth is False else "APPLICABILITY_UNKNOWN",)
            gates = []
            for directive in theory.directives:
                gate, gate_nodes = evaluate_application(json.loads(theory.applicability_expression_json), values,
                    {c.clause_id for c in clauses if c.logical_role.startswith("BOUNDARY_")}, boundary_mode,
                    json.loads(directive.when_expression_json))
                gates.append(ActionGateTrace(directive.directive_id, directive.source_action_id, gate, gate_nodes))
                if directive.target_feature_id is not None:
                    key = directive.target_feature_id
                    witness = witnesses.get(key, missing_witness(key))
                    used.add(key)
                    if witness.observation_id:
                        observed_ids.add(witness.observation_id)
                    if witness.observation_status != "KNOWN" or not witness.valid:
                        missing.add(key)
            decision_id = stable_id("theory_applicability", {
                "version": VERSION, "case": case.identity.to_dict(), "theory": theory.theory_id,
                "study": study.freeze_hash, "knowledge_source": method.value, "boundary_mode": boundary_mode, "observations": sorted(observed_ids),
                "feature_input_artifact_hash": artifact_hash})
            status = "TRUE" if truth is True else "FALSE" if truth is False else "UNKNOWN"
            decisions.append(ApplicabilityDecision(
                case.identity, method, theory.theory_id, theory.theory_id,
                theory.source_rule_ids, status, tuple(sorted(observed_ids)), tuple(sorted(used)), decision_id))
            exclusions = tuple(c.clause_id + ": EXCLUSION_MEASURED_TRUE; scope follows frozen source logic"
                               for c in clauses if c.logical_role == "BOUNDARY_EXCLUDES" and c.measured_truth is True)
            traces.append(TheoryApplicabilityTrace(case.identity, decision_id, theory.theory_id, tuple(clauses),
                tuple(nodes), tuple(gates), tuple(sorted(missing)), exclusions, reasons, method))
    selected_observations = tuple(o for o in features.observations if o.feature_id in theory_feature_ids)
    selected = TheoryFeatureObservations(tuple(c.identity for c in features.cases), selected_observations,
                                        study.freeze_hash, artifact_hash, method)
    from theory_discovery.core.stage_interfaces import ApplicabilityBatch
    return ApplicabilityBatch(tuple(decisions), tuple(traces), study.freeze_hash, artifact_hash, method, boundary_mode), selected


def apply_theories(features, study):
    """Compatibility entry point; both sources execute apply_knowledge."""
    return apply_knowledge(features, study, Method.THEORY)
