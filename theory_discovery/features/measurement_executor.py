"""Interpreter for finite extraction programs. No eval, exec, shell or generated code."""
import re
from theory_discovery.core.workflow_contracts import (
    FeatureObservation, ObservationStatus, MetadataItem, PredictionCase, MeasurementMethod,
    feature_observation_id_for,
)
from .measurement_contracts import (
    validate_specification, specification_hash, unknown, MeasurementValidationError, MeasurementResult,
)
from .objective_features import measure, candidates, VERSION as OBJECTIVE_VERSION
from .semantic_features import semantic_measure, ModelPermit, PROMPT_VERSION
from .hybrid_features import measure_hybrid
from .feature_evidence import validate_evidence, check_value
from .feature_cache import NoFeatureCache
from theory_discovery.core.schemas import require, stable_id

EXECUTOR_VERSION = "measurement-executor-v1"
STEP_VERSIONS = {
    "objective.measure": OBJECTIVE_VERSION, "objective.candidates": OBJECTIVE_VERSION,
    "semantic.measure": PROMPT_VERSION, "semantic.candidates": PROMPT_VERSION,
    "evidence.validate": "evidence-v1", "output.normalize": "normalize-v1",
}


class MeasurementExecutor:
    def __init__(self, execution_authorized=False, model_authorized=False, backend_provider=None,
                 cache=None, configuration_id="", max_text_characters=200000):
        self.authorized = execution_authorized
        self.permit = ModelPermit(execution_authorized, model_authorized)
        self.backend_provider = backend_provider
        self.cache = cache or NoFeatureCache()
        self.configuration_id = configuration_id
        self.max_text_characters = max_text_characters

    def execute(self, spec, case):
        require(self.authorized, "Measurement execution disabled")
        require(type(case) is PredictionCase, "Extractor accepts PredictionCase only")
        program = validate_specification(spec)
        require(all(step.tool_version == STEP_VERSIONS[step.tool] for step in program.steps),
                "Frozen tool version not installed")
        spec_hash = specification_hash(spec)
        input_hash = stable_id("feature_input", case.to_dict())
        key = {"executor": EXECUTOR_VERSION, "configuration_id": self.configuration_id,
               "specification": spec.to_dict(), "input": case.to_dict(), "input_hash": input_hash,
               "system_prompt_version": PROMPT_VERSION}
        def compute():
            if len(case.before_text) > self.max_text_characters:
                result = unknown("Before exceeds configured measurement limit; no text truncated")
            else:
                result = self._interpret(spec, case, program)
            metadata = (MetadataItem("executor_version", EXECUTOR_VERSION),
                        MetadataItem("model_version", result.model_version),
                        MetadataItem("prompt_version", result.prompt_version),
                        MetadataItem("evidence_scope", result.scope),
                        MetadataItem("rationale", result.rationale),
                        MetadataItem("input_context_hash", input_hash))
            return FeatureObservation(
                feature_observation_id_for(case.identity, spec.feature_id, input_hash,
                                           spec_hash, spec.measurement_method),
                case.identity, spec.feature_id, result.value, result.evidence, result.metadata_names,
                spec.measurement_method, ObservationStatus(result.status), result.uncertainty,
                metadata, input_hash, spec_hash)
        observation = self.cache.get_or_compute(key, compute)
        require(observation.identity == case.identity and observation.feature_id == spec.feature_id and
                observation.input_hash == input_hash and observation.specification_hash == spec_hash and
                observation.extraction_method is spec.measurement_method, "Cached feature lineage mismatch")
        for span in observation.evidence:
            span.check_text(case.before_text)
        require(set(observation.context_evidence_names) <= {f.name for f in case.pre_context},
                "Cached metadata evidence not in pre-edit view")
        if observation.status is ObservationStatus.KNOWN:
            metadata = {m.key: m.value for m in observation.extraction_metadata}
            require({"rationale", "evidence_scope", "model_version", "prompt_version"} <= set(metadata),
                    "Cached measurement metadata missing")
            validate_evidence(spec, case, MeasurementResult(
                observation.extracted_value, "KNOWN", observation.evidence,
                observation.context_evidence_names, metadata["evidence_scope"], metadata["rationale"],
                observation.uncertainty, metadata["model_version"], metadata["prompt_version"]))
        return observation

    def _interpret(self, spec, case, program):
        result, selected = None, None
        try:
            for step in program.steps:
                if step.tool == "objective.measure":
                    result = measure(case, program)
                elif step.tool == "objective.candidates":
                    selected = candidates(case, program)
                elif step.tool == "semantic.measure":
                    backend = None
                    if self.permit.model_authorized and self.backend_provider is not None:
                        backend = self.backend_provider(spec)
                    result = semantic_measure(spec, case, backend, self.permit, mode=program.semantic_mode)
                elif step.tool == "semantic.candidates":
                    require(selected is not None, "Objective candidate step must run first")
                    result = measure_hybrid(spec, case, program, self.backend_provider,
                                            self.permit, selected)
                elif step.tool == "evidence.validate":
                    result = validate_evidence(spec, case, result)
                elif step.tool == "output.normalize":
                    # Exact typed normalization only. No coercion, fabricated spans or value repair.
                    require(result is not None, "Missing measurement")
            return result
        except (ValueError, TypeError, KeyError, OverflowError, RuntimeError, OSError, re.error) as exc:
            if step.fallback == "RAISE":
                raise
            return unknown(type(exc).__name__ + ": measurement failed validation; no label inferred",
                           spec.model_version if spec.measurement_method is not MeasurementMethod.CODE else None,
                           spec.prompt_version if spec.measurement_method is not MeasurementMethod.CODE else None)
