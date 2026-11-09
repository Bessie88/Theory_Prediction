"""Shared Condition/Boundary/policy engine -> existing FeatureObservation port."""
from ..stage_interfaces import FeatureExtractionInput, FeatureExtractionOutput, FeatureBatch
from ..feature_spec_registry import FeatureSpecificationRegistry, freeze_measurements
from ..measurement_executor import MeasurementExecutor
from ..feature_review_source import build_feature_review_source
from ..measurement_runtime import configured_backend
from ..feature_cache import SharedFeatureCache
from ..semantic_features import PROMPT_VERSION, CONDITION_LABEL_PROMPT_VERSION, label_before_conditions
from dataclasses import replace
from ..workflow_contracts import MeasurementMethod
from ..schemas import require
from ..theory_application_registry import validate_study


def run(inputs: FeatureExtractionInput) -> FeatureExtractionOutput:
    require(inputs.runtime.execution_authorized, "Feature extraction execution disabled")
    validate_study(inputs.study)
    frozen = inputs.study.measurement_freeze
    rebuilt = freeze_measurements(frozen.snapshot, frozen.clauses, frozen.bindings,
                                  frozen.reviews, frozen.source_theory_hash)
    require(rebuilt == frozen, "Frozen measurement manifest was modified")
    expected = FeatureSpecificationRegistry(inputs.study.theory_features + inputs.study.policy_features).snapshot()
    require(expected == frozen.snapshot, "Study features differ from frozen registry")
    backends = {}
    def provider(spec):
        key = spec.measurement_method
        if key not in backends:
            setting = (inputs.settings.hybrid_model if key is MeasurementMethod.HYBRID
                       else inputs.settings.semantic_model)
            backends[key] = configured_backend(setting, inputs.runtime, PROMPT_VERSION)
        return backends[key]
    executor = MeasurementExecutor(
        True, inputs.runtime.models_authorized, provider,
        SharedFeatureCache(inputs.runtime.cache_directory), inputs.runtime.configuration_id,
        inputs.runtime.max_text_characters)
    # Each semantic request receives ONE case's Before. Future cases in this batch
    # are never inserted into another target's request/context.
    open_coded = inputs.runtime.configuration_id.startswith("open-code-align-critical-v")
    if open_coded:
        from ..open_code_features import execute_case
        observations = tuple(o for case in inputs.later.records for o in execute_case(inputs,case,provider))
    else:
        observations = tuple(executor.execute(spec, case) for case in inputs.later.records
                             for spec in frozen.snapshot.specifications)
    batch = FeatureBatch(inputs.later.records, observations, inputs.study.freeze_hash)
    label_backend = None
    label_failure = None
    setting = inputs.settings.semantic_model if inputs.settings.semantic_model.enabled else inputs.settings.hybrid_model
    if not open_coded and inputs.runtime.models_authorized and setting.enabled:
        try:
            label_backend = configured_backend(replace(setting, prompt_version=CONDITION_LABEL_PROMPT_VERSION,
                max_tokens=min(setting.max_tokens, 2048)), inputs.runtime, CONDITION_LABEL_PROMPT_VERSION)
        except (ValueError, RuntimeError, OSError) as exc:
            label_failure = type(exc).__name__ + ": " + str(exc)
    condition_labels = tuple(label_before_conditions(case, frozen, observations, label_backend) for case in inputs.later.records)
    if label_failure:
        condition_labels = tuple(replace(row, status="UNCERTAIN", explanation=label_failure) for row in condition_labels)
    return FeatureExtractionOutput(batch, build_feature_review_source(batch, frozen.snapshot), condition_labels)
