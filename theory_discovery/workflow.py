"""Future unified coordinator. No CLI, auto-run, model startup, or import-time I/O."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from .artifact_registry import ArtifactRegistry, file_hash
from .stage_dependencies import BY_NAME, dependency_order
from .workflow_contracts import MetadataItem, require_same_cases
from .schemas import require, stable_id
from .stage_interfaces import DiscoveryWorkspace
from .measurement_contracts import MeasurementRuntime
from .review_contracts import ReviewSubmission


@dataclass(frozen=True)
class StageRunReport:
    editor_id: str
    stage: str
    status: str
    artifact_ids: tuple[str, ...]


class Workflow:
    def __init__(self, config):
        self.config = config
        self.registry = ArtifactRegistry(Path(config.output_directory) / "registry",
                                         config.experiment_id, config.configuration_id)

    def plan(self, stage=None, include_dependencies=True):
        return dependency_order(stage, include_dependencies)

    def _configuration_inputs(self, name, editor):
        c = self.config
        if name == "data_preparation":
            return dict(editor=editor, source=c.source, split=c.split, unit=c.prediction_unit,
                        supplied_vocabulary=c.neutral_action_vocabulary)
        if name == "existing_theory_discovery":
            return dict(settings=c.existing_theory_workflow, workspace=DiscoveryWorkspace(
                str(Path(c.output_directory) / "native" / c.configuration_id /
                    stable_id("editor", editor.to_dict())),
                c.experiment_id, c.configuration_id, c.execution.enabled, c.execution.models_enabled))
        if name in ("measurement_planning", "feature_extraction", "npov_preparation", "npov_adapter", "direct_npov_control"):
            extra = dict(policy=c.npov_policy) if name in ("npov_preparation", "npov_adapter", "direct_npov_control") else {}
            if name not in ("npov_adapter", "direct_npov_control"):
                extra["settings"] = c.feature_extraction
            if name == "measurement_planning":
                extra["application"] = c.theory_application
            return dict(**extra, runtime=MeasurementRuntime(
                c.execution.enabled, c.execution.models_enabled, c.configuration_id,
                str(Path(c.output_directory) / "feature_cache" / c.configuration_id /
                    stable_id("editor", editor.to_dict()) / name),
                c.feature_extraction.max_text_characters))
        if name == "action_protocol":
            return dict(settings=c.action_protocol, execution_authorized=c.execution.enabled)
        if name == "annotation_data_preparation":
            return dict(settings=c.independent_action_annotator, execution_authorized=c.execution.enabled)
        if name == "freeze_study":
            return dict(application=c.theory_application, evaluation_settings=c.evaluation)
        if name in ("applicability", "theory_prediction", "npov_prediction"):
            return dict(execution_authorized=c.execution.enabled)
        if name == "policy_snapshot":
            return dict(policy=c.npov_policy, execution_authorized=c.execution.enabled)
        if name == "independent_action_annotation":
            return dict(settings=c.independent_action_annotator, runtime=MeasurementRuntime(
                c.execution.enabled, c.execution.models_enabled, c.configuration_id,
                str(Path(c.output_directory) / "annotation_cache" / c.configuration_id /
                    stable_id("editor", editor.to_dict())), c.independent_action_annotator.max_text_characters),
                extractor_families=tuple(sorted({m.family for m in
                    (c.feature_extraction.semantic_model, c.feature_extraction.hybrid_model) if m.family})))
        if name in ("review_sampling", "human_verification", "feature_review_import"):
            return dict(settings=c.human_review, execution_authorized=c.execution.enabled)
        if name in ("action_review_export", "feature_review_export"):
            return dict(execution_authorized=c.execution.enabled)
        if name in ("boundary_analysis", "final_report"):
            return dict(execution_authorized=c.execution.enabled)
        if name in ("evaluation", "disagreement_analysis"):
            return dict(settings=c.evaluation, execution_authorized=c.execution.enabled) if name == "evaluation" else dict(settings=c.evaluation)
        return {}

    def _model_metadata(self, name):
        c = self.config
        settings = {
            "existing_theory_discovery": (c.existing_theory_workflow.model,),
            "measurement_planning": ((c.feature_extraction.planner_model,)
                                     if c.feature_extraction.planner_model is not None and
                                     c.feature_extraction.planner_enabled else ()),
            "npov_preparation": ((c.feature_extraction.planner_model,)
                                 if c.feature_extraction.planner_enabled and c.feature_extraction.planner_model else ()),
            "npov_adapter": ((c.npov_policy.conversion_model,)
                             if c.npov_policy.conversion_enabled and c.npov_policy.conversion_model and
                             c.npov_policy.converted_rule_packet_json is None else ()),
            "direct_npov_control": ((c.npov_policy.direct_control_model,)
                                    if c.npov_policy.direct_control_enabled and c.npov_policy.direct_control_model else ()),
            "feature_extraction": (c.feature_extraction.semantic_model, c.feature_extraction.hybrid_model),
            "independent_action_annotation": (c.independent_action_annotator.model,),
        }.get(name, ())
        if any(m.enabled for m in settings):
            require(c.execution.models_enabled, "Global model execution gate is disabled")
        return (tuple(m.model for m in settings if m.model),
                tuple(m.prompt_version for m in settings if m.prompt_version))

    def _upstream(self, spec, editor, known):
        selected = {}
        for producer in spec.dependencies:
            if producer not in known:
                parent = BY_NAME[producer]
                upstream = self._upstream(parent, editor, known)
                result = self.registry.completed(parent, editor, tuple(upstream.values()))
                require(result is not None, "Missing/stale dependency: " + producer)
                known[producer] = result
            selected.update({e.artifact_id: e for e in known[producer].values()})
        return selected

    def run(self, stage=None, *, resume=True, include_dependencies=True, review_submissions=()):
        """Explicit future API. Gates BEFORE file reads/writes or handler imports."""
        self.config.require_execution_ready()
        require(type(review_submissions) is tuple and all(type(r) is ReviewSubmission for r in review_submissions),
                "Only typed external human-review submissions may enter the coordinator")
        require(len({r.editor for r in review_submissions}) == len(review_submissions) and
                all(r.editor in self.config.selected_editors for r in review_submissions),
                "Duplicate/unknown external review editor")
        # Hash trusted, local immutable source snapshots before any resume decision.
        require(file_hash(self.config.source.location) == self.config.source.content_sha256,
                "Input source changed")
        require(file_hash(self.config.npov_policy.snapshot_location) ==
                self.config.npov_policy.snapshot_sha256, "Policy snapshot changed")
        require(file_hash(self.config.npov_policy.metadata_location) ==
                self.config.npov_policy.metadata_sha256, "Policy snapshot metadata changed")
        self.registry.record_configuration(self.config)
        reports = []
        for editor in self.config.selected_editors:
            known = {}
            submission = next((r for r in review_submissions if r.editor == editor), None)
            for spec in self.plan(stage, include_dependencies):
                try:
                    upstream = self._upstream(spec, editor, known)
                    human_responses = (
                        submission.action_responses if submission and spec.name == "human_verification" else
                        submission.feature_responses if submission and spec.name == "feature_review_import" else None)
                    saved = (self.registry.completed(spec, editor, tuple(upstream.values()))
                             if resume and human_responses is None else None)
                    if saved is not None and spec.name == "independent_action_annotation":
                        audit = self.registry.read_completed_port(
                            saved["annotation_audit"], spec.name, "annotation_audit")
                        if any(attempt.validation_status == "INVALID" for attempt in audit.attempts):
                            saved = None
                    if saved is not None:
                        if spec.name == "final_report":
                            from .research_reporting import materialize_report
                            entry = saved["research_report"]
                            record = self.registry.read_completed_port(entry, "final_report", "research_report")
                            materialize_report(record, self.registry.root / "exports" / entry.artifact_id, entry.artifact_id)
                        known[spec.name] = saved
                        reports.append(StageRunReport(editor.editor_id, spec.name, "SKIPPED_COMPLETE",
                                                      tuple(e.artifact_id for e in saved.values())))
                        continue
                    self.registry.invalidate(spec.name, editor)
                    kwargs = self._configuration_inputs(spec.name, editor)
                    for binding in spec.bindings:
                        entry = known[binding.producer][binding.output_field]
                        kwargs[binding.input_field] = self.registry.read_for_binding(entry, spec, binding)
                    if human_responses is not None:
                        kwargs["responses"] = human_responses
                    typed_input = spec.input_type(**kwargs)
                    models, prompts = self._model_metadata(spec.name)
                    module = import_module(".integrated." + spec.module, package=__package__)
                    # npov module has two explicitly different typed entry points.
                    handler = (getattr(module, "prepare") if spec.name == "npov_preparation"
                               else getattr(module, "predict") if spec.name == "npov_prediction"
                               else getattr(module, "run"))
                    result = (handler(typed_input, resume=resume) if spec.name == "independent_action_annotation"
                              else handler(typed_input))
                    require(type(result) is spec.output_type, "Invalid stage result")
                    self._validate_join(spec.name, typed_input, result)
                    registered = self.registry.commit(
                        spec, editor, result, tuple(upstream.values()),
                        generation=(MetadataItem("configuration_id", self.config.configuration_id),
                                    MetadataItem("stage_version", spec.version),
                                    MetadataItem("external_review_submission_hash",
                                        stable_id("human_submission", human_responses.to_dict())
                                        if human_responses is not None else None)),
                        models=models, prompts=prompts)
                    if spec.name == "final_report":
                        from .research_reporting import materialize_report
                        entry = registered["research_report"]
                        materialize_report(result.research_report, self.registry.root / "exports" / entry.artifact_id, entry.artifact_id)
                    known[spec.name] = registered
                    reports.append(StageRunReport(editor.editor_id, spec.name, "COMPLETE",
                                                  tuple(e.artifact_id for e in registered.values())))
                except Exception as exc:
                    self.registry.record_failure(spec.name, editor, exc)
                    raise
        if stage is None or stage == "final_report":
            self.generate_cohort_report()
        return tuple(reports)

    @staticmethod
    def _validate_join(name, inputs, result):
        if name == "annotation_data_preparation":
            require_same_cases(inputs.outcomes.records, result.annotation_cases.cases)
        if name == "feature_review_import":
            require({r.subject_id: r.identity for r in inputs.feature_review_source.records} ==
                    {r.subject_id: r.identity for r in result.feature_reviews.records},
                    "Feature human review subjects differ")
        if name == "feature_extraction":
            require_same_cases(inputs.later.records, result.features.cases)
        if name == "theory_prediction":
            require_same_cases(inputs.features.cases, result.theory_predictions.predictions)
        if name == "npov_prediction":
            require_same_cases(inputs.features.cases, result.npov_predictions.predictions)
        if name in ("theory_prediction", "npov_prediction"):
            batch = (result.theory_predictions if name == "theory_prediction"
                     else result.npov_predictions)
            cases = {c.identity.case_id: c for c in inputs.features.cases}
            observations = {o.observation_id: o for o in inputs.features.observations}
            require(batch.study_spec_hash == inputs.study.freeze_hash ==
                    inputs.features.study_spec_hash and
                    batch.action_protocol_hash == inputs.study.action_protocol_hash, "Study/Action protocol freeze mismatch")
            for prediction in batch.predictions:
                require(prediction.original_input_hash ==
                        cases[prediction.identity.case_id].before_hash, "Prediction input mismatch")
                selected = [observations[key] for key in prediction.feature_observation_ids]
                require(all(o.identity == prediction.identity for o in selected),
                        "Prediction cites another case's feature")
                require(set(prediction.matched_feature_ids) <= {o.feature_id for o in selected},
                        "Prediction feature lineage is incomplete")
        if name == "direct_npov_control":
            require_same_cases(inputs.later.records, result.direct_npov_control.records)
        if name == "independent_action_annotation":
            require_same_cases(inputs.annotation_cases.cases, result.annotations.annotations)
        if name == "human_verification":
            # Unselected cases need explicit NOT_SELECTED records, never silent loss.
            require_same_cases(inputs.annotations.annotations, result.reviews.reviews)

    def invalidate(self, stage, editor):
        self.config.require_execution_ready()
        require(editor in self.config.selected_editors, "Editor outside configuration")
        self.registry.invalidate(stage, editor)


    def generate_cohort_report(self, transfers=()):
        """Future multi-editor aggregation in the SAME registry, never fake editor identities."""
        self.config.require_execution_ready()
        from .research_reporting import compare_editors
        spec = BY_NAME["final_report"]
        reports, entries = [], []
        for editor in self.config.selected_editors:
            upstream = self._upstream(spec, editor, {})
            saved = self.registry.completed(spec, editor, tuple(upstream.values()))
            require(saved is not None, "Missing/stale per-editor final report")
            entry = saved["research_report"]
            reports.append(self.registry.read_completed_port(entry, "final_report", "research_report"))
            entries.append(entry)
        result = compare_editors(tuple(reports), transfers)
        entry = self.registry.commit_cohort(result, tuple(entries))
        from .research_reporting import materialize_report
        materialize_report(result, self.registry.root / "exports" / entry.artifact_id, entry.artifact_id)
        return entry
