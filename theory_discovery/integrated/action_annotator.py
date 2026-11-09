"""Actual-action annotation with no prediction, Theory or NPOV rule inputs."""
from pathlib import Path

from ..stage_interfaces import AnnotationInput, AnnotationOutput, AnnotationBatch
from ..action_annotator import annotate_cases
from ..compact_action_annotator import CompactActionBackend
from ..annotation_prompts import PROMPT_VERSION
from ..measurement_runtime import configured_backend


def run(inputs: AnnotationInput, *, resume=True) -> AnnotationOutput:
    records, audit = annotate_cases(
        inputs.annotation_cases, inputs.action_protocol, inputs.settings, inputs.runtime,
        inputs.extractor_families,
        lambda: CompactActionBackend(configured_backend(inputs.settings.model, inputs.runtime, PROMPT_VERSION)),
        checkpoint_dir=Path(inputs.runtime.cache_directory) / "completed_cases" if resume else None)
    return AnnotationOutput(AnnotationBatch(records), audit)
