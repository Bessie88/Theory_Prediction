"""Later evaluation view -> independent annotation view and deterministic exact diff."""
from ..stage_interfaces import AnnotationDataInput, AnnotationDataOutput
from ..annotation_data import prepare_annotation_cases
from ..schemas import require


def run(inputs: AnnotationDataInput) -> AnnotationDataOutput:
    require(inputs.execution_authorized, "Annotation data/diff preparation disabled")
    return AnnotationDataOutput(prepare_annotation_cases(inputs.outcomes, inputs.later,
        inputs.settings.context_allowlist, inputs.settings.max_text_characters))
