"""Connected original-theory -> procedure/specification registry stage."""
from ..stage_interfaces import MeasurementPlanningInput, MeasurementPlanningOutput
from ..measurement_planner import plan_measurements, PLANNER_PROMPT_VERSION
from ..measurement_runtime import configured_backend
from ..schemas import require


def run(inputs: MeasurementPlanningInput, *, workers=1, progress=None) -> MeasurementPlanningOutput:
    require(inputs.runtime.execution_authorized, "Measurement planning execution disabled")
    backend = None
    if inputs.settings.planner_enabled:
        backend = configured_backend(inputs.settings.planner_model, inputs.runtime, PLANNER_PROMPT_VERSION)
    return plan_measurements(inputs, backend, workers=workers, progress=progress)
