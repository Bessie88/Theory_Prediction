"""Final registered per-editor report; template is filled only in authorized future execution."""
from ..stage_interfaces import FinalReportInput, FinalReportOutput
from ..research_reporting import render_report
from ..schemas import require

def run(inputs: FinalReportInput) -> FinalReportOutput:
    require(inputs.execution_authorized, "Report generation disabled")
    return FinalReportOutput(render_report(inputs))
