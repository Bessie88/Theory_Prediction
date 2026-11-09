"""Final registered per-editor report; template is filled only in authorized future execution."""
from theory_discovery.core.stage_interfaces import FinalReportInput, FinalReportOutput
from theory_discovery.evaluation.research_reporting import render_report
from theory_discovery.core.schemas import require

def run(inputs: FinalReportInput) -> FinalReportOutput:
    require(inputs.execution_authorized, "Report generation disabled")
    return FinalReportOutput(render_report(inputs))
