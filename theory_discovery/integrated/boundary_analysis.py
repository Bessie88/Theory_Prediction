"""Registered diagnostic using existing feature observations and frozen rules."""
from theory_discovery.core.stage_interfaces import BoundaryAnalysisInput, BoundaryAnalysisOutput
from theory_discovery.induction.boundary_ablation import analyze_boundaries

def run(inputs: BoundaryAnalysisInput) -> BoundaryAnalysisOutput:
    return BoundaryAnalysisOutput(analyze_boundaries(inputs))
