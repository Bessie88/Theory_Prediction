"""Registered diagnostic using existing feature observations and frozen rules."""
from ..stage_interfaces import BoundaryAnalysisInput, BoundaryAnalysisOutput
from ..boundary_ablation import analyze_boundaries

def run(inputs: BoundaryAnalysisInput) -> BoundaryAnalysisOutput:
    return BoundaryAnalysisOutput(analyze_boundaries(inputs))
