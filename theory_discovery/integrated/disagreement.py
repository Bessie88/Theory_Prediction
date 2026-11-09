"""Evaluation-only qualitative retrieval, preserving abstention and unresolved categories."""
from collections import Counter
from ..stage_interfaces import DisagreementInput, DisagreementOutput, DisagreementReport
from ..disagreement_analysis import analyze_editor
from ..workflow_contracts import MetadataItem

def run(inputs: DisagreementInput) -> DisagreementOutput:
    analysis = analyze_editor(inputs)
    counts = Counter(r.category for r in analysis.disagreements)
    return DisagreementOutput(DisagreementReport(analysis.editor,
        tuple(r.identity.case_id for r in analysis.disagreements),
        tuple(MetadataItem(k, v) for k, v in sorted(counts.items())), "disagreement-v1"), analysis)
