"""Objective candidates + limited contextual judgment + deterministic evidence checks."""
from dataclasses import replace
from theory_discovery.core.workflow_contracts import StructuredValue, StructuredRow, MetadataItem
from .measurement_contracts import MeasurementResult, unknown
from .objective_features import candidates
from .semantic_features import semantic_measure
from .feature_evidence import validate_evidence
from theory_discovery.core.schemas import require


def measure_hybrid(spec, case, program, backend_provider, permit, prepared_candidates=None):
    selected = candidates(case, program) if prepared_candidates is None else prepared_candidates
    exhaustive = program.candidate_coverage == "EXHAUSTIVE_BY_DEFINITION"
    if not selected:
        if spec.value_type == "boolean" and exhaustive:
            return MeasurementResult(False, "KNOWN", (), (), "FULL_BEFORE",
                "EXHAUSTIVE_EMPTY_CANDIDATE_DOMAIN", (), None, None)
        return unknown("No candidates in an incomplete domain; semantic absence is not established")
    if backend_provider is None or not (permit.execution_authorized and permit.model_authorized):
        return unknown("Semantic candidate measurement not authorized/available")
    backend = backend_provider(spec)
    if backend is None:
        return unknown("Semantic candidate backend unavailable")
    rows, results = [], []
    batch = getattr(type(backend), "measure_candidates", None)
    measured = (batch(backend, spec, case, selected, program.semantic_mode) if callable(batch) else
                tuple(semantic_measure(spec, case, backend, permit, c, program.semantic_mode) for c in selected))
    require(len(measured) == len(selected), "Candidate batch must retain every candidate row")
    for candidate, result in zip(selected, measured):
        # Candidate property is boolean even when aggregate feature is structured.
        property_spec = replace(spec, value_type="boolean", allowed_values=(True, False),
                                structured_schema_name="", structured_fields=())
        result = validate_evidence(property_spec, case, result)
        results.append(result)
        rows.append(StructuredRow((
            MetadataItem("candidate_id", candidate.candidate_id),
            MetadataItem("value", result.value if result.status == "KNOWN" else None))))
    if spec.value_type == "structured":
        value = StructuredValue(spec.structured_schema_name,
                                (MetadataItem("exhaustive", exhaustive),), tuple(rows))
        status = "KNOWN"
    else:
        require(spec.value_type == "boolean", "Hybrid aggregation supports boolean or candidate rows")
        values = [r.value for r in results]
        if True in values:
            value, status = True, "KNOWN"
        elif exhaustive and all(r.status == "KNOWN" and r.value is False for r in results):
            value, status = False, "KNOWN"
        else:
            return unknown("No established positive witness; candidate domain/labels incomplete",
                           spec.model_version, spec.prompt_version)
    evidence = {(s.start, s.end, s.quote): s for candidate in selected for s in (candidate.span,)}
    for result in results:
        evidence.update({(s.start, s.end, s.quote): s for s in result.evidence})
    return MeasurementResult(
        value, status, tuple(evidence.values()), (), "CANDIDATES" if spec.value_type == "structured"
        else "FULL_BEFORE", "Candidate properties measured separately; no keyword-to-meaning inference.",
        tuple(dict.fromkeys(u for result in results for u in result.uncertainty)),
        spec.model_version, spec.prompt_version)
