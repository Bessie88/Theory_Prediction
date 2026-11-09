"""Deterministic structural/evidence validation; never certifies semantic truth."""
import math
from theory_discovery.core.workflow_contracts import StructuredValue
from .measurement_contracts import MeasurementValidationError, unknown
from theory_discovery.core.schemas import require


def check_value(spec, value, case=None):
    if spec.value_type == "boolean":
        require(type(value) is bool, "Boolean value required")
    elif spec.value_type in ("number", "numerical"):
        require(type(value) in (int, float) and math.isfinite(value), "Finite numerical value required")
    elif spec.value_type in ("string", "categorical"):
        require(type(value) is str, "Categorical/string value required")
    elif spec.value_type == "structured":
        require(type(value) is StructuredValue and value.schema_name == spec.structured_schema_name,
                "Structured measurement schema mismatch")
        fields = {s.name: s for s in spec.structured_fields}
        for row in value.rows:
            keys = [f.key for f in row.fields]
            require(len(keys) == len(set(keys)), "Duplicate structured result fields")
            row_values = {f.key: f.value for f in row.fields}
            require(set(row_values) <= set(fields), "Unknown structured result field")
            require({s.name for s in spec.structured_fields if s.required} <= set(row_values),
                    "Missing structured result fields")
            if case is not None and {"start", "end", "quote"} <= set(row_values):
                from theory_discovery.core.workflow_contracts import EvidenceSpan
                EvidenceSpan("before", row_values["start"], row_values["end"], row_values["quote"]).check_text(case.before_text)
            for name, raw in row_values.items():
                field = fields[name]
                if raw is None:
                    require(field.nullable, "Null structured field forbidden")
                else:
                    expected = {"string": str, "integer": int, "number": (int, float), "boolean": bool}[field.value_type]
                    require(type(raw) in expected if isinstance(expected, tuple) else type(raw) is expected,
                            "Incorrect structured field value")
        top = {item.key: item.value for item in value.fields}
        require(set(top) <= {"exhaustive"} and len(top) == len(value.fields),
                "Unknown structured result metadata")
        if "exhaustive" in top:
            require(type(top["exhaustive"]) is bool, "Invalid candidate coverage flag")
    else:
        raise ValueError("Unsupported feature output type")
    if spec.allowed_values:
        require(any(type(value) is type(v) and value == v for v in spec.allowed_values),
                "Value outside allowed categories")


def validate_evidence(spec, case, result):
    if result.status != "KNOWN":
        require(result.value is None, "Unknown result must be null")
        return result
    try:
        check_value(spec, result.value, case)
        require(bool(result.rationale.strip()), "Measurement rationale required")
        metadata = {f.name: f for f in case.pre_context}
        require(set(result.metadata_names) <= set(metadata), "Unknown metadata evidence")
        for span in result.evidence:
            require(span.side == "before", "After evidence prohibited")
            span.check_text(case.before_text)
        keys = [(s.start, s.end) for s in result.evidence]
        require(len(keys) == len(set(keys)), "Duplicate/contradictory evidence spans")
        if result.scope == "SPANS":
            require(bool(result.evidence), "Required text evidence missing")
        elif result.scope == "METADATA":
            require(bool(result.metadata_names), "Metadata evidence missing")
        elif result.scope == "FULL_BEFORE":
            require("FULL_BEFORE" in spec.evidence_requirements,
                    "Full-text assessment not allowed by this feature")
        elif result.scope == "CANDIDATES":
            require(spec.value_type == "structured", "Candidate-level result must be structured")
        else:
            raise ValueError("Known value has no declared evidence scope")
        if "SPANS" in spec.evidence_requirements:
            require(bool(result.evidence), "Feature requires exact source spans")
        if "SPANS_WHEN_TRUE" in spec.evidence_requirements and result.value is True:
            require(bool(result.evidence), "Positive feature value requires source spans")
        if "METADATA" in spec.evidence_requirements:
            require(bool(result.metadata_names), "Feature requires metadata references")
        if spec.measurement_method.value == "CODE":
            require(result.model_version is None, "Objective value cannot cite an LLM model")
        elif spec.measurement_method.value == "HYBRID" and result.model_version is None:
            from .measurement_contracts import procedure_for
            from .objective_features import candidates
            program = procedure_for(spec)
            require(result.value is False and program.candidate_coverage == "EXHAUSTIVE_BY_DEFINITION" and
                    not candidates(case, program) and result.rationale == "EXHAUSTIVE_EMPTY_CANDIDATE_DOMAIN",
                    "Unmeasured semantic property cannot become a known value")
        else:
            require(result.model_version == spec.model_version and result.prompt_version == spec.prompt_version,
                    "Model/prompt version mismatch")
        return result
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise MeasurementValidationError("Invalid measurement/evidence: " + str(exc)) from exc
