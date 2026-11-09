"""Standard-library measurement primitives. Names describe surface properties only."""
from decimal import Decimal, InvalidOperation
import re
from .workflow_contracts import EvidenceSpan, StructuredValue, StructuredRow, MetadataItem, stable_id, utc_time
from .measurement_contracts import MeasurementResult, Candidate, unknown
from .schemas import require

VERSION = "stdlib-objective-v1"
NUMERIC_PATTERN = r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?!\w|\.\d)"


def spans(text, pattern, flags=0):
    require(type(pattern) is str and 0 < len(pattern) <= 500, "Invalid pattern length")
    compiled = re.compile(pattern, flags)
    found = tuple(EvidenceSpan("before", m.start(), m.end(), m.group()) for m in compiled.finditer(text))
    require(all(s.start < s.end for s in found), "Zero-width matches are not measurement candidates")
    return found


def candidates(case, program):
    matches = spans(case.before_text, program.candidate_pattern)
    return tuple(Candidate(stable_id("candidate", [case.before_hash, s.start, s.end,
                                                  program.candidate_group, VERSION]), s) for s in matches)


def _row(**values):
    return StructuredRow(tuple(MetadataItem(k, v) for k, v in values.items()))


def measure(case, program):
    text = case.before_text
    params = {p.key: p.value for p in program.parameters}
    require(len(params) == len(program.parameters), "Duplicate objective parameters")
    tool = program.objective_tool
    evidence, metadata, scope = (), (), "FULL_BEFORE"
    if tool == "char_count":
        value = len(text)
    elif tool == "whitespace_token_count":
        value = len(re.findall(r"\S+", text))
    elif tool in ("literal_present", "literal_count", "literal_relative_position"):
        literal = params.get("literal")
        require(type(literal) is str and bool(literal), "Literal parameter required")
        require(set(params) <= {"literal", "case_sensitive"}, "Unknown literal parameters")
        require(type(params.get("case_sensitive", True)) is bool, "case_sensitive must be boolean")
        evidence = spans(text, re.escape(literal), 0 if params.get("case_sensitive", True) else re.IGNORECASE)
        if tool == "literal_present":
            value = bool(evidence)
        elif tool == "literal_count":
            value = len(evidence)
        elif not evidence:
            return unknown("Literal absent; position is undefined")
        else:
            value = evidence[0].start / max(1, len(text))
    elif tool in ("regex_present", "regex_count", "regex_matches"):
        require(set(params) == {"pattern"}, "Regex requires only a declared pattern")
        evidence = spans(text, params["pattern"])
        if tool == "regex_present":
            value = bool(evidence)
        elif tool == "regex_count":
            value = len(evidence)
        else:
            value = StructuredValue("text-spans-v1", (), tuple(
                _row(start=s.start, end=s.end, quote=s.quote) for s in evidence))
    elif tool in ("numeric_literals", "single_numeric_value"):
        require(not params, "Numeric-literal format is fixed/versioned")
        evidence = spans(text, NUMERIC_PATTERN)
        try:
            numbers = tuple(Decimal(s.quote) for s in evidence)
            require(all(v.is_finite() for v in numbers), "Nonfinite numeric literal")
        except (InvalidOperation, ValueError):
            return unknown("Invalid decimal literal")
        if tool == "single_numeric_value":
            if len(numbers) != 1:
                return unknown("Requires exactly one unambiguous decimal literal")
            value = float(numbers[0])
        else:
            value = StructuredValue("numeric-literals-v1", (), tuple(
                _row(start=s.start, end=s.end, lexeme=s.quote, decimal=str(v))
                for s, v in zip(evidence, numbers)))
    elif tool in ("line_count", "line_structure"):
        require(not params, "Line structure has no semantic parameters")
        lines = text.splitlines(keepends=True)
        if tool == "line_count":
            value = len(lines)
        else:
            rows, offset = [], 0
            for index, line in enumerate(lines, 1):
                rows.append(_row(line=index, start=offset, end=offset + len(line), text=line))
                offset += len(line)
            value = StructuredValue("line-structure-v1", (), tuple(rows))
    elif tool in ("metadata_value", "metadata_age_seconds"):
        require(set(params) == {"field"} and type(params["field"]) is str, "Metadata field required")
        field = next((f for f in case.pre_context if f.name == params["field"]), None)
        if field is None or field.value is None:
            return unknown("Required allowlisted pre-edit metadata unavailable")
        metadata, scope = (field.name,), "METADATA"
        if tool == "metadata_value":
            value = field.value
        else:
            try:
                value = (utc_time(case.context_as_of_utc) - utc_time(field.value)).total_seconds()
                require(value >= 0, "Metadata event is in the future")
            except (TypeError, ValueError):
                return unknown("Metadata timestamp unavailable/incompatible")
    else:
        raise ValueError("Unsupported objective extractor")
    return MeasurementResult(value, "KNOWN", evidence, metadata, scope,
                             "Deterministic " + tool + " over declared Before inputs.", (), None, None)
