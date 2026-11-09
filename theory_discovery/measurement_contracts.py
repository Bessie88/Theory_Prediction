"""Typed operationalization contracts; no execution or I/O on import."""
from dataclasses import dataclass
from .workflow_contracts import (
    Record, Scalar, MeasurementValue, FeatureSpecification, EvidenceSpan,
    MetadataItem, PredictionCase, CaseIdentity, Period, Method, require, nonempty, stable_id,
)

OBJECTIVE_TOOLS = (
    "char_count", "whitespace_token_count", "literal_present", "literal_count",
    "regex_present", "regex_count", "regex_matches", "numeric_literals",
    "single_numeric_value", "line_count", "line_structure", "literal_relative_position",
    "metadata_value", "metadata_age_seconds",
)
CONTEXTUAL_CONSTRUCTS = ("EVALUATION", "SUBJECTIVITY", "ATTRIBUTION",
                         "MEANING_PRESERVATION", "CONTEXTUAL_CATEGORY")
OBJECTIVE_CONSTRUCTS = ("LENGTH", "LITERAL", "REGEX", "NUMERIC_LITERAL",
                        "POSITION", "LINE_STRUCTURE", "PREEDIT_METADATA")


@dataclass(frozen=True)
class PlanStep(Record):
    step_id: str
    tool: str
    required_inputs: tuple[str, ...]
    expected_output: str
    possible_errors: tuple[str, ...]
    fallback: str
    tool_version: str

    def validate(self):
        require(self.tool in ("objective.measure", "objective.candidates", "semantic.measure",
                             "semantic.candidates", "evidence.validate", "output.normalize"),
                "Only named measurement tools are executable")
        require(self.fallback in ("UNKNOWN", "RAISE"), "No invented-value fallback")
        nonempty(self.step_id, self.expected_output, self.tool_version)
        expected = {
            "objective.measure": "MEASUREMENT_RESULT", "objective.candidates": "CANDIDATES",
            "semantic.measure": "MEASUREMENT_RESULT", "semantic.candidates": "MEASUREMENT_RESULT",
            "evidence.validate": "VALIDATED_RESULT", "output.normalize": "FEATURE_VALUE"}
        require(self.expected_output == expected[self.tool], "Step output type differs from responsible tool")


@dataclass(frozen=True)
class ExtractionProgram(Record):
    version: str
    objective_tool: str | None
    parameters: tuple[MetadataItem, ...]
    candidate_pattern: str | None
    candidate_group: str | None
    candidate_coverage: str
    semantic_mode: str
    steps: tuple[PlanStep, ...]

    def validate(self):
        require(self.objective_tool is None or self.objective_tool in OBJECTIVE_TOOLS,
                "Unsupported programmatic measurement")
        require(self.candidate_coverage in ("NOT_APPLICABLE", "OPEN_WORLD", "EXHAUSTIVE_BY_DEFINITION"),
                "Candidate-domain coverage must be explicit")
        require(self.semantic_mode in ("FEATURE_VALUE", "CANDIDATE_PROPERTY", "HYPOTHETICAL_REMOVAL"),
                "Unsupported semantic measurement mode")
        ids = tuple(s.step_id for s in self.steps)
        require(len(ids) == len(set(ids)), "Duplicate measurement step IDs")
        require(tuple(s.tool for s in self.steps)[-2:] ==
                ("evidence.validate", "output.normalize"), "Validation/normalization must be last")
        require(all(set(s.required_inputs) <= {"before_text", "pre_context", "context_as_of_utc", "candidates", "result"}
                    for s in self.steps), "After/action/future inputs are forbidden")


@dataclass(frozen=True)
class ClauseSource(Record):
    clause_id: str
    theory_id: str
    source_rule_ids: tuple[str, ...]
    source_pointer: str
    original_text: str
    role: str
    knowledge_source: Method = Method.THEORY
    declared_role: str | None = None
    original_clause_id: str | None = None

    def validate(self):
        require(self.role in ("CONDITION", "BOUNDARY_UNRESOLVED"), "Native role cannot be guessed")


@dataclass(frozen=True)
class ClauseBinding(Record):
    clause_id: str
    logical_role: str
    expression_json: str
    component_feature_ids: tuple[str, ...]
    decomposition_rationale: str
    fidelity_rationale: str
    resolution: str
    limitations: tuple[str, ...]

    def validate(self):
        require(self.logical_role in (
            "CONDITION", "BOUNDARY_EXCLUDES", "BOUNDARY_REQUIRES", "BOUNDARY_RESTRICTS",
            "BOUNDARY_UNRESOLVED"), "Unknown Condition/Boundary role")
        require(self.resolution in ("RESOLVED", "UNRESOLVED", "NON_OBSERVABLE",
                                    "NO_ADDITIONAL_BOUNDARY"), "Unknown measurement resolution")
        nonempty(self.decomposition_rationale, self.fidelity_rationale)


@dataclass(frozen=True)
class Candidate(Record):
    candidate_id: str
    span: EvidenceSpan


@dataclass(frozen=True)
class MeasurementResult(Record):
    value: MeasurementValue
    status: str
    evidence: tuple[EvidenceSpan, ...]
    metadata_names: tuple[str, ...]
    scope: str
    rationale: str
    uncertainty: tuple[str, ...]
    model_version: str | None
    prompt_version: str | None

    def validate(self):
        require(self.status in ("KNOWN", "UNKNOWN", "ERROR"), "Invalid measurement state")
        require(self.scope in ("SPANS", "METADATA", "FULL_BEFORE", "CANDIDATES", "UNKNOWN"),
                "Invalid evidence scope")
        require((self.status == "KNOWN") == (self.value is not None),
                "Unknown/error cannot carry a guessed feature value")
        require(all(s.side == "before" for s in self.evidence), "After spans forbidden")


@dataclass(frozen=True)
class CandidateConditionLabel(Record):
    label_id: str
    name: str
    definition: str
    match_status: str
    matched_feature_id: str | None
    evidence: tuple[EvidenceSpan, ...]
    explanation: str

    def validate(self):
        require(self.match_status in ("EXISTING", "NEW", "UNCERTAIN"), "Invalid candidate condition status")
        nonempty(self.label_id, self.name, self.definition, self.explanation)
        require(self.evidence and all(s.side == "before" for s in self.evidence), "Before evidence required")


@dataclass(frozen=True)
class CaseConditionLabels(Record):
    identity: CaseIdentity
    matched_condition_ids: tuple[str, ...]
    false_condition_ids: tuple[str, ...]
    unknown_condition_ids: tuple[str, ...]
    candidates: tuple[CandidateConditionLabel, ...]
    status: str
    explanation: str
    before_hash: str
    registry_id: str

    def validate(self):
        require(self.status in ("ANNOTATED", "UNCERTAIN", "UNAVAILABLE"), "Invalid condition-label report")
        require(not (set(self.matched_condition_ids) & set(self.false_condition_ids)) and
                not (set(self.unknown_condition_ids) & (set(self.matched_condition_ids) | set(self.false_condition_ids))),
                "Condition truth groups must not overlap")


@dataclass(frozen=True)
class MeasurementReview(Record):
    clause_id: str
    supporting_case_ids: tuple[str, ...]
    contrasting_case_ids: tuple[str, ...]
    ambiguous_case_ids: tuple[str, ...]
    inconsistencies: tuple[str, ...]
    non_observable_reasons: tuple[str, ...]
    status: str
    unclassified_construction_case_ids: tuple[str, ...] = ()
    context_only_case_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureRegistrySnapshot(Record):
    specifications: tuple[FeatureSpecification, ...]
    snapshot_id: str
    prompt_versions: tuple[str, ...]
    extractor_versions: tuple[str, ...]
    semantic_system_prompt: str
    payload_template_version: str


@dataclass(frozen=True)
class FrozenMeasurements(Record):
    snapshot: FeatureRegistrySnapshot
    clauses: tuple[ClauseSource, ...]
    bindings: tuple[ClauseBinding, ...]
    reviews: tuple[MeasurementReview, ...]
    source_theory_hash: str
    freeze_hash: str
    unresolved_clause_ids: tuple[str, ...]
    review_completed: bool


class MeasurementBackendUnavailable(RuntimeError):
    pass


class MeasurementValidationError(ValueError):
    pass


def unknown(reason, model=None, prompt=None):
    return MeasurementResult(None, "UNKNOWN", (), (), "UNKNOWN", reason, (reason,), model, prompt)


def procedure_for(spec):
    import json
    return ExtractionProgram.from_dict(json.loads(spec.extraction_procedure_json))


def specification_hash(spec):
    return stable_id("feature_specification", spec.to_dict())


def validate_specification(spec):
    nonempty(spec.feature_name, spec.measurement_target, spec.version)
    require(spec.required_input_fields and set(spec.required_input_fields) <=
            {"before_text", "pre_context", "context_as_of_utc"}, "Features may use Before-only inputs")
    require(spec.positive_criteria and spec.negative_criteria and spec.ambiguity_criteria,
            "Positive, negative and ambiguous criteria must be explicit")
    program = procedure_for(spec)
    require(program.version == "procedure-v1" and spec.extractor_version == "measurement-executor-v1",
            "Procedure/executor version not installed")
    required = set(spec.required_input_fields)
    if program.objective_tool == "metadata_value":
        require("pre_context" in required, "Metadata measurement must declare its input")
    elif program.objective_tool == "metadata_age_seconds":
        require({"pre_context", "context_as_of_utc"} <= required, "Timestamp inputs must be declared")
    else:
        require("before_text" in required, "Text measurements must declare Before text")
    require(spec.observable_indicators, "Observable indicators required")
    if spec.value_type == "categorical":
        require(bool(spec.allowed_values), "Categorical feature needs explicit allowed values")
    if spec.value_type == "structured":
        require(spec.structured_schema_name and spec.structured_fields, "Structured output schema required")
        require(len({s.name for s in spec.structured_fields}) == len(spec.structured_fields),
                "Duplicate structured schema fields")
    tools = tuple(s.tool for s in program.steps)
    from .workflow_contracts import MeasurementMethod
    expected = {
        MeasurementMethod.CODE: ("objective.measure", "evidence.validate", "output.normalize"),
        MeasurementMethod.SEMANTIC: ("semantic.measure", "evidence.validate", "output.normalize"),
        MeasurementMethod.HYBRID: ("objective.candidates", "semantic.candidates",
                                  "evidence.validate", "output.normalize"),
    }[spec.measurement_method]
    require(tools == expected, "Execution plan must follow its declared method")
    for step in program.steps:
        external = set(step.required_inputs) - {"candidates", "result"}
        require(external <= required, "Plan step uses undeclared feature inputs")
        minimum = {
            "objective.measure": required, "objective.candidates": {"before_text"},
            "semantic.measure": {"before_text"}, "semantic.candidates": {"before_text", "candidates"},
            "evidence.validate": {"result"}, "output.normalize": {"result"}}[step.tool]
        require(minimum <= set(step.required_inputs), "Plan step omits a responsible tool input")
    if spec.construct_kind in CONTEXTUAL_CONSTRUCTS:
        require(spec.measurement_method is not MeasurementMethod.CODE,
                "Regex/keywords cannot directly establish contextual meaning")
    else:
        require(spec.construct_kind in OBJECTIVE_CONSTRUCTS and
                spec.measurement_method is MeasurementMethod.CODE,
                "Objective constructs must use available deterministic measurement")
    if spec.measurement_method is not MeasurementMethod.CODE:
        from .semantic_features import PROMPT_VERSION
        require(spec.prompt_version == PROMPT_VERSION, "Semantic template version not installed")
    if spec.measurement_method is MeasurementMethod.CODE:
        require(program.objective_tool in OBJECTIVE_TOOLS, "Objective tool required")
        expected_construct = {
            "char_count": "LENGTH", "whitespace_token_count": "LENGTH",
            "literal_present": "LITERAL", "literal_count": "LITERAL",
            "literal_relative_position": "POSITION", "regex_present": "REGEX",
            "regex_count": "REGEX", "regex_matches": "REGEX",
            "numeric_literals": "NUMERIC_LITERAL", "single_numeric_value": "NUMERIC_LITERAL",
            "line_count": "LINE_STRUCTURE", "line_structure": "LINE_STRUCTURE",
            "metadata_value": "PREEDIT_METADATA", "metadata_age_seconds": "PREEDIT_METADATA"}
        require(spec.construct_kind == expected_construct[program.objective_tool],
                "Objective procedure does not measure its declared construct")
        output = {
            "literal_present": ("boolean",), "regex_present": ("boolean",),
            "regex_matches": ("structured",), "numeric_literals": ("structured",),
            "line_structure": ("structured",), "metadata_value": ("boolean", "categorical", "string", "numerical", "number")
        }.get(program.objective_tool, ("numerical", "number"))
        require(spec.value_type in output, "Objective tool/output type mismatch")
        params = {p.key: p.value for p in program.parameters}
        require(len(params) == len(program.parameters), "Duplicate procedure parameters")
        if program.objective_tool.startswith("literal_"):
            require(set(params) <= {"literal", "case_sensitive"} and
                    type(params.get("literal")) is str and bool(params["literal"]),
                    "Literal measurement requires an exact literal")
        elif program.objective_tool.startswith("regex_"):
            require(set(params) == {"pattern"} and type(params["pattern"]) is str and
                    0 < len(params["pattern"]) <= 500, "Regex parameter required")
        elif program.objective_tool.startswith("metadata_"):
            require(set(params) == {"field"} and type(params["field"]) is str and
                    bool(params["field"]), "Metadata field parameter required")
        else:
            require(not params, "Unexpected objective parameters")
    if spec.measurement_method is MeasurementMethod.SEMANTIC:
        require(program.objective_tool is None and program.candidate_pattern is None and
                program.semantic_mode == "FEATURE_VALUE", "Semantic plan cannot hide a candidate/After step")
    if spec.measurement_method is MeasurementMethod.HYBRID:
        nonempty(program.candidate_pattern, program.candidate_group)
        require(not program.parameters and program.objective_tool is None,
                "Hybrid candidates use only the explicit regex domain")
        require(program.semantic_mode in ("CANDIDATE_PROPERTY", "HYPOTHETICAL_REMOVAL"),
                "Hybrid semantic stage must measure a candidate property")
    require(not spec.is_proxy or bool(spec.limitations),
            "A proxy requires explicit construct-validity limitations")
    return program


@dataclass(frozen=True)
class MeasurementRuntime(Record):
    execution_authorized: bool
    models_authorized: bool
    configuration_id: str
    cache_directory: str
    max_text_characters: int = 200000
