"""One JSON (YAML-subset) study configuration; loading never starts work."""
from dataclasses import dataclass
from pathlib import Path
import json
from .workflow_contracts import (
    Record, EditorKey, ActionDefinition, MetadataItem, utc_time, nonempty, unique,
)
from .schemas import require, stable_id
from .application_contracts import ApplicationSettings
from .action_comparison import ActionProtocolSettings
from .review_contracts import ReviewSettings


@dataclass(frozen=True)
class ModelSettings(Record):
    enabled: bool
    model: str | None
    endpoint: str | None
    prompt_version: str | None
    temperature: float
    max_tokens: int
    retries: int
    family: str | None = None
    revision: str | None = None

    def validate(self):
        require(self.max_tokens > 0 and self.retries >= 0, "Invalid model budget")
        if self.enabled:
            nonempty(self.model, self.endpoint, self.prompt_version)


@dataclass(frozen=True)
class SourceSettings(Record):
    format: str
    location: str | None
    content_sha256: str | None
    container_format: str = "jsonl"
    field_map: tuple[MetadataItem, ...] = ()
    ordering: str = "SORT_CHRONOLOGICALLY"

    def validate(self):
        require(self.container_format in ("jsonl", "json", "tsv"), "Unsupported local container")
        unique(tuple(item.key for item in self.field_map))
        require(all(type(item.value) is str and item.value for item in self.field_map),
                "Field map values must name source columns")
        require(self.ordering in ("SORT_CHRONOLOGICALLY", "REQUIRE_CHRONOLOGICAL"), "Invalid ordering")
        require(self.format == "wikipedia_revision_history", "Only real revision history is allowed")
        if self.content_sha256 is not None:
            require(len(self.content_sha256) == 64 and
                    all(c in "0123456789abcdef" for c in self.content_sha256), "Invalid source hash")


@dataclass(frozen=True)
class SplitSettings(Record):
    cutoff_utc: str | None
    early_comparison: str
    later_comparison: str
    early_ratio: float | None = None

    def validate(self):
        require((self.early_comparison, self.later_comparison) == ("<", ">="),
                "Only two chronological periods are supported")
        require(self.cutoff_utc is None or self.early_ratio is None, "Choose time OR ratio cutoff")
        if self.cutoff_utc is not None:
            utc_time(self.cutoff_utc)
        if self.early_ratio is not None:
            require(0 < self.early_ratio < 1, "Ratio must be strictly between zero and one")

    def require_configured(self):
        require(self.cutoff_utc is not None or self.early_ratio is not None,
                "Configure time or ratio cutoff explicitly")


@dataclass(frozen=True)
class UnitSettings(Record):
    kind: str | None
    version: str | None
    before_only_enumeration: bool
    unsupported_edit_policy: str
    pre_context_allowlist: tuple[str, ...]

    def validate(self):
        require(self.before_only_enumeration, "Prediction inputs/identity must not use target After")
        require(self.kind in (None, "provided_sentence_pair", "sentence_v1", "whole_revision"),
                "Unknown prediction-unit mode")
        require(self.unsupported_edit_policy == "RETAIN_UNKNOWN", "Unsupported edits must remain visible")
        unique(self.pre_context_allowlist)


@dataclass(frozen=True)
class DiscoverySettings(Record):
    module: str
    constructor_module: str
    parameters: tuple[MetadataItem, ...]
    model: ModelSettings
    llm_parameters: tuple[MetadataItem, ...] = ()
    boundary_workers: int = 4
    integration_workers: int = 4

    def validate(self):
        require(1 <= self.boundary_workers <= 4 and self.integration_workers >= 1, "Invalid worker count")
        unique(tuple(x.key for x in self.llm_parameters))
        require(self.module == "theory_discovery.parallel_discovery" and
                self.constructor_module == "theory_discovery.hierarchical_theory",
                "Existing natural induction/construction must be retained")
        keys = tuple(x.key for x in self.parameters)
        unique(keys)
        require(not set(keys) & {"input_path", "input_format", "output_dir", "cache_dir",
                                "split", "allow_real_data", "llm", "pre_context_keys"},
                "Discovery paths/gates are derived centrally, not overridden")


@dataclass(frozen=True)
class ExtractionSettings(Record):
    semantic_model: ModelSettings
    hybrid_model: ModelSettings
    code_first: bool
    planner_enabled: bool = False
    planner_model: ModelSettings | None = None
    seed_specifications_json: tuple[str, ...] = ()
    seed_bindings_json: tuple[str, ...] = ()
    max_text_characters: int = 200000

    def validate(self):
        require(self.code_first, "Objective extraction has priority")
        require(self.max_text_characters > 0, "Invalid measurement context budget")
        if self.planner_enabled:
            require(self.planner_model is not None and self.planner_model.enabled,
                    "Explicit enabled planner model required")


@dataclass(frozen=True)
class AnnotationSettings(Record):
    model: ModelSettings
    version: str
    prompt_version: str
    max_text_characters: int = 200000
    context_allowlist: tuple[str, ...] = ()
    prefer_distinct_family: bool = True
    same_family_rationale: str | None = None

    def validate(self):
        nonempty(self.version, self.prompt_version)
        require(self.max_text_characters > 0, "Invalid annotation input budget")
        unique(self.context_allowlist)


@dataclass(frozen=True)
class PolicySettings(Record):
    source: str | None
    snapshot_location: str | None
    snapshot_sha256: str | None
    version: str | None
    metadata_location: str | None = None
    metadata_sha256: str | None = None
    conversion_enabled: bool = False
    conversion_model: ModelSettings | None = None
    converted_rule_packet_json: str | None = None
    seed_specifications_json: tuple[str, ...] = ()
    seed_bindings_json: tuple[str, ...] = ()
    direct_control_enabled: bool = False
    direct_control_model: ModelSettings | None = None

    def validate(self):
        if self.source is not None:
            from .policy_contracts import OFFICIAL_NPOV_URL
            require(self.source == OFFICIAL_NPOV_URL, "Only official English NPOV is supported")
        if self.conversion_enabled:
            require(self.conversion_model is not None and self.conversion_model.enabled,
                    "Enabled conversion needs a centrally configured model")
        if self.direct_control_enabled:
            require(self.direct_control_model is not None and self.direct_control_model.enabled,
                    "Enabled diagnostic needs a centrally configured model")


@dataclass(frozen=True)
class EvaluationSettings(Record):
    version: str
    retain_all_cases: bool
    revision_clustering: bool
    human_review_policy: str
    gold_policy: str = "HUMAN_PREFERRED"
    protocol_frozen: bool = False

    def validate(self):
        nonempty(self.version, self.human_review_policy)
        require(self.gold_policy in ("MODEL_ONLY", "HUMAN_PREFERRED", "HUMAN_ONLY"), "Invalid gold policy")
        require(self.retain_all_cases and self.revision_clustering,
                "All-case denominator and revision clustering required")


@dataclass(frozen=True)
class ExecutionSettings(Record):
    enabled: bool
    models_enabled: bool
    workers: int

    def validate(self):
        require(self.workers > 0, "Invalid worker count")
        require(self.enabled or not self.models_enabled, "Models require workflow authorization")


@dataclass(frozen=True)
class WorkflowConfig(Record):
    experiment_id: str
    source: SourceSettings
    selected_editors: tuple[EditorKey, ...]
    split: SplitSettings
    prediction_unit: UnitSettings
    existing_theory_workflow: DiscoverySettings
    feature_extraction: ExtractionSettings
    independent_action_annotator: AnnotationSettings
    npov_policy: PolicySettings
    evaluation: EvaluationSettings
    neutral_action_vocabulary: tuple[ActionDefinition, ...]
    output_directory: str
    execution: ExecutionSettings
    theory_application: ApplicationSettings = ApplicationSettings()
    action_protocol: ActionProtocolSettings = ActionProtocolSettings()
    human_review: ReviewSettings = ReviewSettings()

    def validate(self):
        nonempty(self.experiment_id, self.output_directory)
        unique(tuple((e.wiki_id, e.editor_id) for e in self.selected_editors))
        unique(tuple(a.action_id for a in self.neutral_action_vocabulary))

    @property
    def configuration_id(self):
        return stable_id("workflow_config", [self.schema_version, self.to_dict()])

    def require_execution_ready(self):
        require(self.execution.enabled, "Execution disabled: implementation-only configuration")
        require(bool(self.selected_editors), "Select verified human editors explicitly")
        self.split.require_configured()
        require(self.evaluation.protocol_frozen, "Freeze evaluation policy before execution")
        require(self.action_protocol.comparison.established_before_later,
                "Freeze common Action comparison/guidelines before Later execution")
        nonempty(self.source.location, self.source.content_sha256,
                 self.prediction_unit.kind, self.prediction_unit.version,
                 self.npov_policy.source, self.npov_policy.snapshot_location,
                 self.npov_policy.snapshot_sha256, self.npov_policy.version,
                 self.npov_policy.metadata_location, self.npov_policy.metadata_sha256)
        # Source hashes, editor verification and field availability are checked by
        # the explicit future DataPreparation call. No source is opened here.


def load_workflow_config(path):
    """Explicit file-read API, not invoked at import or by a CLI."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for parent, key in ((raw["source"], "location"),
                        (raw["npov_policy"], "snapshot_location"),
                        (raw["npov_policy"], "metadata_location"), (raw, "output_directory")):
        if parent.get(key):
            parent[key] = str((path.parent / parent[key]).resolve())
    for item in raw["existing_theory_workflow"].get("llm_parameters", []):
        if item["key"] == "tokenizer_path" and item["value"]:
            item["value"] = str((path.parent / item["value"]).resolve())
    return WorkflowConfig.from_dict(raw)
