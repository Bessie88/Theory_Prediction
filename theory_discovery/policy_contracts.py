"""Versioned official-policy source/rule contracts. No policy text is bundled."""
from dataclasses import dataclass
from .workflow_contracts import Record, SourceReference, ActionDefinition, PredictionCase, ActionInstance
from .schemas import require, stable_id

OFFICIAL_NPOV_URL = "https://en.wikipedia.org/wiki/Wikipedia:Neutral_point_of_view"
POLICY_TITLE = "Wikipedia:Neutral point of view"


class PolicySourceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicySection(Record):
    section_id: str
    title: str
    anchor: str
    start: int
    end: int

    def validate(self):
        require(bool(self.section_id) and bool(self.title) and 0 <= self.start < self.end,
                "Invalid supplied policy section")


@dataclass(frozen=True)
class PolicySnapshotMetadata(Record):
    snapshot_id: str
    canonical_url: str
    revision_url: str
    revision_id: str
    revision_timestamp_utc: str
    captured_at_utc: str
    language: str
    page_title: str
    content_sha256: str
    content_format: str
    verification_status: str
    verified_by: str
    verification_evidence: tuple[str, ...]
    sections: tuple[PolicySection, ...]

    def validate(self):
        from urllib.parse import urlsplit, parse_qs
        from .workflow_contracts import utc_time
        require(self.canonical_url == OFFICIAL_NPOV_URL and self.language == "en" and
                self.page_title == POLICY_TITLE, "Only the supplied official English NPOV page is supported")
        url = urlsplit(self.revision_url)
        query = parse_qs(url.query)
        require(url.scheme == "https" and url.netloc == "en.wikipedia.org" and
                url.path == "/w/index.php" and query.get("oldid") == [self.revision_id] and
                query.get("title") == ["Wikipedia:Neutral_point_of_view"],
                "A revision-pinned official URL is required; no URL is fetched")
        require(self.revision_id.isdigit() and self.content_format in ("text", "wikitext"),
                "Explicit revision identity/local content format required")
        utc_time(self.revision_timestamp_utc)
        require(utc_time(self.captured_at_utc) >= utc_time(self.revision_timestamp_utc),
                "Snapshot capture predates policy revision")
        require(self.verification_status == "VERIFIED_OFFICIAL" and self.verified_by.strip() and
                self.verification_evidence, "External source-verification attestation required")
        require(len(self.content_sha256) == 64 and
                all(c in "0123456789abcdef" for c in self.content_sha256), "Policy content hash required")
        require(self.snapshot_id == stable_id("npov_snapshot", [
            self.canonical_url, self.revision_id, self.content_sha256]), "Unstable policy snapshot ID")
        require(self.sections and len({s.section_id for s in self.sections}) == len(self.sections),
                "Supplied policy section map required")


@dataclass(frozen=True)
class PolicySnapshot(Record):
    metadata: PolicySnapshotMetadata
    content: str
    local_reference: SourceReference
    metadata_sha256: str

    def validate(self):
        from hashlib import sha256
        require(sha256(self.content.encode("utf-8")).hexdigest() == self.metadata.content_sha256 and
                self.local_reference.sha256 == self.metadata.content_sha256 and
                self.local_reference.source_id == self.metadata.snapshot_id,
                "Policy content/reference differs from verified snapshot")
        require(len(self.metadata_sha256) == 64 and
                all(c in "0123456789abcdef" for c in self.metadata_sha256), "Metadata digest required")


@dataclass(frozen=True)
class PolicyCitation(Record):
    section_id: str
    start: int
    end: int
    quote: str

    def validate(self):
        require(bool(self.section_id) and 0 <= self.start < self.end and bool(self.quote),
                "Policy citation needs a nonempty exact source span")


@dataclass(frozen=True)
class PolicyClause(Record):
    clause_id: str
    meaning: str
    role: str
    citations: tuple[PolicyCitation, ...]

    def validate(self):
        require(self.meaning.strip() and self.citations, "Policy clause needs original source grounding")
        require(self.role in ("CONDITION", "BOUNDARY_EXCLUDES", "BOUNDARY_REQUIRES",
                             "BOUNDARY_RESTRICTS", "BOUNDARY_UNRESOLVED"), "Unknown policy clause role")


@dataclass(frozen=True)
class PolicyAction(Record):
    catalog_action_id: str
    citations: tuple[PolicyCitation, ...]
    source_meaning: str
    when_expression_json: str


@dataclass(frozen=True)
class PolicyRule(Record):
    rule_id: str
    snapshot_id: str
    section_ids: tuple[str, ...]
    original_meaning: str
    citations: tuple[PolicyCitation, ...]
    conditions: tuple[PolicyClause, ...]
    boundaries: tuple[PolicyClause, ...]
    applicability_expression_json: str
    actions: tuple[PolicyAction, ...]
    action_specificity: str
    action_composition: str
    uncertainty: tuple[str, ...]
    underspecification: tuple[str, ...]
    conversion_version: str

    def validate(self):
        require(self.rule_id.startswith("npov_rule_") and self.original_meaning.strip() and self.citations,
                "Policy rule requires independent source identity/meaning/citations")
        require(self.action_specificity in ("EXPLICIT_OPERATION", "ALLOWABLE_OPERATIONS", "NOT_SPECIFIED"),
                "Policy action specificity must be explicit")
        require(self.action_composition in ("SINGLE", "ALL_SUPPORTED", "ALLOWABLE_ALTERNATIVES", "EXCLUSIVE_ALTERNATIVES", "UNRESOLVED"),
                "Policy ambiguity cannot become a forced action")
        if self.action_specificity == "NOT_SPECIFIED":
            require(not self.actions and self.underspecification,
                    "Unspecified policy operation must stay actionless with a reason")
        if self.action_composition == "UNRESOLVED":
            require(not self.actions and self.underspecification, "Unresolved operation cannot become a joint edit")
        if self.action_specificity != "NOT_SPECIFIED":
            require(bool(self.actions), "Specified operation needs source-backed action references")
        if self.action_composition == "SINGLE":
            require(len(self.actions) == 1, "A single policy operation cannot hide alternatives")
        require(all(c.role == "CONDITION" for c in self.conditions) and
                all(c.role.startswith("BOUNDARY_") for c in self.boundaries), "Policy clause role mismatch")


@dataclass(frozen=True)
class PolicySectionAccounting(Record):
    section_id: str
    disposition: str
    rule_ids: tuple[str, ...]
    explanation: str

    def validate(self):
        require(self.disposition in ("RULES", "GUIDANCE_ONLY", "UNRESOLVED") and self.explanation.strip(),
                "Policy sections cannot be silently discarded")
        require((self.disposition == "RULES") == bool(self.rule_ids),
                "Section rule disposition disagrees with its rule references")


@dataclass(frozen=True)
class PolicyRuleRegistry(Record):
    snapshot: PolicySnapshotMetadata
    snapshot_reference: SourceReference
    metadata_sha256: str
    rules: tuple[PolicyRule, ...]
    section_accounting: tuple[PolicySectionAccounting, ...]
    conversion_version: str
    registry_hash: str

    def validate(self):
        require(self.snapshot_reference.source_id == self.snapshot.snapshot_id and
                self.snapshot_reference.sha256 == self.snapshot.content_sha256,
                "Policy registry source reference changed")
        require(len(self.metadata_sha256) == 64 and
                all(c in "0123456789abcdef" for c in self.metadata_sha256), "Policy metadata hash required")
        require(all(r.snapshot_id == self.snapshot.snapshot_id and
                    r.conversion_version == self.conversion_version for r in self.rules),
                "Policy registry mixes source versions")


@dataclass(frozen=True)
class BaselineComparability(Record):
    feature_input_artifact_hash: str
    case_ids: tuple[str, ...]
    action_catalog_hash: str
    theory_registry_hash: str
    policy_registry_hash: str
    policy_snapshot_id: str
    input_contract: str
    extraction_contract: str
    require_shared_action_ids: bool


def policy_registry_hash(registry):
    return stable_id("npov_rule_registry", {k: v for k, v in registry.to_dict().items() if k != "registry_hash"})
