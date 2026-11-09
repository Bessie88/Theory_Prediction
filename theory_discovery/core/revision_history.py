"""Future local revision-history adapters. No retrieval or I/O at import time.

Supported exports are explicitly normalized JSONL, JSON and headered TSV, NOT WNC.
No actual source format has yet been supplied/connected.
"""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Protocol
import csv
import json
from .workflow_contracts import (
    Record, EditorKey, EditorRevision, SourceReference, PreEditField,
    nonempty, utc_time,
)
from .workflow_config import SourceSettings
from .schemas import require, stable_id

SENTENCE_PAIR_UNITS = ("EDITED_SENTENCE_PAIR", "AFTER_ALIGNED_SENTENCE_PAIR")


@dataclass(frozen=True)
class LoadedRevision(Record):
    revision: EditorRevision
    before_text: str | None
    after_text: str | None
    parent_timestamp: str | None
    origin: str
    human_verification_note: str
    text_unit: str
    pre_context: tuple[PreEditField, ...]
    sentence_pair_id: str | None = None

    def validate(self):
        nonempty(self.origin)
        require(self.text_unit in ("REVISION", *SENTENCE_PAIR_UNITS),
                "Declare whether the input is a full revision or an outcome-selected pair")
        if self.parent_timestamp is not None:
            utc_time(self.parent_timestamp)

    def require_usable(self):
        r = self.revision
        require(r.editor_verified_human and bool(self.human_verification_note.strip()),
                "Verified human editor identity evidence required")
        require(not r.missing_fields and r.parent_revision_id != r.revision_id,
                "Revision has missing fields or is its own parent")
        nonempty(r.parent_revision_id, r.timestamp, r.page_id, self.parent_timestamp)
        require(type(self.before_text) is str and type(self.after_text) is str,
                "Both Before and After must be supplied")
        require(utc_time(self.parent_timestamp) < utc_time(r.timestamp),
                "Parent timestamp must precede target revision")
        if self.text_unit in SENTENCE_PAIR_UNITS:
            nonempty(self.sentence_pair_id)
            require(self.before_text != self.after_text,
                    "Edited-sentence samples must contain an actual change")


def revision_groups(records):
    """Validate sample uniqueness and shared revision metadata without merging pairs."""
    groups, samples = {}, set()
    for row in records:
        r = row.revision
        key = (r.editor.wiki_id, r.revision_id)
        identity = (key, row.sentence_pair_id if row.text_unit in SENTENCE_PAIR_UNITS else None)
        require(identity not in samples, "Duplicate revision/sample identity")
        samples.add(identity)
        previous = groups.setdefault(key, [])
        if previous:
            first = previous[0]
            require(row.text_unit in SENTENCE_PAIR_UNITS and first.text_unit in SENTENCE_PAIR_UNITS,
                    "A full revision cannot be mixed with duplicate revision/sample rows")
            require((r.editor, r.parent_revision_id, r.timestamp, r.page_id, r.editor_verified_human,
                     row.parent_timestamp) ==
                    (first.revision.editor, first.revision.parent_revision_id, first.revision.timestamp,
                     first.revision.page_id, first.revision.editor_verified_human, first.parent_timestamp),
                    "Samples from one revision disagree on editor, parent, page or time")
        previous.append(row)
    return groups


def unique_revision_metadata(records):
    """One revision record per revision, retaining the provenance of every sample."""
    return tuple(replace(rows[0].revision,
                         provenance=tuple(dict.fromkeys(p for row in rows for p in row.revision.provenance)))
                 for rows in revision_groups(records).values())


@dataclass(frozen=True)
class RetrievalRequest(Record):
    editor: EditorKey
    earliest_timestamp: str
    latest_timestamp: str
    include_parent_content: bool
    expected_export_schema: str


class RevisionRetriever(Protocol):
    def retrieve(self, request: RetrievalRequest) -> SourceSettings:
        """Future separately authorized retrieval; must return a hashed local snapshot."""
        ...


class UnconfiguredRetriever:
    def retrieve(self, request: RetrievalRequest) -> SourceSettings:
        raise NotImplementedError("No retrieval backend configured; no network fallback")


def normalize_timestamp(value):
    if value is None or value == "":
        return None
    nonempty(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Timezone cannot be guessed")
    return parsed.astimezone(timezone.utc).isoformat()


def _rows(text, container):
    if container == "jsonl":
        for number, line in enumerate(text.splitlines(), 1):
            require(bool(line.strip()), "Blank JSONL record")
            yield "/lines/" + str(number), json.loads(line)
    elif container == "json":
        root = json.loads(text)
        prefix = ""
        if isinstance(root, dict):
            require(set(root) == {"schema", "records"} and root["schema"] == "wiki-editor-revision-v1",
                    "Unknown export envelope")
            root = root["records"]
            prefix = "/records"
        require(isinstance(root, list), "JSON export must be a record array")
        for index, row in enumerate(root):
            yield prefix + "/" + str(index), row
    elif container == "tsv":
        reader = csv.DictReader(StringIO(text), delimiter="\t")
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "TSV requires unique named columns; headerless/WNC TSV rejected")
        for number, row in enumerate(reader, 2):
            require(None not in row, "TSV column count mismatch")
            # The explicitly structured context cell uses JSON; scalar strings stay strings.
            for key in ("pre_context",):
                if row.get(key):
                    row[key] = json.loads(row[key])
            yield "/rows/" + str(number), row
    else:
        raise ValueError("Unsupported local container")


def decode_revision(raw, source, pointer, field_map=()):
    require(isinstance(raw, dict), "Revision record must be an object")
    mapping = {item.key: item.value for item in field_map}
    require(all(type(v) is str and v for v in mapping.values()), "Field mappings name source columns")
    def value(key, default=None):
        return raw.get(mapping.get(key, key), default)
    require(value("source_kind") == "wikipedia_revision_history",
            "Explicit Wikipedia revision-history source_kind required; no WNC")
    origin = value("origin")
    nonempty(origin)
    require(not any(term in origin.casefold() for term in ("wnc", "biased.full", "neutralizing-bias")),
            "Historical WNC provenance is not permitted")
    editor = EditorKey(value("wiki_id"), value("editor_id"))
    rid = value("revision_id")
    nonempty(rid)
    verified = value("editor_verified_human")
    if type(verified) is str:
        require(verified in ("true", "false"), "TSV verification must be true/false")
        verified = verified == "true"
    require(type(verified) is bool, "Explicit editor_verified_human boolean required")
    timestamp = normalize_timestamp(value("timestamp"))
    parent_timestamp = normalize_timestamp(value("parent_timestamp"))
    parent = value("parent_revision_id") or None
    page = value("page_id") or None
    before, after = value("before_text"), value("after_text")
    require(before is None or type(before) is str, "Before must be text or null")
    require(after is None or type(after) is str, "After must be text or null")
    missing = tuple(name for name, item in (
        ("parent_revision_id", parent), ("timestamp", timestamp), ("page_id", page),
        ("before_reference", before), ("after_reference", after),
        ("parent_timestamp", parent_timestamp)) if item is None)
    def reference(field):
        return SourceReference(source.source_id, source.location, source.sha256,
                               pointer + "/" + mapping.get(field, field).replace("~", "~0").replace("/", "~1"))
    revision = EditorRevision(
        editor, rid, parent, timestamp, page,
        (SourceReference(source.source_id, source.location, source.sha256, pointer),),
        reference("before_text") if before is not None else None,
        reference("after_text") if after is not None else None, verified, missing)
    context = value("pre_context", [])
    require(isinstance(context, list), "pre_context must be a typed field array")
    if context:
        require("pre_context" not in mapping,
                "Mapped structured context needs an explicit future format adapter")
    return LoadedRevision(
        revision, before, after, parent_timestamp, value("origin"),
        value("human_verification_note", ""), value("text_unit"),
        tuple(PreEditField.from_dict(field) for field in context), value("sentence_pair_id"))


def load_local_revisions(settings: SourceSettings) -> tuple[LoadedRevision, ...]:
    """Explicit future API. Unconfigured source fails BEFORE opening a path."""
    nonempty(settings.location, settings.content_sha256)
    require(settings.format == "wikipedia_revision_history", "WNC adapters are forbidden")
    path = Path(settings.location).resolve()
    require(path.name != "biased.full", "WNC source is forbidden")
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    require(digest == settings.content_sha256, "Source snapshot hash mismatch")
    source = SourceReference(stable_id("revision_source", digest), str(path), digest, "")
    records = tuple(decode_revision(row, source, pointer, settings.field_map)
                    for pointer, row in _rows(data.decode("utf-8"), settings.container_format))
    require(bool(records), "Empty revision snapshot")
    revision_groups(records)
    return records
