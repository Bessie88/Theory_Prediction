"""Explicit local snapshot ingestion only. No network/retrieval backend exists."""
from hashlib import sha256
from pathlib import Path
import json
from .policy_contracts import PolicySnapshotMetadata, PolicySnapshot, PolicySourceUnavailable
from .workflow_contracts import SourceReference
from .schemas import require


def ingest_policy(settings):
    # Gate before opening any path. There is no fallback to remembered policy
    # guidance, the legacy BASELINE_TASK, or a live Wikipedia request.
    if not all((settings.snapshot_location, settings.snapshot_sha256,
                settings.metadata_location, settings.metadata_sha256, settings.version)):
        raise PolicySourceUnavailable("POLICY_SOURCE_UNAVAILABLE: verified local snapshot and metadata required")
    content_path, metadata_path = Path(settings.snapshot_location).resolve(), Path(settings.metadata_location).resolve()
    if not content_path.is_file() or not metadata_path.is_file():
        raise PolicySourceUnavailable("POLICY_SOURCE_UNAVAILABLE: local snapshot files missing")
    raw, meta_bytes = content_path.read_bytes(), metadata_path.read_bytes()
    require(sha256(raw).hexdigest() == settings.snapshot_sha256 and
            sha256(meta_bytes).hexdigest() == settings.metadata_sha256, "Policy snapshot/metadata changed")
    metadata = PolicySnapshotMetadata.from_dict(json.loads(meta_bytes.decode("utf-8")))
    require(metadata.content_sha256 == settings.snapshot_sha256 and
            metadata.revision_id == settings.version and settings.source == metadata.canonical_url,
            "Policy settings disagree with the supplied official snapshot identity")
    content = raw.decode("utf-8")
    require(content.strip() and all(section.end <= len(content) for section in metadata.sections),
            "Policy section map exceeds snapshot content")
    cursor = 0
    for section in sorted(metadata.sections, key=lambda s: (s.start, s.end)):
        require(section.start <= cursor, "Policy section map silently omits content")
        cursor = max(cursor, section.end)
    require(cursor == len(content), "Policy section map must cover the full supplied snapshot")
    # Section maps may nest (section/subsection); all offsets refer to exact bytes'
    # UTF-8 decoded Unicode text, never a fresh HTML-to-text conversion.
    reference = SourceReference(metadata.snapshot_id, str(content_path), metadata.content_sha256, "")
    return PolicySnapshot(metadata, content, reference, settings.metadata_sha256)


def validate_citation(snapshot, citation):
    sections = {s.section_id: s for s in snapshot.metadata.sections}
    require(citation.section_id in sections, "Unknown official policy section")
    section = sections[citation.section_id]
    require(section.start <= citation.start < citation.end <= section.end and
            snapshot.content[citation.start:citation.end] == citation.quote,
            "Policy citation quote/offset does not match the verified local snapshot")
