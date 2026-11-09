"""Pure two-period split over supplied records, never invoked at import."""
from dataclasses import dataclass
from .workflow_contracts import EditorKey, Record, utc_time, nonempty
from .workflow_config import SplitSettings
from .revision_history import LoadedRevision, revision_groups, unique_revision_metadata
from .schemas import require


@dataclass(frozen=True)
class RevisionSplit(Record):
    editor: EditorKey
    cutoff: str
    early: tuple[LoadedRevision, ...]
    later: tuple[LoadedRevision, ...]
    requested_ratio: float | None


def split_editor_revisions(records, editor, settings: SplitSettings, ordering="SORT_CHRONOLOGICALLY"):
    selected = tuple(r for r in records if r.revision.editor == editor)
    require(len(selected) >= 2, "Editor needs at least two revision records")
    for item in selected:
        item.require_usable()
    groups = revision_groups(records)
    key = lambda r: (utc_time(r.revision.timestamp), r.revision.revision_id, r.sentence_pair_id or "")
    ordered = tuple(sorted(selected, key=key))
    require(ordering in ("SORT_CHRONOLOGICALLY", "REQUIRE_CHRONOLOGICAL"), "Unknown ordering policy")
    if ordering == "REQUIRE_CHRONOLOGICAL":
        require(selected == ordered, "Source revision ordering is invalid")
    # A parent need not be an edit by the selected editor. If supplied in the
    # local snapshot, validate it against the actual parent record.
    parents = {key: rows[0] for key, rows in groups.items()}
    for item in ordered:
        r = item.revision
        parent = parents.get((r.editor.wiki_id, r.parent_revision_id))
        if parent is not None:
            require(parent.revision.page_id == r.page_id, "Parent belongs to another page")
            require(parent.revision.timestamp is not None and
                    utc_time(parent.revision.timestamp) == utc_time(item.parent_timestamp),
                    "Parent timestamp mismatch")
            if item.text_unit == parent.text_unit == "REVISION":
                require(parent.after_text is not None and parent.after_text == item.before_text,
                        "Before is not the supplied parent's content")
    if settings.cutoff_utc is not None:
        cutoff = utc_time(settings.cutoff_utc)
    else:
        ratio = settings.early_ratio
        require(type(ratio) is float and 0 < ratio < 1, "Explicit early_ratio required")
        revisions = unique_revision_metadata(ordered)
        candidates = [i for i in range(1, len(revisions))
                      if utc_time(revisions[i-1].timestamp) < utc_time(revisions[i].timestamp)]
        require(bool(candidates), "Timestamp ties cannot be split between periods")
        # Closest achievable revision ratio; ties choose the smaller EARLY set.
        index = min(candidates, key=lambda i: (abs(i / len(revisions) - ratio), i))
        cutoff = utc_time(revisions[index].timestamp)
    early = tuple(r for r in ordered if utc_time(r.revision.timestamp) < cutoff)
    later = tuple(r for r in ordered if utc_time(r.revision.timestamp) >= cutoff)
    require(bool(early) and bool(later), "Cutoff must create two nonempty periods")
    require(max(utc_time(r.revision.timestamp) for r in early) <
            min(utc_time(r.revision.timestamp) for r in later), "Invalid temporal separation")
    return RevisionSplit(editor, cutoff.isoformat(), early, later, settings.early_ratio)


def early_manifest_projection(manifest):
    """Metadata projection for native discovery provenance; contains no LATER IDs."""
    from .workflow_contracts import Period
    selected = set(manifest.early_revision_ids)
    return {
        "editor": manifest.editor.to_dict(), "cutoff": manifest.cutoff,
        "early_revision_ids": list(manifest.early_revision_ids),
        "cases": [c.to_dict() for c in manifest.cases if c.period is Period.EARLY],
        "revision_timestamps": [t.to_dict() for t in manifest.revision_timestamps if t.key in selected],
        "split_configuration": [item.to_dict() for item in manifest.split_configuration]}
