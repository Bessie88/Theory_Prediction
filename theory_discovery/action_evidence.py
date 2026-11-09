"""Exact-source Action evidence validation, shared with human corrections."""
from .workflow_contracts import ObservedOperation, ActionInstance
from .action_comparison import validate_protocol, permits_novel_actions, novel_action_id, NOVEL_ACTION_VERSION
import json
from .text_diff import validate_diff
from .schemas import require


def _touch(span, target):
    if target.start == target.end:
        return span.start == span.end == target.start
    return span.start < target.end and span.end > target.start


def _covered(target, spans):
    if target.start == target.end:
        return any(_touch(s, target) for s in spans)
    cursor = target.start
    for s in sorted(spans, key=lambda v: (v.start, v.end)):
        if s.end <= cursor or s.start >= target.end:
            continue
        if s.start > cursor:
            return False
        cursor = max(cursor, s.end)
    return cursor >= target.end


def validate_annotation_response(response, case, protocol, novel_definitions=()):
    validate_protocol(protocol)
    require(case.availability == "AVAILABLE", "Cannot label an unavailable actual edit")
    validate_diff(case.actual_diff, case.before_text, case.after_text)
    hunks = {h.hunk_id: h for h in case.actual_diff.changed_hunks}
    shared = {a.action_id: a for a in protocol.actions}
    for definition in novel_definitions:
        require(permits_novel_actions(protocol) and definition.version == NOVEL_ACTION_VERSION and
                definition.action_id == novel_action_id(definition) and definition.action_id not in shared,
                "Novel Action is not independently defined or conflicts with an existing label")
        shared[definition.action_id] = definition
    require(len(set(response.unassigned_diff_hunk_ids)) == len(response.unassigned_diff_hunk_ids) and
            set(response.unassigned_diff_hunk_ids) <= set(hunks), "Unknown/duplicate unassigned diff")
    if not hunks:
        require(response.status == "NO_CHANGE" and not response.operations and
                not response.unassigned_diff_hunk_ids, "Unchanged text is not a semantic editing Action")
        return ()
    require(response.status != "NO_CHANGE", "Changed text cannot be labeled unchanged")
    operations = []
    coverage = {key: {"before": [], "after": []} for key in hunks}
    for item in response.operations:
        require(item.action_id in shared and item.explanation.strip() and
                item.confidence in ("HIGH", "MEDIUM", "LOW", "UNCERTAIN"), "Unknown label or missing operation explanation")
        require(item.diff_hunk_ids and len(set(item.diff_hunk_ids)) == len(item.diff_hunk_ids) and
                set(item.diff_hunk_ids) <= set(hunks), "Operation needs exact changed CHARACTER hunk IDs")
        require(item.before_evidence and item.after_evidence, "Both sides, including zero-width anchors, are required")
        for side, spans, text in (("before", item.before_evidence, case.before_text),
                                  ("after", item.after_evidence, case.after_text)):
            for span in spans:
                require(span.side == side, "Evidence side mismatch")
                span.check_text(text)
                require(any(_touch(span, getattr(hunks[key], side)) for key in item.diff_hunk_ids),
                        "Evidence must touch a cited textual change/anchor")
            for key in item.diff_hunk_ids:
                require(any(_touch(s, getattr(hunks[key], side)) for s in spans),
                        "Cited change lacks supporting evidence on one side")
                coverage[key][side].extend(spans)
        evidence = item.before_evidence + item.after_evidence
        is_novel = item.action_id.startswith("novel_action_")
        if is_novel and response.status != "UNCERTAIN":
            require(item.confidence in ("HIGH", "MEDIUM") and response.confidence in ("HIGH", "MEDIUM"),
                    "Uncertain new operations must remain UNCERTAIN")
        description = (json.dumps(shared[item.action_id].to_dict(), sort_keys=True) if is_novel else
                       item.explanation if item.action_id == "OTHER" else shared[item.action_id].description)
        action = ActionInstance(item.action_id, None, description, evidence)
        operations.append(ObservedOperation(action, item.diff_hunk_ids, item.before_evidence,
            item.after_evidence, item.explanation, item.confidence, item.uncertainty))
    incomplete = {key for key, hunk in hunks.items()
                  if not all(_covered(getattr(hunk, side), coverage[key][side]) for side in ("before", "after"))}
    require(incomplete <= set(response.unassigned_diff_hunk_ids), "Unexplained changed characters must remain explicitly unassigned")
    uncertain = (bool(response.unassigned_diff_hunk_ids) or response.confidence == "UNCERTAIN" or
                 any(o.confidence == "UNCERTAIN" for o in response.operations))
    if response.status == "UNCERTAIN":
        require(response.uncertainty, "Uncertain annotation needs a reason")
    else:
        require(not uncertain, "Uncertain/unassigned edits cannot be presented as fully annotated")
        require(bool(operations), "Changed text needs observed operations or explicit uncertainty")
        require((response.status == "OTHER") == any(o.action.action_id == "OTHER" for o in operations),
                "OTHER status/operation mismatch")
    return tuple(operations)
