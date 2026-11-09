"""Validate externally completed submissions; never synthesize human decisions."""
from hashlib import sha256
from pathlib import Path
import json
from .review_contracts import (
    HumanResponseBatch, FeatureHumanReview, FeatureHumanReviews, ReviewImportAudit, ActionReviewLabel,
)
from theory_discovery.core.workflow_contracts import HumanReviewRecord, AnnotationStatus
from .review_export import packet_hash
from .action_comparison import validate_protocol
from .action_annotator import parse_annotation
from theory_discovery.features.feature_evidence import check_value
from theory_discovery.core.schemas import require, stable_id


def read_review_responses(location, expected_sha256):
    """Explicit future local ingress; not called by imports or a CLI."""
    path = Path(location)
    require(path.stat().st_size <= 32 * 1024 * 1024, "Review submission exceeds local ingress budget")
    raw = path.read_bytes()
    require(len(raw) <= 32 * 1024 * 1024 and sha256(raw).hexdigest() == expected_sha256,
            "Submitted human response bytes/hash changed")
    return HumanResponseBatch.from_dict(json.loads(raw.decode("utf-8")))


def _responses(packet, submissions, settings, kind):
    require(packet.kind == kind and packet.packet_id == packet_hash(packet), "Review packet changed/wrong kind")
    validate_protocol(packet.protocol)
    if submissions is None:
        return {}
    require(submissions.packet_id == packet.packet_id, "Human responses belong to another packet")
    tasks = {item.task_id: item for item in packet.items}
    require(all(r.task_id in tasks and r.mode == tasks[r.task_id].mode for r in submissions.responses),
            "Unknown task or mode mismatch")
    require(all(r.reviewer_id in settings.permitted_reviewers for r in submissions.responses),
            "Completed submissions require an independently identified permitted reviewer")
    for response in submissions.responses:
        require(response.notes.strip() or response.decision != "UNCERTAIN", "Uncertain human decision needs notes")
        if response.decision in ("ACCEPT", "UNCERTAIN"):
            require(response.action_annotation is None and response.feature_annotation is None,
                    "Acceptance/uncertain decision cannot silently carry a correction")
    return {r.task_id: r for r in submissions.responses}


def import_action_reviews(packet, cases, annotations, protocol, submissions, settings):
    require(packet.protocol == protocol, "Action review protocol mismatch")
    responses = _responses(packet, submissions, settings, "ACTION")
    source = {a.annotation_id: a for a in annotations.annotations}
    require(all(item.subject_id in source for item in packet.items), "Review subject missing raw annotation")
    items = {item.subject_id: item for item in packet.items}
    require(len(items) == len(packet.items), "Duplicate review subject")
    actual = {c.identity.case_id: c for c in cases.cases}
    records = []
    for model in annotations.annotations:
        item = items.get(model.annotation_id)
        if item is None:
            records.append(HumanReviewRecord(model.identity, model, None, None, "NOT_SELECTED", None, None))
            continue
        case = actual[model.identity.case_id]
        require(item.identity == model.identity == case.identity and case.input_hash == model.input_hash and
                model.protocol_hash == protocol.protocol_hash and
                item.task_id == stable_id("action_review_task", [model.annotation_id, item.mode, protocol.protocol_hash]),
                "Raw annotation/case/task changed")
        expected_visible = (ActionReviewLabel(model.annotation_status.value, model.operations, model.explanation,
            model.confidence, model.uncertainty, model.unassigned_diff_hunk_ids)
            if item.mode == "LABEL_VERIFICATION" else None)
        require(item.before_text == case.before_text and item.after_text == case.after_text and
                item.actual_diff == case.actual_diff and item.context == case.context and
                item.availability == case.availability and item.limitations == case.limitations and
                item.action_label == expected_visible,
                "Reviewer packet did not show the registered content/model label")
        response = responses.get(item.task_id)
        if response is None:
            records.append(HumanReviewRecord(model.identity, model, None, None, "PENDING", None, None,
                                             item.mode, packet.packet_id))
            continue
        corrected, operations = None, ()
        status = "UNRESOLVED"
        if response.decision == "ACCEPT":
            require(item.action_label is not None and response.mode == "LABEL_VERIFICATION",
                    "Cannot accept an unseen model label")
            operations = model.operations
            if model.annotation_status in (AnnotationStatus.ANNOTATED, AnnotationStatus.OTHER, AnnotationStatus.NO_CHANGE):
                # Revalidate raw label evidence, not just its rendered appearance.
                from .annotation_contracts import AnnotationResponse, AnnotationOperationResponse
                converted = AnnotationResponse(model.annotation_status.value, tuple(
                    AnnotationOperationResponse(o.action.action_id, o.diff_hunk_ids, o.before_evidence,
                        o.after_evidence, o.explanation, o.confidence, o.uncertainty) for o in model.operations),
                    model.unassigned_diff_hunk_ids, model.explanation, model.confidence, model.uncertainty)
                from .action_comparison import novel_action_definition
                definitions = tuple(d for o in model.operations if (d := novel_action_definition(o.action)) is not None)
                parse_annotation(converted.to_dict(), case, protocol, definitions)
                status = "CONFIRMED"
        elif response.decision in ("ASSIGN", "CORRECT"):
            require(response.action_annotation is not None and response.feature_annotation is None,
                    "Independent/corrected Action response required")
            parsed, operations = parse_annotation(response.action_annotation.to_dict(), case, protocol)
            if parsed.status != "UNCERTAIN":
                new_actions = tuple(o.action for o in operations)
                status = ("CONFIRMED" if new_actions == model.observed_actions and
                          model.annotation_status.value == parsed.status else "CORRECTED")
                if status == "CORRECTED":
                    corrected = new_actions
        key = stable_id("human_action_response", response.to_dict())
        records.append(HumanReviewRecord(model.identity, model, response.decision, corrected, status,
            response.reviewer_id, response.notes or
            (response.action_annotation.explanation if response.action_annotation else "Reviewer decision."),
            item.mode, packet.packet_id, key, response.completed_at_utc, response.notes, operations))
    return tuple(records), ReviewImportAudit(submissions)


def validate_feature_label(label, source):
    definition, case = source.definition, source.case
    require(label.status in ("KNOWN", "UNKNOWN", "ERROR") and label.explanation.strip(),
            "Feature human label status/explanation required")
    require(set(label.context_evidence_names) <= {v.name for v in case.pre_context},
            "Unknown feature context evidence")
    for span in label.evidence:
        require(span.side == "before", "Human Feature evidence cannot expose After")
        span.check_text(case.before_text)
    require(len({(s.start, s.end) for s in label.evidence}) == len(label.evidence),
            "Duplicate human feature evidence")
    if label.status != "KNOWN":
        require(label.value is None and label.uncertainty, "Uncertain feature must remain null with a reason")
        return
    check_value(definition, label.value, case)
    require(label.value is not None, "Known human feature value required")
    if label.evidence_scope == "SPANS":
        require(bool(label.evidence), "Human feature text evidence missing")
    elif label.evidence_scope == "METADATA":
        require(bool(label.context_evidence_names), "Human feature context evidence missing")
    elif label.evidence_scope == "FULL_BEFORE":
        require("FULL_BEFORE" in definition.evidence_requirements, "Full Before assessment not permitted")
    elif label.evidence_scope == "CANDIDATES":
        require(definition.value_type == "structured", "Candidate feature must be structured")
    else:
        raise ValueError("Unknown human evidence scope")
    require("SPANS" not in definition.evidence_requirements or bool(label.evidence), "Exact spans required")
    require("METADATA" not in definition.evidence_requirements or bool(label.context_evidence_names), "Context required")


def import_feature_reviews(packet, source, submissions, settings):
    responses = _responses(packet, submissions, settings, "FEATURE")
    original = {r.subject_id: r for r in source.records}
    require(all(i.subject_id in original for i in packet.items), "Feature review source missing")
    items = {i.subject_id: i for i in packet.items}
    require(len(items) == len(packet.items), "Duplicate feature review subject")
    records = []
    for row in source.records:
        item = items.get(row.subject_id)
        response = responses.get(item.task_id) if item else None
        status, corrected, response_id, reviewer, completed, notes = (
            "PENDING" if item else "NOT_SELECTED", None, "", None, None, "")
        if item:
            require(item.identity == row.identity and item.feature_definition == row.definition and
                    item.before_text == row.case.before_text and
                    item.task_id == stable_id("feature_review_task", [row.subject_id, row.observation_hash,
                                                                     item.mode, packet.protocol.protocol_hash]),
                    "Feature source/task changed")
            require(item.context == row.case.pre_context and
                    item.feature_label == (row.model_label if item.mode == "LABEL_VERIFICATION" else None),
                    "Reviewer packet feature label/context differs from raw source")
        if response:
            response_id, reviewer, completed, notes = (
                stable_id("human_feature_response", response.to_dict()), response.reviewer_id,
                response.completed_at_utc, response.notes)
            status = "UNRESOLVED"
            if response.decision == "ACCEPT":
                require(item.feature_label is not None, "Cannot accept an unseen feature label")
                validate_feature_label(row.model_label, row)
                status = "CONFIRMED" if row.model_label.status == "KNOWN" else "UNRESOLVED"
            elif response.decision in ("ASSIGN", "CORRECT"):
                require(response.feature_annotation is not None and response.action_annotation is None,
                        "Independent/corrected Feature response required")
                validate_feature_label(response.feature_annotation, row)
                corrected = response.feature_annotation
                status = ("UNRESOLVED" if corrected.status != "KNOWN" else
                          "CONFIRMED" if corrected == row.model_label else "CORRECTED")
        records.append(FeatureHumanReview(row.identity, row.subject_id, row.model_label, corrected, status,
            item.mode if item else "UNASSIGNED", reviewer, completed, response_id,
            packet.packet_id if item else "", notes))
    return FeatureHumanReviews(tuple(records)), ReviewImportAudit(submissions)
