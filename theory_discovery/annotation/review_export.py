"""Future packet construction, strictly separated from sampling and prediction workers."""
from .review_contracts import ReviewItem, ReviewPacket, ReviewExport, ActionReviewLabel
from .action_comparison import validate_protocol
from theory_discovery.core.schemas import require, stable_id
from .review_html import render_review_html, response_template


def packet_hash(packet):
    return stable_id("human_review_packet", {k: v for k, v in packet.to_dict().items() if k != "packet_id"})


def _finish(kind, protocol, items):
    from dataclasses import replace
    packet = ReviewPacket("", kind, protocol, tuple(items))
    packet = replace(packet, packet_id=packet_hash(packet))
    return ReviewExport(packet, render_review_html(packet), response_template(packet))


def export_action_reviews(cases, annotations, assignments, protocol):
    validate_protocol(protocol)
    actual = {c.identity.case_id: c for c in cases.cases}
    original = {a.annotation_id: a for a in annotations.annotations}
    items = []
    for selected in assignments.assignments:
        require(selected.subject_id in original, "Review assignment references unknown annotation")
        model = original[selected.subject_id]
        case = actual[selected.identity.case_id]
        require(case.identity == model.identity == selected.identity and
                model.input_hash == case.input_hash and model.protocol_hash == protocol.protocol_hash,
                "Review source identity/protocol mismatch")
        label = (ActionReviewLabel(model.annotation_status.value, model.operations, model.explanation,
                 model.confidence, model.uncertainty, model.unassigned_diff_hunk_ids)
                 if selected.mode == "LABEL_VERIFICATION" else None)
        task = stable_id("action_review_task", [model.annotation_id, selected.mode, protocol.protocol_hash])
        items.append(ReviewItem(task, model.annotation_id, case.identity, selected.mode,
            case.before_text, case.after_text, case.actual_diff, case.context, case.availability, case.limitations, None, label, None))
    return _finish("ACTION", protocol, items)


def export_feature_reviews(source, assignments, protocol):
    validate_protocol(protocol)
    original = {r.subject_id: r for r in source.records}
    items = []
    for selected in assignments.assignments:
        record = original[selected.subject_id]
        require(record.identity == selected.identity == record.case.identity, "Feature review identity mismatch")
        task = stable_id("feature_review_task", [record.subject_id, record.observation_hash,
                                                selected.mode, protocol.protocol_hash])
        items.append(ReviewItem(task, record.subject_id, record.identity, selected.mode,
            record.case.before_text, None, None, record.case.pre_context, "AVAILABLE", (), record.definition, None,
            record.model_label if selected.mode == "LABEL_VERIFICATION" else None))
    return _finish("FEATURE", protocol, items)


def materialize_registered_export(exported, directory, registered_artifact_id, execution_authorized=False):
    """Optional future local delivery of an ALREADY registered packet, not a review request.
    The original registered JSON remains authoritative; only sanitized packet/UI is written.
    """
    require(execution_authorized and bool(registered_artifact_id), "Review delivery disabled/unregistered")
    require(exported.packet.packet_id == packet_hash(exported.packet) and
            exported.html_document == render_review_html(exported.packet) and
            exported.response_template_json == response_template(exported.packet), "Review export changed")
    from pathlib import Path
    from hashlib import sha256
    from theory_discovery.core.storage import write_json
    import json
    root = Path(directory) / exported.packet.packet_id
    root.mkdir(parents=True, exist_ok=True)
    files = {"review.html": exported.html_document,
             "packet.json": json.dumps(exported.packet.to_dict(), ensure_ascii=False, indent=2),
             "response-template.json": exported.response_template_json}
    hashes = {}
    for name, text in files.items():
        raw = text.encode("utf-8")
        path = root / name
        if path.exists():
            require(path.read_bytes() == raw, "Do not overwrite a different review delivery")
        else:
            path.write_bytes(raw)
        hashes[name] = sha256(raw).hexdigest()
    manifest = {"registered_artifact_id": registered_artifact_id, "packet_id": exported.packet.packet_id,
                "files": hashes, "status": "EXPORTED_NOT_REVIEWED"}
    write_json(root / "delivery.json", manifest)
    return manifest
