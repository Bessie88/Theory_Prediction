"""Configurable, independently prompted annotation; provider constructed only on demand."""
from dataclasses import fields, replace
from typing import Protocol
from pathlib import Path
import json
from .annotation_contracts import (
    AnnotationResponse, AnnotationOperationResponse, AnnotationAttempt, AnnotationAudit, AnnotatorIndependence,
)
from .workflow_contracts import ObservedAction, AnnotationStatus, EvidenceSpan, MetadataItem, SCHEMA_VERSION, ActionDefinition, contract_schema
from .action_evidence import validate_annotation_response
from .action_comparison import (validate_protocol, permits_novel_actions, action_meaning_key,
                               novel_action_id, novel_action_definition, NOVEL_ACTION_VERSION)
from .annotation_data import annotation_case_hash
from .text_diff import validate_diff, VERSION as DIFF_VERSION
from .annotation_prompts import SYSTEM, PROMPT_VERSION
from .schemas import require, stable_id
from .storage import read_json, write_json

COMPLETE_ACTION_STATUSES = (AnnotationStatus.ANNOTATED, AnnotationStatus.NO_CHANGE)
CONFIDENCE_VALUES = ("HIGH", "MEDIUM", "LOW", "UNCERTAIN")


class AnnotationBackend(Protocol):
    def complete(self, system: str, payload: dict) -> dict:
        ...


def independence(settings, extractor_families):
    family = settings.model.family
    status = ("UNVERIFIED" if not family or not extractor_families else
              "SAME_FAMILY" if family in extractor_families else "DISTINCT_DECLARED_FAMILY")
    if status == "SAME_FAMILY" and settings.prefer_distinct_family:
        require(bool(settings.same_family_rationale), "Prefer independent model family; explicit reuse rationale required")
    return AnnotatorIndependence(family, extractor_families, status,
        settings.same_family_rationale or "Family declarations are configured provenance, not proof of statistical independence.")


def parse_annotation(raw, case, protocol, novel_definitions=()):
    require(isinstance(raw, dict), "Annotation response must be a structured object")
    raw = dict(raw)
    for field in ("status", "confidence"):
        if isinstance(raw.get(field), str):
            raw[field] = raw[field].strip().upper()
    raw.setdefault("new_actions", [])
    if raw["new_actions"] is None:
        raw["new_actions"] = []
    if isinstance(raw.get("operations"), dict):
        raw["operations"] = [raw["operations"]]
    raw["operations"] = [dict(row) for row in raw.get("operations", [])]
    hunks = {h.hunk_id: h for h in case.actual_diff.changed_hunks}
    for row in raw["operations"]:
        if isinstance(row.get("confidence"), str):
            row["confidence"] = row["confidence"].strip().upper()
        if isinstance(row.get("diff_hunk_ids"), str):
            row["diff_hunk_ids"] = [row["diff_hunk_ids"]]
        for field, side, text in (("before_evidence", "before", case.before_text), ("after_evidence", "after", case.after_text)):
            spans = []
            source_spans = row.get(field, [])
            source_spans = [source_spans] if isinstance(source_spans, dict) else source_spans
            for original in source_spans:
                item = dict(original)
                require(item.get("side") == side and isinstance(item.get("quote"), str), "Annotation evidence side/quote mismatch")
                start, end, quote = item.get("start"), item.get("end"), item["quote"]
                exact = type(start) is int and type(end) is int and 0 <= start <= end <= len(text) and text[start:end] == quote
                if not exact and quote:
                    first = text.find(quote)
                    require(first >= 0 and text.find(quote, first + 1) < 0, "Annotation quote has no unique exact source match")
                    item.update(start=first, end=first + len(quote))
                elif not exact:
                    anchors = {getattr(hunks[key], side).start for key in row.get("diff_hunk_ids", [])
                               if key in hunks and getattr(hunks[key], side).start == getattr(hunks[key], side).end}
                    require(len(anchors) == 1, "Empty evidence needs an unambiguous changed-side anchor")
                    anchor = next(iter(anchors)); item.update(start=anchor, end=anchor)
                spans.append(item)
            row[field] = spans
    response = AnnotationResponse.from_dict(raw)
    require(not response.new_actions or permits_novel_actions(protocol), "This protocol does not allow new Action definitions")
    known = {a.action_id: a for a in (*protocol.actions, *novel_definitions)}
    meanings = {}
    for definition in known.values():
        if definition.action_id != "OTHER":
            meanings.setdefault(action_meaning_key(definition), []).append(definition)
    aliases, definitions = {}, dict((a.action_id, a) for a in novel_definitions)
    for proposed in response.new_actions:
        require(proposed.target_feature.strip(), "New definitions need a concrete general operation target")
        if proposed.action_id in known:
            require(action_meaning_key(proposed) == action_meaning_key(known[proposed.action_id]),
                    "An existing label cannot be redefined")
            aliases[proposed.action_id] = proposed.action_id
            continue
        require(proposed.action_id not in aliases, "Conflicting temporary Action IDs")
        matches = meanings.get(action_meaning_key(proposed), [])
        require(len(matches) <= 1, "Ambiguous existing Action definition; select its established ID explicitly")
        if matches:
            aliases[proposed.action_id] = matches[0].action_id
        else:
            canonical = replace(proposed, action_id=novel_action_id(proposed), version=NOVEL_ACTION_VERSION)
            aliases[proposed.action_id] = canonical.action_id
            definitions[canonical.action_id] = canonical
            meanings.setdefault(action_meaning_key(canonical), []).append(canonical)
    operations = tuple(replace(row, action_id=aliases.get(row.action_id, row.action_id)) for row in response.operations)
    used = {row.action_id for row in operations}
    response = replace(response, operations=operations, new_actions=tuple(d for key, d in definitions.items() if key in used))
    return response, validate_annotation_response(response, case, protocol, tuple(definitions.values()))


def annotate_cases(cases, protocol, settings, runtime, extractor_families, backend_provider, *, checkpoint_dir=None):
    """Persist complete cases as they finish; resume with the same evidence and protocol."""
    require(runtime.execution_authorized, "Independent annotation execution disabled")
    require(settings.prompt_version == PROMPT_VERSION, "Annotation prompt version mismatch")
    validate_protocol(protocol)
    independent = independence(settings, extractor_families)
    records, attempts, backend = [], [], None
    novel_registry = {}
    if checkpoint_dir is not None:
        configuration = settings.to_dict()
        # A restarted allocation can expose the same model at a different local port.
        configuration["model"].pop("endpoint", None)
        scope = stable_id("action_annotation_checkpoint", [configuration, runtime.configuration_id,
            protocol.protocol_hash, PROMPT_VERSION, DIFF_VERSION, independent.to_dict()])
        checkpoint_dir = Path(checkpoint_dir) / scope

    def remember_novel(operations):
        for operation in operations:
            definition = novel_action_definition(operation.action)
            if definition is not None:
                novel_registry[definition.action_id] = definition

    for case in cases.cases:
        require(case.input_hash == annotation_case_hash(case.identity, case.before_text, case.after_text,
                case.context, case.availability), "Annotation input changed")
        checkpoint = checkpoint_dir / (case.input_hash + ".json") if checkpoint_dir is not None else None
        if checkpoint is not None and checkpoint.is_file():
            saved = read_json(checkpoint)
            payload = saved["payload"]
            require(saved["sha256"] == stable_id("action_checkpoint_record", payload),
                    "Annotation checkpoint integrity mismatch")
            record = ObservedAction.from_dict(payload["record"])
            attempt = AnnotationAttempt.from_dict(payload["attempt"])
            require(record.identity == attempt.identity == case.identity
                    and record.input_hash == attempt.input_hash == case.input_hash
                    and record.protocol_hash == protocol.protocol_hash
                    and record.prompt_version == PROMPT_VERSION
                    and attempt.independence == independent
                    and record.annotation_status in COMPLETE_ACTION_STATUSES, "Annotation checkpoint input differs")
            records.append(record)
            attempts.append(attempt)
            remember_novel(record.operations)
            continue
        response, operations, raw_text, reasons, validation = None, (), "", (), "VALID"
        model = settings.model.model or "UNCONFIGURED"
        model_family, model_revision = settings.model.family, settings.model.revision
        if case.availability == "AVAILABLE":
            validate_diff(case.actual_diff, case.before_text, case.after_text)
        if case.availability != "AVAILABLE":
            reasons, validation = case.limitations, "UNAVAILABLE_INPUT"
            model, model_family, model_revision = "NOT_INVOKED", None, None
        elif not case.actual_diff.changed_hunks:
            response = AnnotationResponse("NO_CHANGE", (), (), "Exact texts are unchanged; no editing Action assigned.",
                                          "HIGH", ())
            response, operations = parse_annotation(response.to_dict(), case, protocol)
            model, model_family, model_revision = "DETERMINISTIC_NO_CHANGE", None, DIFF_VERSION
            validation = "DETERMINISTIC_NO_CHANGE"
        else:
            require(runtime.models_authorized and settings.model.enabled, "Annotation model execution disabled")
            try:
                if backend is None:
                    backend = backend_provider()
                require(backend is not None, "No configured independent annotation backend")
                response_schema = contract_schema(AnnotationResponse)
                response_schema["properties"]["status"]["enum"] = ["ANNOTATED", "OTHER", "UNCERTAIN", "NO_CHANGE"]
                response_schema["properties"]["new_actions"]["items"]["properties"]["version"]["const"] = NOVEL_ACTION_VERSION
                response_schema["properties"]["confidence"]["enum"] = list(CONFIDENCE_VALUES)
                operation_schema = response_schema["properties"]["operations"]["items"]["properties"]
                operation_schema["confidence"]["enum"] = list(CONFIDENCE_VALUES)
                operation_schema["diff_hunk_ids"]["items"]["enum"] = [h.hunk_id for h in case.actual_diff.changed_hunks]
                operation_schema["before_evidence"]["items"]["properties"]["side"]["const"] = "before"
                operation_schema["after_evidence"]["items"]["properties"]["side"]["const"] = "after"
                payload = {
                    "case": case.to_dict(), "frozen_action_protocol": protocol.to_dict(),
                    "previous_new_action_definitions": [a.to_dict() for a in novel_registry.values()],
                    "response_fields": [field.name for field in fields(AnnotationResponse)],
                    "operation_fields": [field.name for field in fields(AnnotationOperationResponse)],
                    "evidence_fields": [field.name for field in fields(EvidenceSpan)],
                    "response_schema": response_schema,
                    "response_schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION}
                def accept(value):
                    parsed, ops = parse_annotation(value, case, protocol, tuple(novel_registry.values()))
                    return value, parsed, ops
                checked = getattr(type(backend), "complete_checked", None)
                raw, response, operations = (checked(backend, SYSTEM, payload, accept) if callable(checked) else
                                             accept(backend.complete(SYSTEM, payload)))
                raw_text = json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False)
                # Only confirmed, complete labels enter the growing actual-edit vocabulary.
                if response.status == "ANNOTATED":
                    remember_novel(operations)
            except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
                response, operations = None, ()
                reasons, validation = ("MODEL_OR_EVIDENCE_VALIDATION_FAILED", type(exc).__name__ + ": " + str(exc)), "INVALID"
        status = AnnotationStatus(response.status) if response is not None else AnnotationStatus.UNKNOWN
        evidence = tuple(dict.fromkeys(e for op in operations for e in op.before_evidence + op.after_evidence))
        explanation = response.explanation if response else "; ".join(reasons)
        key = stable_id("observed_annotation", [case.input_hash, protocol.protocol_hash,
            settings.to_dict(), raw_text, status.value, runtime.configuration_id])
        record = ObservedAction(
            case.identity, tuple(o.action for o in operations), evidence, status,
            (MetadataItem("independence", independent.status), MetadataItem("validation", validation)),
            settings.version, key, operations, explanation,
            response.confidence if response else "UNCERTAIN", response.uncertainty if response else reasons,
            model, model_family, model_revision, PROMPT_VERSION, case.input_hash, protocol.protocol_hash,
            response.unassigned_diff_hunk_ids if response else ())
        attempt = AnnotationAttempt(case.identity, case.input_hash, raw_text, validation, reasons, independent)
        if checkpoint is not None:
            payload = {"record": record.to_dict(), "attempt": attempt.to_dict()}
            target = (checkpoint if status in COMPLETE_ACTION_STATUSES else
                      checkpoint.parent / "unresolved" / checkpoint.name)
            write_json(target, {"payload": payload, "sha256": stable_id("action_checkpoint_record", payload)})
        records.append(record)
        attempts.append(attempt)
    return tuple(records), AnnotationAudit(tuple(attempts))
