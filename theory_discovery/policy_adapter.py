"""Verified official guidance -> structured policy records; no induced theory input."""
from dataclasses import replace
from typing import Protocol
import json
from .policy_contracts import (
    PolicyRule, PolicySectionAccounting, PolicyRuleRegistry, policy_registry_hash,
    PolicySourceUnavailable,
)
from .policy_ingestion import validate_citation
from .policy_readable import readable_policy, resolve_readable_quote, PROJECTION_VERSION
from .schemas import require, stable_id
from .llm import PLAIN_LANGUAGE_INSTRUCTION

CONVERSION_VERSION = "official-npov-structure-v1"
PROMPT_VERSION = "npov-policy-adapter-v6"
POLICY_ADAPTER_PROMPT = (
    "Turn only the supplied official NPOV snapshot into rules. Preserve conditions, boundary scope/polarity "
    "and uncertainty; cite exact Unicode offsets and quotes. Use catalog Action IDs only when policy supports "
    "the operation. A general goal does not imply a specific edit: use NOT_SPECIFIED and no actions when unspecified. "
    "Do not force allowable choices into joint or exclusive actions; exclusivity needs explicit policy support. "
    "Convert only the focus section in this request; the full snapshot supplies context and exceptions. "
    "If focus_excerpt is present, convert only that excerpt; the full source supplies context and exceptions. "
    "Keep citations short; do not repeat long policy paragraphs. Every rule must cite the focus section. Return exactly one accounting row for it, including guidance-only sections. "
    "Use the typed schema and op/args logic grammar. JSON expression fields may be objects. "
    "Keep responses compact with short exact quotes; merge repeated guidance while preserving distinct conditions and operations. "
    "Quotes may copy the raw source or readable_focus_section exactly; Python maps readable or approximately matching quotes "
    "back to verified raw wikitext. Offsets refer to raw Unicode source; prefer short source quotations; minor typography or wording differences are tolerated. "
    "Explicit ... or … omission between ordered source fragments is allowed; full source spans are restored. "
    "Templates are not expanded in the readable projection. Matching is restricted to the declared section; unsupported quotations are rejected. "
    "NOT_SPECIFIED or UNRESOLVED operations must have actions=[] and an underspecification reason, "
    "never an OTHER placeholder. Use no editor theories or edit outcomes. Return only rules and section_accounting."
) + "\n" + PLAIN_LANGUAGE_INSTRUCTION


class PolicyConversionBackend(Protocol):
    def complete(self, system: str, payload: dict) -> dict:
        ...


def _remap(expression, mapping):
    if isinstance(expression, list):
        return [_remap(value, mapping) for value in expression]
    if isinstance(expression, dict):
        result = {key: _remap(value, mapping) for key, value in expression.items()}
        if "clause_id" in result:
            result["clause_id"] = mapping.get(result["clause_id"], result["clause_id"])
        return result
    return expression


def canonical_policy_rule_id(rule):
    body = rule.to_dict()
    body.pop("rule_id")
    return stable_id("npov_rule", body)


def adapt_policy(snapshot, catalog, backend=None, converted_json=None):
    if converted_json is not None:
        raw = json.loads(converted_json)
    else:
        if backend is None:
            raise PolicySourceUnavailable("POLICY_CONVERSION_UNAVAILABLE: no configured conversion/verified rule packet")
        # Process every section; each request still receives the whole official source.
        # This bounds response size without dropping cross-section exceptions/context.
        from .workflow_contracts import contract_schema
        from .policy_contracts import PolicyClause, PolicyAction, PolicyCitation
        rule_schema = contract_schema(PolicyRule)
        props = rule_schema["properties"]
        props["rule_id"]["pattern"] = "^npov_rule_"
        props["snapshot_id"]["const"] = snapshot.metadata.snapshot_id
        props["conversion_version"]["const"] = CONVERSION_VERSION
        props["action_specificity"]["enum"] = ["EXPLICIT_OPERATION", "ALLOWABLE_OPERATIONS", "NOT_SPECIFIED"]
        props["action_composition"]["enum"] = ["SINGLE", "ALL_SUPPORTED", "ALLOWABLE_ALTERNATIVES", "EXCLUSIVE_ALTERNATIVES", "UNRESOLVED"]
        props["conditions"]["items"]["properties"]["role"]["enum"] = ["CONDITION"]
        props["boundaries"]["items"]["properties"]["role"]["enum"] = ["BOUNDARY_EXCLUDES", "BOUNDARY_REQUIRES", "BOUNDARY_RESTRICTS", "BOUNDARY_UNRESOLVED"]
        expression_schema = {"anyOf": [{"type": "object"}, {"type": "string", "contentMediaType": "application/json"}]}
        props["applicability_expression_json"] = expression_schema
        props["actions"]["items"]["properties"]["when_expression_json"] = expression_schema
        props["actions"]["items"]["properties"]["catalog_action_id"]["enum"] = [a.action_id for a in catalog.actions]
        accounting_schema = contract_schema(PolicySectionAccounting)
        accounting_schema["properties"]["disposition"]["enum"] = ["RULES", "GUIDANCE_ONLY", "UNRESOLVED"]
        payload = {
            "snapshot_metadata": snapshot.metadata.to_dict(), "official_policy_text": snapshot.content,
            "neutral_action_catalog": [a.to_dict() for a in catalog.actions],
            "conversion_version": CONVERSION_VERSION,
            "expression_examples": [{"op": "clause", "clause_id": "ID"}, {"op": "constant", "value": "TRUE"},
                {"op": "and", "args": [{"op": "clause", "clause_id": "A"}, {"op": "boundary_clear", "clause_id": "B"}]}],
            "output_schema": {"type": "object", "properties": {
                "rules": {"type": "array", "items": rule_schema},
                "section_accounting": {"type": "array", "minItems": 1, "maxItems": 1, "items": accounting_schema}},
                "required": ["rules", "section_accounting"], "additionalProperties": False}}
        collected, accounting = {}, []
        for section in snapshot.metadata.sections:
            request = dict(payload, focus_section=section.to_dict(),
                readable_focus_section=readable_policy(snapshot.content[section.start:section.end],
                    section.start, snapshot.metadata.content_format == "wikitext").text,
                readable_projection_version=PROJECTION_VERSION)
            def accept(result):
                result = _policy_wire(snapshot, result)
                parsed = _decode_packet(snapshot, catalog, result, (section.section_id,))
                require(all(section.section_id in rule.section_ids for rule in parsed.rules),
                        "Each extracted rule must cite the focus section")
                return parsed
            checked = getattr(type(backend), "complete_checked", None)
            packet = (checked(backend, POLICY_ADAPTER_PROMPT, request, accept) if callable(checked)
                      else accept(backend.complete(POLICY_ADAPTER_PROMPT, request)))
            for rule in packet.rules:
                if rule.rule_id in collected:
                    require(collected[rule.rule_id] == rule, "Conflicting canonical policy rule")
                collected[rule.rule_id] = rule
            accounting.extend(packet.section_accounting)
        # Account also for cross-section citations used to preserve source qualifications.
        accounting = [replace(row, rule_ids=tuple(sorted(set(row.rule_ids) |
                        {rule.rule_id for rule in collected.values() if row.section_id in rule.section_ids})))
                      for row in accounting]
        accounting = [replace(row, disposition="RULES") if row.rule_ids else row for row in accounting]
        raw = {"rules": [r.to_dict() for r in collected.values()], "section_accounting": [r.to_dict() for r in accounting]}
    return _decode_packet(snapshot, catalog, _policy_wire(snapshot, raw))


def _policy_wire(snapshot, value):
    """Normalize representation; quotes must still match the verified official source."""
    if isinstance(value, list):
        return [_policy_wire(snapshot, row) for row in value]
    if not isinstance(value, dict):
        return value
    value = {key: _policy_wire(snapshot, row) for key, row in value.items()}
    for key in ("applicability_expression_json", "when_expression_json"):
        if isinstance(value.get(key), dict):
            value[key] = json.dumps(value[key], sort_keys=True)
    for key in ("uncertainty", "underspecification"):
        if isinstance(value.get(key), str):
            value[key] = [value[key]]
    # OTHER is a wire placeholder, not a specified executable policy operation.
    # Honor the model's explicit unspecified status; preserve its explanation.
    if (value.get("action_specificity") == "NOT_SPECIFIED" or value.get("action_composition") == "UNRESOLVED"):
        actions = value.get("actions")
        if isinstance(actions, list) and actions and all(
                isinstance(action, dict) and action.get("catalog_action_id") == "OTHER" for action in actions):
            reasons = list(value.get("underspecification", []))
            reasons.extend("Unspecified operation (OTHER placeholder): " + action.get("source_meaning", "")
                           for action in actions)
            value.update(actions=[], underspecification=reasons)
    if {"section_id", "start", "end", "quote"} <= set(value):
        sections = {s.section_id: s for s in snapshot.metadata.sections}
        section = sections.get(value["section_id"])
        quote = value["quote"]
        require(section is not None and isinstance(quote, str) and bool(quote), "Unknown citation section or empty quote")
        start, end = value["start"], value["end"]
        exact = (type(start) is int and type(end) is int and section.start <= start < end <= section.end
                 and snapshot.content[start:end] == quote)
        if not exact:
            first = snapshot.content.find(quote, section.start, section.end)
            if first >= 0:
                value.update(start=first, end=first + len(quote))
            else:
                start, end, raw_quote = resolve_readable_quote(snapshot, section, quote)
                value.update(start=start, end=end, quote=raw_quote)
    return value


def _decode_packet(snapshot, catalog, raw, required_sections=None):
    required_sections = set(required_sections or [s.section_id for s in snapshot.metadata.sections])
    require(isinstance(raw, dict) and set(raw) == {"rules", "section_accounting"} and
            isinstance(raw["rules"], list) and isinstance(raw["section_accounting"], list),
            "Policy conversion has unexpected fields")
    catalog_ids = {action.action_id for action in catalog.actions}
    rules, aliases = [], {}
    for source in raw["rules"]:
        rule = PolicyRule.from_dict(source)
        require(rule.snapshot_id == snapshot.metadata.snapshot_id and
                rule.conversion_version == CONVERSION_VERSION, "Policy conversion version/source mismatch")
        require(set(rule.section_ids) == {c.section_id for c in rule.citations}, "Policy rule section mismatch")
        clauses = rule.conditions + rule.boundaries
        require(len({a.catalog_action_id for a in rule.actions}) == len(rule.actions),
                "Repeat action gates must be combined explicitly without duplicate catalog labels")
        require(len({c.clause_id for c in clauses}) == len(clauses), "Duplicate policy clause ID")
        require({v.section_id for c in clauses for v in c.citations} |
                {v.section_id for a in rule.actions for v in a.citations} <= set(rule.section_ids),
                "Clause/action citations must be included in the rule section references")
        for citation in (*rule.citations, *(v for c in clauses for v in c.citations),
                         *(v for a in rule.actions for v in a.citations)):
            validate_citation(snapshot, citation)
        require(all(action.catalog_action_id in catalog_ids and action.citations and action.source_meaning.strip()
                    for action in rule.actions), "Policy action needs shared label and policy evidence")
        clause_aliases = {c.clause_id: stable_id("npov_clause", [
            snapshot.metadata.snapshot_id, c.role, c.meaning, [v.to_dict() for v in c.citations]])
            for c in clauses}
        rule = replace(rule,
            conditions=tuple(replace(c, clause_id=clause_aliases[c.clause_id]) for c in rule.conditions),
            boundaries=tuple(replace(c, clause_id=clause_aliases[c.clause_id]) for c in rule.boundaries),
            applicability_expression_json=json.dumps(_remap(
                json.loads(rule.applicability_expression_json), clause_aliases), sort_keys=True),
            actions=tuple(replace(action, when_expression_json=json.dumps(_remap(
                json.loads(action.when_expression_json), clause_aliases), sort_keys=True)) for action in rule.actions))
        clauses = rule.conditions + rule.boundaries
        require(len({c.clause_id for c in clauses}) == len(clauses),
                "Duplicate canonical policy clauses need an explicit source-faithful merge")
        # Structural policy logic is validated by the SAME source-clause grammar.
        from types import SimpleNamespace
        from .theory_logic import validate_logic
        roles = {c.clause_id: SimpleNamespace(logical_role=c.role) for c in clauses}
        validate_logic(json.loads(rule.applicability_expression_json), roles)
        for action in rule.actions:
            validate_logic(json.loads(action.when_expression_json), roles)
        normalized = replace(rule, rule_id=canonical_policy_rule_id(rule))
        require(rule.rule_id not in aliases, "Duplicate provisional policy rule ID")
        aliases[rule.rule_id] = normalized.rule_id
        rules.append(normalized)
    require(len({r.rule_id for r in rules}) == len(rules), "Duplicate canonical policy rules")
    accounting = tuple(PolicySectionAccounting.from_dict(row) for row in raw["section_accounting"])
    require(len(accounting) == len(required_sections) and
            {row.section_id for row in accounting} == required_sections,
            "Every official section must be accounted for")
    accounting = tuple(replace(row, rule_ids=tuple(aliases.get(key, key) for key in row.rule_ids))
                       for row in accounting)
    ids = {rule.rule_id for rule in rules}
    require(all(set(row.rule_ids) <= ids for row in accounting), "Unknown policy rule in section accounting")
    for rule in rules:
        require(all(any(rule.rule_id in row.rule_ids and row.section_id == section for row in accounting)
                    for section in set(rule.section_ids) & required_sections),
                "Policy rule lacks source section accounting")
    registry = PolicyRuleRegistry(snapshot.metadata, snapshot.local_reference, snapshot.metadata_sha256,
                                  tuple(rules), accounting, CONVERSION_VERSION, "")
    return replace(registry, registry_hash=policy_registry_hash(registry))
