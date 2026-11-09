"""Official-policy-only projection into the shared canonical rule/planning shape.

Legacy field names in the shared IR do not import induced theories. All objects
here are built exclusively from a verified PolicyRuleRegistry and neutral catalog.
"""
from dataclasses import replace
from hashlib import sha256
import json
from .workflow_contracts import (
    KnowledgeRuleRecord, ProseClause, ActionOption, NativeRepresentation, Method,
)
from .policy_contracts import policy_registry_hash
from .application_contracts import FrozenAuthority, TheoryDeclaration, ActionDirective, ActionAlias, ClauseOmission
from .schemas import require, stable_id
from .policy_clause_dispositions import REVIEW_VERSION, SNAPSHOT_SHA256, DISPOSITIONS
from .theory_logic import clause_references

PROJECTION_VERSION = 'policy-application-v2'


def planning_bundle(registry, editor, catalog):
    require(registry.registry_hash == policy_registry_hash(registry), "Policy rule registry changed")
    shared = {a.action_id: a for a in catalog.actions}
    source = registry.snapshot_reference
    records, original, actions = [], [], {}
    for index, rule in enumerate(registry.rules):
        def clause(value, position, category):
            return ProseClause(value.meaning, (rule.rule_id,),
                f"/policy_rules/{index}/{category}/{position}", "UNPLANNED",
                value.role, value.clause_id)
        conditions = tuple(clause(c, i, "conditions") for i, c in enumerate(rule.conditions))
        boundaries = tuple(clause(c, i, "boundaries") for i, c in enumerate(rule.boundaries))
        options = []
        for action_index, policy_action in enumerate(rule.actions):
            require(policy_action.catalog_action_id in shared, "Policy action is outside the common catalog")
            definition = shared[policy_action.catalog_action_id]
            raw = {"group_id": rule.rule_id, "family": definition.family, "method": definition.method,
                   "target_feature": definition.target_feature, "scope": definition.scope,
                   "direction": definition.direction, "semantic": True}
            action_id = stable_id("npov_action", raw)
            actions.setdefault(action_id, NativeRepresentation(
                action_id, "ACTION", json.dumps(raw, sort_keys=True),
                replace(source, json_pointer=f"/policy_rules/{index}/actions/{action_index}")))
            options.append(ActionOption((action_id,), (rule.rule_id,), (action_id,), "UNSPECIFIED"))
        records.append(KnowledgeRuleRecord(
            rule.rule_id, editor, rule.original_meaning, rule.rule_id, conditions, boundaries,
            tuple(options), (), (rule.rule_id,), (source,), "VERIFIED_SOURCE_LINKS",
            rule.uncertainty + rule.underspecification, (), (), json.dumps(rule.to_dict(), sort_keys=True),
            Method.NPOV))
        original.append(NativeRepresentation(rule.rule_id, "RULE", json.dumps(rule.to_dict(), sort_keys=True),
                        replace(source, json_pointer=f"/policy_rules/{index}")))
    from .stage_interfaces import TheoryBundle
    return TheoryBundle(editor, tuple(records), source, (),
        ("Official policy source; no induced-theory or editing-action evidence used.",),
        tuple(original), (), tuple(actions.values()), (), "[]", Method.NPOV, registry.snapshot.snapshot_id,
        tuple(row.section_id for row in registry.section_accounting if row.disposition == "UNRESOLVED"))


def attach_policy_application(plan, bundle, registry):
    """Transfer only policy-derived logic into registered measurement metadata."""
    from .policy_adapter import _remap
    authorities, declarations, aliases = [], [], []
    for index, rule in enumerate(registry.rules):
        clause_map = {s.original_clause_id: s.clause_id for s in plan.clauses if s.theory_id == rule.rule_id}
        authority_id = stable_id("npov_authority", [registry.registry_hash, rule.rule_id])
        pointers = tuple(c.source_pointer for c in plan.clauses if c.theory_id == rule.rule_id)
        pointers += tuple(r.provenance.json_pointer for r in bundle.source_rules if r.record_id == rule.rule_id)
        authorities.append(FrozenAuthority(authority_id, "SOURCE_POLICY", PROJECTION_VERSION,
            sha256(json.dumps(registry.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            registry.snapshot.content_sha256, pointers, "Verified policy citation/structured operationalization.",
            True))
        directives = []
        for action_index, action in enumerate(rule.actions):
            native = bundle.theories[index].actions[action_index].action_ids[0]
            directives.append(ActionDirective(
                stable_id("npov_directive", [rule.rule_id, action_index]), native, (rule.rule_id,),
                json.dumps(_remap(json.loads(action.when_expression_json), clause_map), sort_keys=True),
                "SOURCE_LEVEL", None, authority_id))
            aliases.append(ActionAlias(native, action.catalog_action_id, authority_id, action.source_meaning))
        expression = json.dumps(_remap(json.loads(rule.applicability_expression_json), clause_map), sort_keys=True)
        composition = rule.action_composition
        if composition == "UNRESOLVED":
            # Applicability may be measured while the operation remains absent.
            # Shared prediction then reports UNKNOWN instead of inventing a strategy.
            composition = "ALL_SUPPORTED"
        refs = clause_references(json.loads(expression))
        for directive in directives:
            refs |= clause_references(json.loads(directive.when_expression_json))
        local = [c for c in plan.clauses if c.theory_id == rule.rule_id]
        omitted, unknown_scope = [], False
        for clause in local:
            if clause.clause_id in refs:
                continue
            require(clause.role != "CONDITION", "Policy application lost an original condition")
            review = DISPOSITIONS.get(clause.original_clause_id) if registry.snapshot.content_sha256 == SNAPSHOT_SHA256 else None
            if review is not None:
                require(review["meaning"] == clause.original_text, "Policy clause disposition/source mismatch")
                kind, reason = review["kind"], review["reason"]
            else:
                kind, reason = "UNRESOLVED_SCOPE", "No reviewed source-faithful application disposition; preserve unknown applicability."
            unknown_scope |= kind == "UNRESOLVED_SCOPE"
            # Explicit disposition, never deletion of the source/measurement record.
            # Output requirements remain descriptive; no After data or new operation is used.
            omitted.append(ClauseOmission(clause.clause_id, REVIEW_VERSION + " / " + kind + ": " + reason,
                                           authority_id))
        if unknown_scope:
            expression = json.dumps({"op": "and", "args": [json.loads(expression),
                {"op": "constant", "value": "UNKNOWN"}]}, sort_keys=True)
        declarations.append(TheoryDeclaration(rule.rule_id, expression, tuple(directives), composition,
                                             tuple(omitted), authority_id, rule.original_meaning))
    return replace(plan, application_authorities=tuple(authorities),
                   application_declarations=tuple(declarations), action_aliases=tuple(aliases))
