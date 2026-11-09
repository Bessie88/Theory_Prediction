"""Pure compiler from registered ORIGINAL theory/procedures to a frozen application view."""
from dataclasses import replace
import json
from theory_discovery.core.application_contracts import (
    FrozenAuthority, TheoryDeclaration, ActionCompatibility, ActionAlias, FrozenSourceAction,
    FrozenTheoryApplication, FrozenTheoryRegistry, theory_registry_hash, study_hash,
)
from theory_discovery.core.workflow_contracts import ActionDefinition, Method
from theory_discovery.features._support.feature_spec_registry import freeze_measurements
from theory_discovery.features._support.condition_normalization import clause_inventory
from theory_discovery.induction._support.theory_logic import validate_logic, clause_references
from theory_discovery.core.schemas import require, stable_id

VERSION = "theory-application-v1"


def _source_action(representation, method=Method.THEORY):
    require(representation.record_type == "ACTION", "Expected original source action")
    raw = json.loads(representation.original_json)
    require(isinstance(raw, dict), "Original action must be an object")
    require(representation.record_id == stable_id("native_action" if method is Method.THEORY else "npov_action", raw), "Original action identity mismatch")
    # No operation taxonomy is invented; family/method/target/scope are native strings.
    allowed = {"group_id", "family", "method", "target_feature", "scope", "semantic",
               "direction", "transformation", "target_expression", "action_id"}
    require(set(raw) <= allowed, "Unsupported source action payload; no examples/gold/After copied into prediction")
    keys = ("family", "method", "target_feature", "scope")
    valid = all(type(raw.get(k)) is str and raw[k].strip() for k in keys)
    direction = raw.get("direction")
    require(direction is None or type(direction) is str, "Invalid original action direction")
    definition = (ActionDefinition(
        representation.record_id, raw["family"], raw["method"], raw["target_feature"],
        direction, raw["scope"], representation.original_json, VERSION) if valid else None)
    # Only a native GROUP identity is excluded when testing exact duplicate semantics.
    signature = stable_id("native_action_semantics", {k: v for k, v in raw.items() if k != "group_id"})
    return FrozenSourceAction(representation.record_id, representation.original_json, signature,
                              definition, ("OTHER" if raw.get("family") == "OTHER" else "NATIVE") if valid else "UNMAPPED",
                              representation.provenance.json_pointer)


def _combine(registered, configured, key):
    rows = {}
    for item in (*registered, *configured):
        identity = key(item)
        require(identity not in rows or rows[identity] == item,
                "Upstream and configured application declarations conflict")
        rows[identity] = item
    return tuple(rows.values())


def compile_knowledge_registry(bundle, measurements, settings, catalog=None):
    method = bundle.knowledge_source
    require(measurements.knowledge_source is method, "Knowledge-source planning mismatch")
    require(settings.version == VERSION and measurements.source_theory_hash == bundle.native_source.sha256,
            "Theory/procedure generation mismatch")
    require(measurements.clauses == clause_inventory(bundle),
            "Measurement clauses differ from original theory representation")
    authority_rows = _combine(measurements.application_authorities,
        tuple(FrozenAuthority.from_dict(json.loads(row)) for row in settings.authorities_json),
        lambda row: row.authority_id)
    authorities = {a.authority_id: a for a in authority_rows}
    require(len(authorities) == len(authority_rows), "Duplicate application authority")
    anchors = {c.source_pointer for c in measurements.clauses}
    anchors |= {r.provenance.json_pointer for r in bundle.source_rules + bundle.source_actions}
    # Whole branch anchors may justify relationships among its fields.
    anchors |= {"/".join(c.source_pointer.split("/")[:3]) for c in measurements.clauses
                if c.source_pointer.startswith("/branches/")}
    for authority in authority_rows:
        if authority.kind in ("SOURCE_THEORY", "SOURCE_POLICY"):
            require((authority.kind == "SOURCE_THEORY") == (method is Method.THEORY),
                    "One knowledge source cannot authorize rules from the other")
            require(authority.source_theory_sha256 == bundle.native_source.sha256 and
                    authority.source_pointers and set(authority.source_pointers) <= anchors,
                    "Source-theory authority must resolve to this frozen native output")
    declarations = _combine(measurements.application_declarations,
        tuple(TheoryDeclaration.from_dict(json.loads(row)) for row in settings.declarations_json),
        lambda row: row.theory_id)
    declared = {d.theory_id: d for d in declarations}
    require(len(declared) == len(declarations) and
            set(declared) <= {t.theory_id for t in bundle.theories}, "Unknown/duplicate declared theory")
    actions = tuple(_source_action(row, method) for row in bundle.source_actions)
    action_map = {a.source_action_id: a for a in actions}
    require(len(action_map) == len(actions), "Duplicate frozen source action")
    aliases = _combine(measurements.action_aliases,
        tuple(ActionAlias.from_dict(json.loads(row)) for row in settings.action_aliases_json),
        lambda row: row.source_action_id)
    require(len({a.source_action_id for a in aliases}) == len(aliases), "Duplicate source action alias")
    shared = {} if catalog is None else {a.action_id: a for a in catalog.actions}
    alias_meanings = {}
    for alias in aliases:
        require(alias.source_action_id in action_map and alias.catalog_action_id in shared and
                alias.authority_id in authorities and bool(alias.rationale.strip()),
                "Action alias must reference source, shared vocabulary and frozen authority")
        source, target = action_map[alias.source_action_id], shared[alias.catalog_action_id]
        require(source.definition is not None and all(
            getattr(source.definition, key) == getattr(target, key) for key in
            ("family", "method", "target_feature", "direction", "scope")),
            "Action alias changes the original operation semantics")
        previous = alias_meanings.get(target.action_id)
        require(previous is None or previous == source.semantic_signature,
                "Shared action ID would collapse distinct granular source operations")
        alias_meanings[target.action_id] = source.semantic_signature
        action_map[alias.source_action_id] = replace(source, definition=target, mapping_status="CATALOG_MAPPED")
    actions = tuple(action_map[a.source_action_id] for a in actions)
    specs = {s.feature_id: s for s in measurements.specifications}
    bindings = {b.clause_id: b for b in measurements.bindings}
    require(len(bindings) == len(measurements.bindings) and
            set(bindings) == {c.clause_id for c in measurements.clauses}, "Incomplete measurement bindings")
    frozen_theories = []
    for theory in bundle.theories:
        clauses = tuple(c for c in measurements.clauses if c.theory_id == theory.theory_id)
        local = {c.clause_id: bindings[c.clause_id] for c in clauses}
        sources = {}
        for option in theory.actions:
            for action_id in option.action_ids:
                sources.setdefault(action_id, set()).update(option.source_rule_ids)
        require(set(sources) <= set(action_map), "Original theory action representation missing")
        declaration = declared.get(theory.theory_id)
        if declaration is None:
            frozen_theories.append(FrozenTheoryApplication(
                theory.theory_id, theory.native_branch_id, theory.source_rule_ids,
                tuple(local), '{"op":"constant","value":"UNKNOWN"}', (), "UNRESOLVED",
                tuple(sorted(sources)), (), None, "UNRESOLVED",
                ("Natural prose has no declared aggregate logic; do not conjoin source alternatives.",
                 "Source action options are preserved without choosing their first member.")))
            continue
        require(declaration.authority_id in authorities, "Theory declaration has no frozen authority")
        expression = json.loads(declaration.applicability_expression_json)
        validate_logic(expression, local)
        refs = clause_references(expression)
        nonempty_conditions = {c.clause_id for c in clauses if c.role == "CONDITION" and c.original_text.strip()}
        require(not nonempty_conditions or refs & nonempty_conditions,
                "A declaration cannot discard all original conditions and become constant true")
        require(len({d.directive_id for d in declaration.directives}) == len(declaration.directives),
                "Duplicate action directive")
        require({d.source_action_id for d in declaration.directives} == set(sources),
                "Every original action alternative needs an explicit retained directive")
        if declaration.composition == "SINGLE":
            require(len(sources) == 1, "SINGLE cannot erase a native multi-action theory")
        for directive in declaration.directives:
            require(directive.authority_id in authorities and
                    set(directive.source_rule_ids) <= sources[directive.source_action_id],
                    "Action mapping is not backed by its original theory rules")
            gate = json.loads(directive.when_expression_json)
            validate_logic(gate, local)
            refs |= clause_references(gate)
            if directive.target_feature_id is not None:
                require(directive.target_feature_id in specs and
                        theory.theory_id in specs[directive.target_feature_id].associated_theory_ids,
                        "Target selector must be a frozen, theory-grounded Before feature")
        omissions = {o.clause_id: o for o in declaration.omitted_clauses}
        require(len(omissions) == len(declaration.omitted_clauses), "Duplicate source clause disposition")
        require(all(o.authority_id in authorities and bool(o.reason.strip()) for o in omissions.values()),
                "Omitted/redundant views need explicit frozen justification")
        nontrivial = {c.clause_id for c in clauses if local[c.clause_id].resolution != "NO_ADDITIONAL_BOUNDARY"}
        require(not refs & set(omissions) and refs | set(omissions) == nontrivial,
                "Source clause lost, invented, or simultaneously used/omitted")
        frozen_theories.append(FrozenTheoryApplication(
            theory.theory_id, theory.native_branch_id, theory.source_rule_ids, tuple(local),
            json.dumps(expression, sort_keys=True), declaration.directives, declaration.composition,
            tuple(sorted(sources)), declaration.omitted_clauses, declaration.authority_id, "DECLARED",
            (declaration.fidelity_rationale, "Native granularity retained; no fresh editing decision generated.")))
    relations = _combine(measurements.action_compatibility,
        tuple(ActionCompatibility.from_dict(json.loads(row)) for row in settings.compatibility_json),
        lambda row: tuple(sorted((row.left_action_id, row.right_action_id))))
    seen, semantic_relations = set(), {}
    for relation in relations:
        pair = tuple(sorted((relation.left_action_id, relation.right_action_id)))
        require(pair not in seen and set(pair) <= set(action_map) and relation.authority_id in authorities,
                "Duplicate/unknown/unauthorized action compatibility")
        seen.add(pair)
        semantic_pair = tuple(sorted(action_map[key].semantic_signature for key in pair))
        require(semantic_pair[0] != semantic_pair[1] or relation.relation == "COMPATIBLE",
                "Exact duplicate source actions cannot be declared incompatible")
        previous = semantic_relations.get(semantic_pair)
        require(previous is None or previous == relation.relation,
                "Contradictory compatibility declarations for equivalent source actions")
        semantic_relations[semantic_pair] = relation.relation
    registry = FrozenTheoryRegistry(bundle.editor, bundle.native_source.sha256, tuple(frozen_theories),
        actions, authority_rows, relations, VERSION, "", aliases, method, bundle.source_snapshot_id, bundle.unresolved_source_sections)
    return replace(registry, registry_hash=theory_registry_hash(registry))


def compile_theory_registry(bundle, measurements, settings, catalog=None):
    require(bundle.knowledge_source is Method.THEORY, "Theory bridge cannot consume policy rules")
    return compile_knowledge_registry(bundle, measurements, settings, catalog)


def validate_study(study):
    registry = study.theory_registry
    require(registry.version == VERSION and study.conflict_policy_version == "source-backed-compatibility-v1",
            "Application/conflict protocol version not installed")
    require(registry.registry_hash == theory_registry_hash(registry) and study.freeze_hash == study_hash(study),
            "Frozen study/theory registry changed")
    frozen = study.measurement_freeze
    require(frozen == freeze_measurements(frozen.snapshot, frozen.clauses, frozen.bindings,
                                         frozen.reviews, frozen.source_theory_hash),
            "Frozen feature specification/prompt/procedure changed")
    require(frozen.source_theory_hash == registry.source_theory_hash,
            "Application and measurement do not use the same original theory")
    require(len({t.theory_id for t in registry.theories}) == len(registry.theories),
            "Duplicate frozen theory")
    if study.policy_registry is not None:
        policy = study.policy_registry
        require(policy.knowledge_source is Method.NPOV and registry.knowledge_source is Method.THEORY and
                policy.registry_hash == theory_registry_hash(policy), "Policy/theory registry source mismatch")
        require(study.policy_measurement_freeze is not None and
                study.policy_measurement_freeze.snapshot == frozen.snapshot,
                "Both sources must use the same frozen extraction registry")
        pf = study.policy_measurement_freeze
        require(pf == freeze_measurements(pf.snapshot, pf.clauses, pf.bindings, pf.reviews, pf.source_theory_hash)
                and pf.source_theory_hash == policy.source_theory_hash == study.policy_snapshot_hash,
                "Policy source/measurement freeze changed")
        require(study.require_shared_action_ids and policy.editor == registry.editor and policy.source_snapshot_id,
                "Integrated sources require the same editor and label space")
    return registry
