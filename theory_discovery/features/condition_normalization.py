"""Source clause identity and finite, downstream measurement expressions."""
import json
import math
from .measurement_contracts import ClauseSource, ClauseBinding
from theory_discovery.core.workflow_contracts import StructuredValue, stable_id, require


def clause_inventory(bundle):
    rows = []
    method = bundle.knowledge_source
    for theory in bundle.theories:
        for role, clauses in (("CONDITION", theory.conditions),
                              ("BOUNDARY_UNRESOLVED", theory.boundaries)):
            for clause in clauses:
                key = stable_id("measurement_clause", [
                    theory.theory_id, role, clause.original_field_pointer,
                    clause.source_rule_ids, clause.text, method.value, clause.original_clause_id])
                rows.append(ClauseSource(key, theory.theory_id, clause.source_rule_ids,
                                         clause.original_field_pointer, clause.text, role, method,
                                         clause.logical_role_hint, clause.original_clause_id))
    require(len({r.clause_id for r in rows}) == len(rows), "Duplicate source clause")
    return tuple(rows)


def validate_expression(expression, feature_ids, depth=0):
    require(depth <= 24 and isinstance(expression, dict), "Expression depth/type invalid")
    op = expression.get("op")
    if op == "constant":
        require(set(expression) == {"op", "value"} and
                expression["value"] in ("TRUE", "FALSE", "UNKNOWN"), "Invalid truth constant")
    elif op in ("and", "or"):
        require(set(expression) == {"op", "args"} and isinstance(expression["args"], list) and
                bool(expression["args"]), "Composite expression needs operands")
        for child in expression["args"]:
            validate_expression(child, feature_ids, depth + 1)
    elif op == "not":
        require(set(expression) == {"op", "arg"}, "Invalid NOT")
        validate_expression(expression["arg"], feature_ids, depth + 1)
    elif op in ("feature", "compare"):
        expected = {"op", "feature_id"} if op == "feature" else {"op", "feature_id", "operator", "value"}
        require(set(expression) == expected and expression["feature_id"] in feature_ids,
                "Unknown feature expression")
        if op == "compare":
            require(expression["operator"] in ("eq", "ne", "gt", "ge", "lt", "le"),
                    "Unknown comparison")
            require(type(expression["value"]) in (str, int, float, bool), "Scalar comparison only")
            require(type(expression["value"]) is not float or math.isfinite(expression["value"]),
                    "Comparison threshold must be finite")
    elif op == "exists_same_candidate":
        require(set(expression) == {"op", "feature_ids"} and
                isinstance(expression["feature_ids"], list) and len(expression["feature_ids"]) >= 2 and
                len(set(expression["feature_ids"])) == len(expression["feature_ids"]) and
                set(expression["feature_ids"]) <= set(feature_ids), "Invalid shared-witness expression")
    else:
        raise ValueError("Unsupported expression operator")


def referenced_features(expression):
    if expression["op"] in ("feature", "compare"):
        return {expression["feature_id"]}
    if expression["op"] == "exists_same_candidate":
        return set(expression["feature_ids"])
    if expression["op"] in ("and", "or"):
        return set().union(*(referenced_features(c) for c in expression["args"]))
    if expression["op"] == "not":
        return referenced_features(expression["arg"])
    return set()


def validate_binding(source, binding, specifications):
    require(source.clause_id == binding.clause_id, "Source clause mismatch")
    require(source.declared_role is None or source.declared_role == binding.logical_role,
            "Measurement changed the declared policy boundary/condition role")
    if source.role == "CONDITION":
        require(binding.logical_role == "CONDITION", "Condition role cannot become a boundary")
    else:
        require(binding.logical_role.startswith("BOUNDARY_"), "Boundary role cannot become a condition")
    expression = json.loads(binding.expression_json)
    ids = {s.feature_id for s in specifications}
    validate_expression(expression, ids)
    spec_by_id = {s.feature_id: s for s in specifications}
    def typed(node):
        op = node["op"]
        if op == "feature":
            require(spec_by_id[node["feature_id"]].value_type == "boolean",
                    "Non-Boolean measurements need an explicit comparison")
        elif op == "compare":
            spec = spec_by_id[node["feature_id"]]
            value = node["value"]
            if spec.value_type in ("number", "numerical"):
                require(type(value) in (int, float), "Numerical comparison threshold required")
            elif spec.value_type in ("categorical", "string"):
                require(type(value) is str and node["operator"] in ("eq", "ne"),
                        "Categorical/string comparisons use equality only")
            elif spec.value_type == "boolean":
                require(type(value) is bool and node["operator"] in ("eq", "ne"),
                        "Boolean comparison invalid")
            else:
                raise ValueError("Structured measurements need explicit shared-witness logic")
        elif op == "exists_same_candidate":
            from .measurement_contracts import procedure_for
            selected = [spec_by_id[key] for key in node["feature_ids"]]
            require(all(s.measurement_method.value == "HYBRID" and
                        s.value_type == "structured" and s.structured_schema_name == "candidate-properties-v1"
                        for s in selected), "Shared witnesses require hybrid candidate-property rows")
            require(all({f.name: (f.value_type, f.nullable) for f in s.structured_fields} ==
                        {"candidate_id": ("string", False), "value": ("boolean", True)} for s in selected),
                    "Candidate witnesses need typed boolean-or-unknown values")
            programs = [procedure_for(s) for s in selected]
            require(len({(p.candidate_group, p.candidate_pattern) for p in programs}) == 1,
                    "Shared witnesses require the same objective candidate domain")
        elif op in ("and", "or"):
            for child in node["args"]:
                typed(child)
        elif op == "not":
            typed(node["arg"])
    typed(expression)
    require(referenced_features(expression) == set(binding.component_feature_ids),
            "Composite feature inventory mismatch")
    require(all(source.clause_id in s.original_clause_ids for s in specifications
                if s.feature_id in binding.component_feature_ids), "Feature lacks source grounding")
    if binding.resolution == "RESOLVED":
        require(bool(binding.component_feature_ids), "Nonempty source construct needs actual measurement")
        require(binding.logical_role != "BOUNDARY_UNRESOLVED", "Resolved boundary needs explicit polarity")
    if binding.resolution in ("UNRESOLVED", "NON_OBSERVABLE"):
        require(expression == {"op": "constant", "value": "UNKNOWN"},
                "Unresolved meaning must remain unknown")
    if binding.resolution == "NO_ADDITIONAL_BOUNDARY":
        require(source.role == "BOUNDARY_UNRESOLVED" and not source.original_text.strip(),
                "Only explicitly empty boundary prose permits no additional boundary")
        require(binding.logical_role == "BOUNDARY_EXCLUDES" and
                expression == {"op": "constant", "value": "FALSE"},
                "No-additional-boundary must not introduce applicability constraints")
    if any(s.is_proxy for s in specifications if s.feature_id in binding.component_feature_ids):
        require(binding.resolution != "RESOLVED", "A proxy alone cannot resolve the original construct")


def evaluate_expression(expression, observations):
    """Future code-only consumption of shared FeatureObservations; None means unknown."""
    require(len({o.identity for o in observations.values()}) <= 1, "Composite cannot join different cases")
    op = expression["op"]
    if op == "constant":
        return {"TRUE": True, "FALSE": False, "UNKNOWN": None}[expression["value"]]
    if op in ("and", "or"):
        values = [evaluate_expression(c, observations) for c in expression["args"]]
        if op == "and":
            return False if False in values else None if None in values else True
        return True if True in values else None if None in values else False
    if op == "not":
        value = evaluate_expression(expression["arg"], observations)
        return None if value is None else not value
    if op == "exists_same_candidate":
        maps, exhaustive = [], []
        for feature in expression["feature_ids"]:
            obs = observations.get(feature)
            if obs is None or obs.status.value != "KNOWN" or not isinstance(obs.extracted_value, StructuredValue):
                return None
            value = obs.extracted_value
            maps.append({dict((f.key, f.value) for f in row.fields)["candidate_id"]:
                         dict((f.key, f.value) for f in row.fields)["value"] for row in value.rows})
            exhaustive.append(dict((f.key, f.value) for f in value.fields).get("exhaustive") is True)
        witnesses = set().union(*(set(m) for m in maps))
        outcomes = []
        for key in witnesses:
            values = [m.get(key) for m in maps]
            outcomes.append(False if False in values else None if None in values else True)
        return True if True in outcomes else False if all(exhaustive) and None not in outcomes else None
    observation = observations.get(expression["feature_id"])
    if observation is None or observation.status.value != "KNOWN":
        return None
    value = observation.extracted_value
    if op == "feature":
        return value if type(value) is bool else None
    target = expression["value"]
    if type(value) is bool or type(target) is bool:
        if type(value) is not type(target):
            return None
    elif type(value) is not type(target) and not (
            type(value) in (int, float) and type(target) in (int, float)):
        return None
    operator = expression["operator"]
    if operator in ("gt", "ge", "lt", "le") and (
            type(value) not in (int, float) or type(target) not in (int, float)):
        return None
    return {"eq": lambda: value == target, "ne": lambda: value != target,
            "gt": lambda: value > target, "ge": lambda: value >= target,
            "lt": lambda: value < target, "le": lambda: value <= target}[operator]()
