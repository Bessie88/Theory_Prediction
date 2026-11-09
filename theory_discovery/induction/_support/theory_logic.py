"""Finite source-clause logic; no new interpretation or applicability model."""
import json
from theory_discovery.core.application_contracts import LogicNodeTrace
from theory_discovery.core.schemas import require

LOGIC_VERSION = "three-valued-source-clauses-v1"


def truth_and(values):
    return False if False in values else None if None in values else True


def truth_or(values):
    return True if True in values else None if None in values else False


def validate_logic(expression, bindings, depth=0):
    require(isinstance(expression, dict) and depth <= 24, "Invalid/deep application expression")
    op = expression.get("op")
    if op == "constant":
        require(set(expression) == {"op", "value"} and
                expression["value"] in ("TRUE", "FALSE", "UNKNOWN"), "Invalid truth constant")
    elif op in ("clause", "boundary_clear"):
        require(set(expression) == {"op", "clause_id"} and expression["clause_id"] in bindings,
                "Unknown original measurement clause")
        role = bindings[expression["clause_id"]].logical_role
        if op == "boundary_clear":
            require(role == "BOUNDARY_EXCLUDES", "Only exclusion boundaries use boundary_clear")
        else:
            require(role in ("CONDITION", "BOUNDARY_REQUIRES", "BOUNDARY_RESTRICTS",
                             "BOUNDARY_UNRESOLVED"), "Exclusion polarity cannot be silently reversed")
    elif op in ("and", "or"):
        require(set(expression) == {"op", "args"} and isinstance(expression["args"], list) and
                expression["args"], "Composite source logic needs operands")
        for child in expression["args"]:
            validate_logic(child, bindings, depth + 1)
    elif op == "not":
        require(set(expression) == {"op", "arg"}, "Invalid source NOT")
        validate_logic(expression["arg"], bindings, depth + 1)
    else:
        raise ValueError("Unsupported source logic operator")


def clause_references(expression):
    op = expression["op"]
    if op in ("clause", "boundary_clear"):
        return {expression["clause_id"]}
    if op in ("and", "or"):
        return set().union(*(clause_references(c) for c in expression["args"]))
    if op == "not":
        return clause_references(expression["arg"])
    return set()


def evaluate_logic(expression, clause_values, path="$"):
    op = expression["op"]
    nodes = []
    clause_id = None
    if op == "constant":
        value = {"TRUE": True, "FALSE": False, "UNKNOWN": None}[expression["value"]]
    elif op in ("clause", "boundary_clear"):
        clause_id = expression["clause_id"]
        value = clause_values.get(clause_id)
        if op == "boundary_clear" and value is not None:
            value = not value
    elif op in ("and", "or"):
        values = []
        for index, child in enumerate(expression["args"]):
            value, trace = evaluate_logic(child, clause_values, path + "." + str(index))
            values.append(value)
            nodes.extend(trace)
        value = truth_and(values) if op == "and" else truth_or(values)
    elif op == "not":
        value, trace = evaluate_logic(expression["arg"], clause_values, path + ".not")
        nodes.extend(trace)
        value = None if value is None else not value
    else:
        raise ValueError("Unknown source logic operator")
    return value, (LogicNodeTrace(path, op, value, clause_id), *nodes)
