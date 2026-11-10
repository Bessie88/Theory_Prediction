"""Audit prediction-rate gaps against independently reviewed Before conditions."""
from collections import defaultdict
from hashlib import sha256
from itertools import islice

from .condition_evolution import MIN_TEXTS_PER_PERIOD, complete, condition_identity, rate_difference
from theory_discovery.core.schemas import require, stable_id
from theory_discovery.annotation._support.text_diff import extract_diff

VERSION = "condition-shift-audit-v1"
BATCH_SIZE = 8
BEFORE_TASK = """Independently assess whether each frozen Condition holds in the supplied Before text.
Apply its complete Boolean expression and criteria. Use only Before, including for alternative Conditions.
Return YES only with an exact supporting Before quote. Use NO for a supported negative decision and
UNCERTAIN when evidence or required context is unavailable. A NO can have an empty quote.
Return {"rows":[{"id":str,"verdict":"YES|NO|UNCERTAIN","quote":str,"reason":str}]} for all supplied IDs."""
RELEVANCE_TASK = """Each listed Condition was independently supported by Before.
Assess whether its observable circumstance is relevant to any of the actual editing directions supplied.
Use Before, verified changes, and observed Action definitions. A relevant Condition can support several
editing choices. Do not assume what the theory predicted or infer intent. For NO_CHANGE, consider
whether the circumstance is relevant to preserving the original text. Use UNCERTAIN if evidence is unclear.
For YES, identify at least one supplied observed Action and give an exact Before quote.
Return {"rows":[{"id":str,"verdict":"YES|NO|UNCERTAIN","quote":str,"reason":str,"action_ids":[str]}]}."""


def _large(delta, threshold):
    return delta is not None and abs(delta) + 1e-9 >= threshold


def prediction_gaps(new, threshold=10):
    """Compare predictions and observations on exactly the same NEW texts."""
    require(type(threshold) in (int, float) and 0 < threshold <= 100, "Invalid audit threshold")
    groups = defaultdict(list)
    for row in new["records"]:
        if complete(row) and row.get("predicted_complete", True) and "OTHER" not in row["predicted_action_ids"]:
            groups[row["condition_id"]].append(row)
    gaps = []
    for condition, rows in sorted(groups.items()):
        n = len(rows)
        if n < MIN_TEXTS_PER_PERIOD:
            continue
        actions = sorted({a for row in rows for field in ("observed_action_ids", "predicted_action_ids") for a in row[field]})
        for action in actions:
            predicted = sum(action in r["predicted_action_ids"] for r in rows)
            observed = sum(action in r["observed_action_ids"] for r in rows)
            delta = (observed - predicted) * 100 / n
            if _large(delta, threshold):
                gaps.append(dict(condition_id=condition, action_id=action, samples=n,
                                 predicted_count=predicted, observed_count=observed,
                                 predicted_rate=predicted/n, observed_rate=observed/n,
                                 observed_minus_predicted_pp=delta))
    return gaps


def condition_catalog(compiled):
    def view(gate, clauses):
        op = gate["op"]
        if op == "clause":
            row = clauses[gate["clause_id"]]
            return dict(op=op, predicate=row["predicate_text"], scope=row["truth_scope"])
        if op == "not":
            return dict(op=op, arg=view(gate["arg"], clauses))
        require(op in ("and", "or"), "Audit needs executable frozen Boolean Conditions")
        return dict(op=op, args=[view(child, clauses) for child in gate["args"]])

    catalog = {}
    for owner in compiled["owners"]:
        if owner["method"] != "THEORY":
            continue
        for action in owner["actions"]:
            for variant in action["variants"]:
                gate = variant["gate"]
                key = condition_identity(gate, owner["clauses"])
                catalog.setdefault(key, view(gate, owner["clauses"]))
    return catalog


def decode_review(raw, ids, before, actions=None):
    require(isinstance(raw, dict) and set(raw) == {"rows"} and isinstance(raw["rows"], list),
            "Expected a review rows list")
    result = {}
    fields = {"id", "verdict", "quote", "reason"} | ({"action_ids"} if actions is not None else set())
    for row in raw["rows"]:
        require(isinstance(row, dict) and set(row) == fields, "Invalid review row fields")
        cid, verdict, quote = row["id"], row["verdict"], row["quote"]
        require(isinstance(cid, str) and cid in ids and cid not in result, "Unknown or duplicate Condition ID")
        require(verdict in ("YES", "NO", "UNCERTAIN") and isinstance(quote, str)
                and isinstance(row["reason"], str) and row["reason"].strip(), "Invalid review judgment")
        require((not quote or quote in before) and (verdict != "YES" or quote.strip()),
                "Review evidence must be an exact Before quote")
        if actions is not None:
            observed = row["action_ids"]
            require(isinstance(observed, list) and all(isinstance(a, str) and a in actions for a in observed)
                    and len(set(observed)) == len(observed) and (verdict != "YES" or observed),
                    "A relevant Condition must cite supplied actual Actions")
        result[cid] = {k:v for k,v in row.items() if k != "id"}
    require(set(result) == set(ids), "Review did not cover every requested Condition")
    return result


def _batches(ids):
    pending = iter(sorted(ids))
    while batch := tuple(islice(pending, BATCH_SIZE)):
        yield batch


def _review_case(outcome, action_ids, condition_ids, catalog, definitions, request, previous):
    todo = set(condition_ids) - previous.keys()
    if not todo:
        return
    if outcome is None or outcome.alignment_status != "ALIGNED" or outcome.after_text is None:
        previous.update({cid:dict(status="UNRESOLVED", reason="Missing aligned outcome")
                         for cid in todo})
        return
    before = outcome.before_text

    def review(task, batch, actions=None):
        # These short IDs are local to this request and cannot change the frozen identity.
        ids = {f"C{i}": cid for i,cid in enumerate(batch, 1)}
        payload = dict(before=before, conditions=[dict(id=key, expression=catalog[cid]) for key,cid in ids.items()])
        if actions is not None:
            payload.update(actual_actions=[dict(id=aid, **definitions[aid]) for aid in sorted(actions)],
                           observed_changes=changes, unchanged=before == outcome.after_text)
        try:
            answers = request(task, payload, lambda raw: decode_review(raw, ids, before, actions))
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
            if len(batch) > 1 and any(word in str(error).lower() for word in ("budget", "context", "output_length")):
                middle = len(batch)//2
                return {**review(task, batch[:middle], actions), **review(task, batch[middle:], actions)}
            return {cid:dict(verdict="ERROR", reason=type(error).__name__ + ": " + str(error)[:300])
                    for cid in batch}
        return {ids[key]:value for key,value in answers.items()}

    truth = {}
    for batch in _batches(todo):
        truth.update(review(BEFORE_TASK, batch))
    matched = {cid for cid,v in truth.items() if v["verdict"] == "YES"}
    relevant = {}
    if matched and action_ids:
        changes = [dict(operation=h.operation, before=h.before.quote, after=h.after.quote)
                   for h in extract_diff(before, outcome.after_text).changed_hunks]
        for batch in _batches(matched):
            relevant.update(review(RELEVANCE_TASK, batch, action_ids))
    elif matched:
        relevant = {cid:dict(verdict="UNCERTAIN", reason="Missing complete observed Actions") for cid in matched}
    for cid, first in truth.items():
        second = relevant.get(cid)
        verdict = second["verdict"] if second else first["verdict"]
        status = {"YES":"APPLICABLE", "NO":"REJECTED", "UNCERTAIN":"UNRESOLVED", "ERROR":"ERROR"}[verdict]
        previous[cid] = dict(status=status, before=first, relevance=second)


def audit_condition_shifts(old, new, catalog, outcomes, definitions, request, threshold=10):
    """Recount reviewed conditions and separate rejected, uncovered, and unresolved cases."""
    gaps = prediction_gaps(new, threshold)
    flagged = {g["condition_id"] for g in gaps}
    result = dict(protocol=VERSION, threshold_pp=threshold, minimum_texts_per_period=MIN_TEXTS_PER_PERIOD,
                  prediction_gaps=gaps, cases={}, reviewed_actual_rates=[], uncovered_case_ids=[],
                  old_membership_source="FROZEN_BEFORE_BASELINE", new_membership_source="INDEPENDENT_BEFORE_REVIEW",
                  reassigned_case_ids=[], unresolved_case_ids=[], retained_case_ids=[])
    if not flagged:
        return dict(result, state="NO_RATE_GAPS")

    rows_by_case = defaultdict(list)
    for row in new["records"]:
        rows_by_case[row["case_id"]].append(row)
    all_conditions = {row["condition_id"] for row in new["records"]}
    require(all_conditions == set(catalog), "Audit catalogue differs from frozen evaluation Conditions")
    by_id = {o.identity.case_id:o for o in outcomes}
    require(len(by_id) == len(outcomes) and set(by_id) <= set(rows_by_case), "Unexpected or duplicate NEW outcomes")

    actions_by_case, original = {}, {}
    for case_id, rows in rows_by_case.items():
        require(all(row["attribution_scope"] == "CASE_ASSOCIATION" for row in rows),
                "Audit requires case-level observed Action labels")
        if case_id in by_id:
            require(all(sha256(by_id[case_id].before_text.encode()).hexdigest() == row["before_hash"] for row in rows),
                    "Audit outcome Before differs from the frozen prediction")
        sets = {frozenset(row["observed_action_ids"]) for row in rows if row["observed_complete"]}
        require(len(sets) <= 1, "Observed Actions differ across Conditions for one case")
        action_ids = next(iter(sets), frozenset())
        if not all(row["observed_complete"] for row in rows) or "OTHER" in action_ids:
            action_ids = frozenset()
        require(set(action_ids) <= definitions.keys(), "Missing observed Action definition for the audit")
        actions_by_case[case_id] = action_ids
        original[case_id] = {row["condition_id"] for row in rows if row["condition"] is True} & flagged

    decisions = {case_id:{} for case_id in rows_by_case}
    def review(case_id, conditions):
        _review_case(by_id.get(case_id), actions_by_case[case_id], conditions, catalog, definitions,
                     request, decisions[case_id])

    # Re-measure flagged Conditions across the whole cohort, including original negatives.
    for case_id in sorted(decisions):
        review(case_id, flagged)
    recount = set(flagged)
    for case_id, states in sorted(decisions.items()):
        rejected = {cid for cid in original[case_id] if states[cid]["status"] == "REJECTED"}
        if not rejected:
            continue
        review(case_id, all_conditions)
        alternatives = {cid for cid,v in states.items() if v["status"] == "APPLICABLE"} - rejected
        if alternatives:
            result["reassigned_case_ids"].append(case_id)
            recount.update(alternatives)
        elif all(v["status"] == "REJECTED" for v in states.values()):
            result["uncovered_case_ids"].append(case_id)
        else:
            result["unresolved_case_ids"].append(case_id)
        result["cases"][case_id] = dict(rejected_original=sorted(rejected), alternatives=sorted(alternatives))

    # Alternatives need a whole-cohort denominator too.
    for case_id in sorted(decisions):
        review(case_id, recount)
    result["condition_reviews"] = decisions
    result["retained_case_ids"] = sorted(case_id for case_id, states in decisions.items()
        if any(states[cid]["status"] == "APPLICABLE" for cid in original[case_id]))
    result["unresolved_case_ids"] = sorted(set(result["unresolved_case_ids"]) | {
        case_id for case_id, states in decisions.items()
        if any(states[cid]["status"] in ("UNRESOLVED","ERROR") for cid in original[case_id])
    })
    old_groups = defaultdict(list)
    for row in old["records"]:
        if complete(row):
            old_groups[row["condition_id"]].append(set(row["observed_action_ids"]))
    for cid in sorted(recount):
        before = old_groups[cid]
        # Preserve a Before-only denominator. An Action-dependent exclusion would
        # make the older and newer cohorts measure different populations.
        after = [actions_by_case[case_id] for case_id, states in decisions.items()
                 if states[cid].get("before", {}).get("verdict") == "YES" and actions_by_case[case_id]]
        unrelated = sum(states[cid].get("before", {}).get("verdict") == "YES"
                        and states[cid]["status"] == "REJECTED" for states in decisions.values())
        pending = sum(states[cid]["status"] in ("ERROR","UNRESOLVED") for states in decisions.values())
        candidates = {g["action_id"] for g in gaps if g["condition_id"] == cid}
        if not candidates:
            candidates = set().union(*before, *after)
        for aid in sorted(candidates):
            a = sum(aid in actions for actions in before)
            b = sum(aid in actions for actions in after)
            rate = rate_difference(a, len(before), b, len(after))
            status = ("INCOMPLETE_AUDIT" if pending else "INSUFFICIENT_TEXTS" if rate["delta"] is None
                      else "MIXED_CONDITION_RELEVANCE" if unrelated
                      else "OBSERVED_SHIFT" if _large(rate["percentage_point_difference"], threshold)
                      else "NO_OBSERVED_SHIFT")
            result["reviewed_actual_rates"].append(dict(
                condition_id=cid, action_id=aid, old_success=a, new_success=b, old_n=len(before), new_n=len(after),
                pending_new_membership=pending, unrelated_new_actions=unrelated, status=status, **rate))
    return dict(result, state="AUDIT_RECORDED")


def audit_manifest(compiled, new, old, outcomes, definitions, model, threshold, induction):
    return dict(protocol=VERSION, condition_freeze=compiled["freeze_hash"],
                input_hash=stable_id("condition_shift_inputs", [old, new, outcomes.to_dict(), definitions]),
                review_model=model, threshold_pp=threshold, induction_config=induction)
