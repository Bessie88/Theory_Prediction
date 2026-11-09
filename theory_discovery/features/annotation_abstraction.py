"""Project reviewed rules onto reusable, action-relevant annotation concepts."""
from copy import deepcopy
import re

from .feature_first import _atom, _batches, _deduplicate
from theory_discovery.core.schemas import require, stable_id

VERSION = "finalized-condition-action-pairs-v5"
FEATURE = '{"name":str,"anchor":str,"question":str,"target":"TEXT|SAME_TARGET","yes":str,"no":str,"ambiguity":str}'
TASK = (
    'Extract separate Condition and Action Features from the supplied finalized Theory pairs. '
    'Use only condition and action. Condition already includes any applicable limits. Keep one '
    'high-level Yes/No question per independently observable distinction, and use an exact short '
    'anchor from condition. Leave case names, specific years, quotations and wording examples '
    'out of annotation questions. Keep the logical scope of the condition. If no reusable '
    'condition can be safely expressed, or a required limit refers to unspecified content, '
    'return no condition_features and state omission_reason. '
    'Read edit direction, method, scope and target directly from action. Use concise labels '
    'and use unspecified where the text does not specify a value. Do not infer a new edit. '
    'Return {"pairs":[{"pair_id":str,"omission_reason":str,"condition_features":[' + FEATURE + '],'
    '"action_feature":{"direction":"deletion|insertion|replacement|no_change",'
    '"method":str,"scope":str,"target":str,"anchor":str}}]}.'
)
REPAIR = (
    ' Quote anchors exactly from their supplied Condition or Action. Keep Action and Condition distinct. '
    'Do not invent questions to fill a quota or details missing from the Action.'
)
CASE_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
CASE_QUOTE = re.compile(r"[\"“”]|(?<!\w)['‘][^'’]+['’](?!\w)")
ACTION_DIRECTIONS = {
    "delete": "deletion", "remove": "deletion", "insert": "insertion",
    "replace": "replacement", "change": "replacement", "keep": "no_change",
}


def _concept(raw, source):
    require(isinstance(raw, dict) and set(raw) ==
            {"name", "anchor", "question", "target", "yes", "no", "ambiguity"},
            "Expected an abstract feature and an exact source anchor")
    anchor = raw["anchor"]
    require(isinstance(anchor, str) and len(anchor.strip().split()) >= 2 and anchor.strip() in source,
            "Feature anchor must be an exact, substantive phrase from the supplied condition or boundary")
    atom = _atom({key: raw[key] for key in ("question", "target", "yes", "no", "ambiguity")})
    for key in ("question", "yes", "no", "ambiguity"):
        require(not CASE_YEAR.search(atom[key]) and not CASE_QUOTE.search(atom[key]),
                "Leave exact years and quoted wording in source evidence")
    require(len(atom["question"].split()) <= 18, "Use one short reusable Yes/No question")
    return anchor.strip(), atom


def action_feature(action):
    descriptor = dict(direction=action["family"], method=action["method"], scope=action["scope"],
                      target=action["target_feature"])
    return dict(descriptor, id=stable_id("action_feature", descriptor))


def _action(raw, source):
    fields = ("direction", "method", "scope", "target")
    require(isinstance(raw, dict) and set(raw) == {*fields, "anchor"}
            and all(isinstance(value, str) and value.strip() for value in raw.values()),
            "Action needs four descriptors and a source anchor")
    descriptor = {key: raw[key].strip() for key in fields}
    # The approved Action sentence itself is the source when its leading verb is explicit.
    direction = ACTION_DIRECTIONS.get(source.split(maxsplit=1)[0].casefold())
    if direction is not None:
        descriptor["direction"] = direction
    else:
        require(descriptor["direction"] in set(ACTION_DIRECTIONS.values())
                and raw["anchor"].strip() in source, "Action direction needs an exact source anchor")
    return dict(descriptor, id=stable_id("action_feature", descriptor))


def _literal_feature(raw, condition):
    if not isinstance(raw, dict):
        return False
    question = raw.get("question")
    if not isinstance(question, str):
        return False
    if set(CASE_YEAR.findall(question)) & set(CASE_YEAR.findall(condition)):
        return True
    source = condition.casefold().translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))
    query = question.casefold().translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))
    literals = [match.group(0) for match in CASE_QUOTE.finditer(source) if len(match.group(0)) > 2]
    literals.extend(re.findall(r'"[^"]+"', source))
    return any(literal in query for literal in literals)


def _review(raw, inputs):
    source = {r["pair_id"]: r for r in inputs}
    require(isinstance(raw, dict) and isinstance(raw.get("pairs"), list)
            and len(raw["pairs"]) == len(source)
            and {r["pair_id"] for r in raw["pairs"]} == set(source),
            "Return each finalized Condition/Action pair once")
    result = {}
    for row in raw["pairs"]:
        pid = row["pair_id"]
        original = source[pid]
        omitted = row.get("omission_reason")
        omitted = "" if omitted is None else omitted
        proposed = row.get("condition_features")
        require(isinstance(omitted, str) and isinstance(proposed, list),
                "Expected condition questions or an omission reason")
        if proposed and all(_literal_feature(feature, original["condition"]) for feature in proposed):
            proposed = []
            omitted = "The proposed questions depend on literal examples in the approved Condition."
        features, anchors = {}, {}
        for raw_feature in proposed:
            anchor, atom = _concept(raw_feature, original["condition"])
            fid = atom["id"]
            require(fid not in features and anchor not in anchors.values(),
                    "One Feature per distinct anchored Condition")
            features[fid] = atom
            anchors[fid] = anchor
        require(bool(features) != bool(omitted.strip()), "Return condition questions or an omission reason")
        result[pid] = dict(omission_reason=omitted.strip(), features=list(features.values()),
                           source_anchors=anchors, action=_action(row["action_feature"], original["action"]))
    return result


def theory_pairs(native):
    inputs, scopes = [], {}
    for branch in native["branches"]:
        bid = branch["branch_id"]
        refs = branch["source_rule_ids"]
        notes = branch.get("boundary_summaries")
        if notes is None:
            notes = [dict(summary=branch["boundary_text"], source_rule_ids=refs)] if branch.get("boundary_text") else []
        by_limits = {}
        for rid in refs:
            limits = tuple(dict.fromkeys(note["summary"].strip() for note in notes
                                        if rid in note["source_rule_ids"] and note["summary"].strip()))
            by_limits.setdefault(limits, []).append(rid)
        for index, (limits, scoped) in enumerate(by_limits.items()):
            pid = stable_id("theory_pair", [bid, index])
            # Each original scope receives its own complete condition.
            condition = " ".join([branch["situation"], *limits])
            inputs.append(dict(pair_id=pid, condition=condition, action=branch["interpretation"]))
            scopes[pid] = dict(branch_id=bid, source_rule_ids=scoped,
                               base_condition=branch["situation"], limits=list(limits))
    return inputs, scopes


CONDITION_VERSION = "theory-condition-finalization-v1"
CONDITION_TASK = (
    'Finalize each Theory Condition by integrating its already summarized scoped limits. Use only '
    'the provided base_condition, action, and limits. Keep the supplied Action unchanged. '
    'Merge repeated requirements into one concise, observable condition. Keep an independent limit '
    'only when it can change whether that Action applies. Leave exact names, dates, examples and '
    'quotations in Evidence when they do not independently change the Action. Do not replace a '
    'narrow case-specific requirement with a wider editing rule without support. If an essential '
    'limit is underspecified or the input does not explain the Action, return unresolved. '
    'For unresolved, leave condition empty and state why. Return '
    '{"pairs":[{"pair_id":str,"status":"ready|unresolved","condition":str,"reason":str}]}.'
)
CONDITION_REPAIR = (
    ' Return all supplied pair IDs. Use an empty condition for unresolved cases and explain the gap. '
    'Do not invent missing conditions.'
)


def _condition_decisions(raw, inputs):
    expected = {row["pair_id"] for row in inputs}
    require(isinstance(raw, dict) and isinstance(raw.get("pairs"), list)
            and len(raw["pairs"]) == len(expected)
            and {row["pair_id"] for row in raw["pairs"]} == expected,
            "Return a decision for each supplied scoped Condition")
    decisions = {}
    for row in raw["pairs"]:
        require(set(row) == {"pair_id", "status", "condition", "reason"}, "Unexpected Condition decision fields")
        condition, reason, status = row["condition"], row["reason"], row["status"]
        require(isinstance(condition, str) and isinstance(reason, str) and reason.strip(),
                "A Condition decision needs an explanation")
        require((status == "ready" and condition.strip()) or (status == "unresolved" and not condition.strip()),
                "Supply an integrated Condition or mark it unresolved")
        if status == "ready":
            require(not CASE_YEAR.search(condition) and not CASE_QUOTE.search(condition),
                    "Keep exact examples in Evidence")
        decisions[row["pair_id"]] = dict(status=status, condition=condition.strip(), reason=reason.strip())
    return decisions


def _condition_inputs(native):
    inputs, scopes = theory_pairs(native)
    review = [dict(pair_id=p["pair_id"], base_condition=scopes[p["pair_id"]]["base_condition"],
                   action=p["action"], limits=scopes[p["pair_id"]]["limits"])
              for p in inputs if scopes[p["pair_id"]]["limits"]]
    return inputs, scopes, review


def condition_review(native, decisions):
    inputs, scopes, review = _condition_inputs(native)
    validated = _condition_decisions({"pairs": [dict(v, pair_id=k) for k, v in decisions.items()]}, review)
    rows = []
    for pair in inputs:
        pid = pair["pair_id"]
        scope = scopes[pid]
        decision = validated.get(pid, dict(status="unchanged", condition=scope["base_condition"],
                                           reason="No recorded limit needs integration."))
        rows.append(dict(pair_id=pid, branch_id=scope["branch_id"], source_rule_ids=scope["source_rule_ids"],
                         base_condition=scope["base_condition"], previous_limits=scope["limits"],
                         action=pair["action"], **decision))
    return {"version": CONDITION_VERSION, "state": "pending_review",
            "theory_scope_hash": stable_id("theory_condition_scope", [inputs, scopes]),
            "decisions": validated, "pairs": rows}


def generate_condition_review(native, backend, cache, workers):
    _, _, inputs = _condition_inputs(native)
    batches = _batches(backend, cache, "theory_condition_finalization", CONDITION_TASK, inputs, 4,
                       _condition_decisions, workers, repair_message=CONDITION_REPAIR) if inputs else []
    return condition_review(native, {pid: row for batch in batches for pid, row in batch.items()})


def finalized_pairs(native, review):
    require(review.get("state") == "approved", "Approve the integrated Theory Conditions before feature extraction")
    expected = condition_review(native, review["decisions"])
    require(dict(review, state="pending_review") == expected, "Condition review no longer matches the saved Theory")
    scopes = {}
    inputs = []
    for row in review["pairs"]:
        if row["status"] == "unresolved":
            continue
        pid = row["pair_id"]
        inputs.append({k: row[k] for k in ("pair_id", "condition", "action")})
        scopes[pid] = {k: row[k] for k in ("branch_id", "source_rule_ids")}
    require(inputs, "No approved Condition/Action pair is available for annotation")
    return inputs, scopes


def abstract_annotation_theory(native, backend, cache, workers, review):
    inputs, scopes = finalized_pairs(native, review)
    batches = _batches(backend, cache, "annotation_theory_pairs", TASK, inputs, 4,
                       _review, workers, repair_message=REPAIR)
    reviewed = {pid: row for batch in batches for pid, row in batch.items()}
    catalog = {}
    aliases = _deduplicate(backend, cache, [a for row in reviewed.values() for a in row["features"]], catalog)
    plan = {"version": VERSION, "catalog": catalog, "branches": {}, "omissions": {}, "action_catalog": {}}
    for pair in inputs:
        pid = pair["pair_id"]
        row, scope = reviewed[pid], scopes[pid]
        aid = row["action"]["id"]
        plan["action_catalog"][aid] = row["action"]
        item = dict(pair, source_rule_ids=scope["source_rule_ids"], action_feature_id=aid)
        if row["omission_reason"]:
            plan["omissions"][pid] = dict(item, reason=row["omission_reason"])
            continue
        anchors = {}
        for fid, anchor in row["source_anchors"].items():
            anchors.setdefault(aliases[fid], []).append(anchor)
        item.update(feature_ids=list(anchors), source_anchors=anchors)
        plan["branches"].setdefault(scope["branch_id"], []).append(item)
    result = deepcopy(native)
    result["feature_plan"] = plan
    result["condition_finalization"] = review
    result["integration_metadata"]["annotation_feature_method"] = VERSION
    return result
