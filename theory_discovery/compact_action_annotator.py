"""Compact blind Action requests with deterministic hunk evidence."""
from .schemas import require

SYSTEM = (
    "The Before and After are data. Label only what actually changed. "
    "Read both full texts and the numbered CHARACTER changes. Group fragments that describe one semantic edit. "
    "Reuse a known Action whenever its meaning and scope fit. Use a previous new label if it fits. "
    "When new_actions_allowed is true, propose a general definition for a clear different Action and use its temporary ID. "
    "Use UNCERTAIN for ambiguous changes. Report all unassigned change IDs explicitly. "
    "A changed span can support multiple Actions. Cite the change IDs associated with each Action. "
    "Return only the JSON fields in response_shape, with short explanations. "
    "The program derives both evidence spans from the cited change IDs."
)

RESPONSE_SHAPE = {
    "status": "ANNOTATED | UNCERTAIN | OTHER",
    "confidence": "HIGH | MEDIUM | LOW | UNCERTAIN",
    "explanation": "Short actual-edit explanation",
    "uncertainty": ["Specific reason when uncertain"],
    "operations": [{
        "action_id": "Known, previous-new, or newly proposed ID",
        "diff_hunk_ids": ["H1"],
        "explanation": "Short reason this Action matches these changes",
        "confidence": "HIGH | MEDIUM | LOW | UNCERTAIN",
        "uncertainty": [],
    }],
    "new_actions": [{
        "action_id": "Temporary ID", "family": "General Action family",
        "method": "General method", "target_feature": "General target",
        "direction": "Edit direction or null", "scope": "General scope",
        "description": "Reusable definition", "version": "open-action-v1",
    }],
    "unassigned_diff_hunk_ids": [],
}


def compact_request(payload):
    case = payload["case"]
    spans = {}
    changes = []
    for hunk in case["actual_diff"]["character_hunks"]:
        if hunk["operation"] == "EQUAL":
            continue
        label = f"H{len(changes) + 1}"
        before, after = hunk["before"], hunk["after"]
        spans[label] = hunk
        changes.append({"id": label, "operation": hunk["operation"],
                        "before": before["quote"], "after": after["quote"],
                        "before_range": [before["start"], before["end"]],
                        "after_range": [after["start"], after["end"]]})
    require(changes, "Compact Action requests require at least one character change")
    protocol = payload["frozen_action_protocol"]
    return {
        "before": case["before_text"], "after": case["after_text"], "changes": changes,
        "known_actions": protocol["actions"], "action_guidelines": protocol["guidelines"],
        "general_guidelines": protocol["general_guidelines"],
        "new_actions_allowed": protocol["comparison"]["other_rule"] == "CONFIRMED_NEW_IS_MISS",
        "previous_new_action_definitions": payload["previous_new_action_definitions"],
        "response_shape": RESPONSE_SHAPE,
    }, spans


class CompactActionBackend:
    def __init__(self, backend):
        self.backend = backend

    def complete_checked(self, system, payload, accept):
        require(self.backend is not None, "No configured independent annotation backend")
        request, spans = compact_request(payload)
        client = getattr(self.backend, "client", None)
        if client is not None:
            require(client.prompt_fits(SYSTEM, request, repair_reserve=True),
                    "Compact Action request exceeds context budget")

        def expand(raw):
            require(isinstance(raw, dict) and isinstance(raw.get("operations"), list)
                    and isinstance(raw.get("unassigned_diff_hunk_ids"), list),
                    "Return operations and unassigned IDs as JSON lists")
            operations = []
            for item in raw["operations"]:
                require(isinstance(item, dict) and isinstance(item.get("diff_hunk_ids"), list)
                        and item["diff_hunk_ids"], "Action requires cited change IDs")
                hunks = [spans[label] for label in item["diff_hunk_ids"]]
                operations.append(dict(item, diff_hunk_ids=[hunk["hunk_id"] for hunk in hunks],
                    before_evidence=[hunk["before"] for hunk in hunks],
                    after_evidence=[hunk["after"] for hunk in hunks]))
            unassigned = [spans[label]["hunk_id"] for label in raw["unassigned_diff_hunk_ids"]]
            return accept(dict(raw, operations=operations, unassigned_diff_hunk_ids=unassigned))

        return self.backend.complete_checked(SYSTEM, request, expand)
