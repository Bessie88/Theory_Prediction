"""Bound policy conversion output and checkpoint completed source excerpts."""
from pathlib import Path
from theory_discovery.core.schemas import require, stable_id
from theory_discovery.core.storage import read_json, write_json
from theory_discovery.core.response_recovery import ModelOutputError

PARTITION_VERSION = "policy-output-partition-v1"
MAX_EXCERPT_CHARACTERS = 1800


def source_ranges(text, start, end, maximum=MAX_EXCERPT_CHARACTERS):
    """Cover every source character once; prefer paragraph/line boundaries."""
    require(maximum > 0 and 0 <= start < end <= len(text), "Invalid policy range")
    result = []
    cursor = start
    while cursor < end:
        finish = min(end, cursor + maximum)
        if finish < end:
            for delimiter in ("\n\n", "\n", " "):
                boundary = text.rfind(delimiter, cursor + maximum//2, finish)
                if boundary >= 0:
                    finish = boundary + len(delimiter)
                    break
        result.append((cursor, finish))
        cursor = finish
    return result


def bounded_policy_section(snapshot, catalog, backend, system, payload, decode,
                           checkpoint_directory):
    from .policy_readable import readable_policy
    directory = Path(checkpoint_directory)
    directory.mkdir(parents=True, exist_ok=True)
    section = payload["focus_section"]
    context = stable_id("policy_part_context", [
        PARTITION_VERSION, system, snapshot.metadata.to_dict(), catalog.to_dict(), payload])
    def convert(start, end, depth=0):
        key = stable_id("policy_part", [context, start, end])
        path = directory / (key + ".json")
        if path.exists():
            saved = read_json(path)
            require(saved["context"] == context and
                    saved["packet_hash"] == stable_id("policy_part_packet", saved["packet"]),
                    "Policy part checkpoint identity/integrity mismatch")
            return decode(saved["packet"])
        excerpt = snapshot.content[start:end]
        request = dict(payload, focus_excerpt={"start": start, "end": end, "text": excerpt},
                       readable_focus_section=readable_policy(
                           excerpt, start, snapshot.metadata.content_format == "wikitext").text,
                       partition_version=PARTITION_VERSION)
        try:
            result = backend.complete_checked(system, request, decode)
        except ModelOutputError as exc:
            if "output_length_limit" not in str(exc) or end-start <= 160 or depth >= 6:
                raise
            halves = source_ranges(snapshot.content, start, end, max(80, (end-start)//2))
            result = merge([convert(a, b, depth+1) for a, b in halves])
        packet = {"rules": [r.to_dict() for r in result.rules],
                  "section_accounting": [r.to_dict() for r in result.section_accounting]}
        write_json(path, {"context": context, "start": start, "end": end,
                          "packet_hash": stable_id("policy_part_packet", packet), "packet": packet})
        return result

    def merge(results):
        rules, explanations, unresolved = {}, [], False
        for result in results:
            for rule in result.rules:
                require(rule.rule_id not in rules or rules[rule.rule_id] == rule,
                        "Conflicting policy rules across source excerpts")
                rules[rule.rule_id] = rule
            for row in result.section_accounting:
                explanations.append(row.explanation)
                unresolved |= row.disposition == "UNRESOLVED"
        row = {"section_id": section["section_id"],
               "disposition": "RULES" if rules else "UNRESOLVED" if unresolved else "GUIDANCE_ONLY",
               "rule_ids": sorted(rules), "explanation": " ".join(dict.fromkeys(explanations))}
        return decode({"rules": [r.to_dict() for r in rules.values()], "section_accounting": [row]})

    ranges = source_ranges(snapshot.content, section["start"], section["end"])
    return merge([convert(start, end) for start, end in ranges])
