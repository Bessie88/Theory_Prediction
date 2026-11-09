"""Staged global integration with auditable source coverage."""
import logging
from pathlib import Path

from .boundary_candidates import propose_candidates
from .hierarchical_theory import OUTPUT_SCHEMA, _decode, _fit
from .llm import checked_infer
from .response_recovery import ModelOutputError, output_size_hint, remember_smaller_output
from .schemas import require
from .storage import write_json

LOGGER = logging.getLogger(__name__)
VERSION = "paired-rule-boundary-then-global-v2"
BOUNDARY_FIELDS = (
    '"boundary_summaries":[{"source_rule_ids":[str],"summary":str}],'
    '"boundary_detail_only_rule_ids":[str]'
)
BOUNDARY_POLICY = (
    'Summarize limits into broad, reusable distinctions that an annotator can judge from Before. '
    'Keep a distinction if it changes whether the rule applies. Consolidate equivalent limits '
    'into a single note with all and only the rule IDs to which it applies. A narrower exception '
    'keeps its narrower list of IDs. Omit individual names, quotations, examples and wording details '
    'unless essential to the decision. List a source with only illustrative boundary details in '
    'boundary_detail_only_rule_ids. For a source with no documented boundary, leave both fields '
    'free of that ID. Do not infer unrestricted applicability. Do not invent evidence. '
)
JOINT_TASK = (
    'Jointly summarize each editing Rule with its own Boundary. Group genuinely similar rules, '
    'including rules that keep the text unchanged. Explain the shared triggers and actions, '
    'preserving meaningful differences. ' + BOUNDARY_POLICY +
    'Use every source rule ID exactly once across the summaries. Keep the combined rule and boundary '
    'prose within the supplied average words per source. Return {"summaries":[{"source_rule_ids":[str],'
    '"rule_summary":str,' + BOUNDARY_FIELDS + '}]}.'
)
MERGE_TASK = (
    'Integrate the paired Rule and Boundary summaries into general editing principles. '
    'Combine equivalent conditions and actions across batches, and preserve independent variants. '
    'Write one short Condition in situation and one short Action explanation in interpretation. '
    'Avoid examples and lists of rule-specific facts. Do not output boundary summaries or Features. '
    'The program restores rule-specific boundaries from the supplied summaries. Use all supplied rule IDs '
    'and keep the existing Action directions. ' + OUTPUT_SCHEMA.replace('"boundary_text":str,', '')
)
COMPACT_MERGE_TASK = (
    'Be very concise. Combine genuinely equivalent propositions and keep one short sentence per prose field. '
    'Preserve complete rule coverage without repeating evidence or limits. ' + MERGE_TASK
)


def _refs(value, allowed, message):
    require(isinstance(value, list) and all(isinstance(rid, str) for rid in value), message)
    require(len(value) == len(set(value)) and set(value) <= allowed, message)
    return set(value)


def _boundaries(row, rule_ids, documented):
    documented = documented.intersection(rule_ids)
    summaries = row.get("boundary_summaries")
    require(isinstance(summaries, list), "Expected boundary summaries with scoped source IDs")
    details = _refs(row.get("boundary_detail_only_rule_ids"), rule_ids, "Invalid detail-only source IDs")
    # Absence is determined from the frozen source, even if the model labels it detail-only.
    details.intersection_update(documented)
    covered = set()
    notes, hints = [], []
    for note in summaries:
        require(isinstance(note, dict) and set(note) == {"source_rule_ids", "summary"},
                "Expected boundary prose and source IDs")
        ids = _refs(note["source_rule_ids"], rule_ids, "Boundary applied outside its source rules")
        require(ids and isinstance(note["summary"], str) and note["summary"].strip(),
                "A boundary note requires source IDs and prose")
        recorded = ids.intersection(documented)
        proposed = ids - documented
        if recorded:
            notes.append(dict(note, source_rule_ids=sorted(recorded)))
            covered.update(recorded)
        if proposed:
            hints.append(dict(note, source_rule_ids=sorted(proposed)))
    require(not covered.intersection(details) and covered | details == documented,
            "Each documented source boundary needs a scoped note or explicit detail-only status")
    return notes, sorted(details), hints


def _summaries(raw, allowed, documented, word_budget):
    require(isinstance(raw, dict) and isinstance(raw.get("summaries"), list),
            "Expected a complete summaries list")
    covered = set()
    words = 0
    normalized = []
    for row in raw["summaries"]:
        require(isinstance(row, dict), "Expected a joint Rule and Boundary summary")
        ids = _refs(row.get("source_rule_ids"), allowed, "Unknown or duplicate rule IDs in summary")
        require(ids and not covered.intersection(ids), "Summaries must use each supplied ID exactly once")
        text = row.get("rule_summary")
        require(isinstance(text, str) and text.strip(), "A rule summary needs prose")
        notes, details, hints = _boundaries(row, ids, documented)
        normalized.append(dict(row, boundary_summaries=notes, boundary_detail_only_rule_ids=details,
                               boundary_candidate_hints=hints))
        words += len(text.split()) + sum(len(note["summary"].split()) for note in notes)
        covered.update(ids)
    require(covered == allowed, "Summaries must cover every supplied source ID")
    require(words <= word_budget, "Summaries exceed the packet word budget")
    return normalized


def _summarize(backend, cache, stage, task, rows, words_per_rule):
    def packet(items):
        word_budget = words_per_rule * len(items)
        payload = {"sources": items, "average_words_per_source": words_per_rule}
        allowed = {row["rule_id"] for row in items}

        documented = {row["rule_id"] for row in items if row["boundary_text"].strip()}

        def decode(raw):
            return _summaries(raw, allowed, documented, word_budget)

        hint = output_size_hint(backend, stage, task, payload)
        if (hint is None or len(items) <= hint) and _fit(backend, task, payload):
            try:
                return cache.get_or_compute(stage, {
                    "version": VERSION, "task": task, "payload": payload,
                    "backend": backend.fingerprint,
                }, lambda: checked_infer(backend, task, payload, decode))
            except ModelOutputError as error:
                if len(items) == 1:
                    raise
                remember_smaller_output(backend, stage, task, payload, len(items), error)
        require(len(items) > 1, "One summary source exceeds the context budget")
        midpoint = len(items) // 2
        LOGGER.info("Splitting %s packet of %d sources", stage, len(items))
        return packet(items[:midpoint]) + packet(items[midpoint:])

    summaries = []
    for offset in range(0, len(rows), 24):
        summaries.extend(packet(rows[offset:offset + 24]))
    return summaries


def complete_staged_global_theory(backend, cache, allowed, source, *, output, save_progress, candidate_evidence=None):
    ordered = sorted((source[rid] for rid in allowed), key=lambda r: (r["group_id"], r["rule_id"]))
    pairs = [{"rule_id": r["rule_id"], "action": r["action"], "rule_text": r["natural_rule_text"],
              "boundary_text": r["boundary_text"]} for r in ordered]
    documented = {row["rule_id"] for row in pairs if row["boundary_text"].strip()}
    save_progress(stage="global_paired_summary")
    LOGGER.info("Joint Rule and Boundary summaries: %d source rules", len(pairs))
    summaries = _summarize(backend, cache, "hierarchical_global_paired_summaries", JOINT_TASK, pairs, 45)
    if output is not None:
        write_json(Path(output) / "global_paired_summaries.json", summaries)

    hints = [hint for row in summaries for hint in row.get("boundary_candidate_hints", [])]
    if candidate_evidence is not None:
        save_progress(stage="global_boundary_candidates")
    candidates = propose_candidates(backend, cache, pairs, hints, candidate_evidence)
    save_progress(stage="global_final_merge")
    LOGGER.info("Global Theory merge: %d paired summaries", len(summaries))
    payload = {"paired_rule_boundary_summaries": [
        {k: v for k, v in row.items() if k != "boundary_candidate_hints"} for row in summaries
    ]}
    source_notes = [note for row in summaries for note in row["boundary_summaries"]]
    detail_only = {rid for row in summaries for rid in row["boundary_detail_only_rule_ids"]}

    def decode(raw):
        result = _decode(raw, allowed)
        for row in result["branches"]:
            refs = set(row["source_rule_ids"])
            notes = []
            for note in source_notes:
                scoped = refs.intersection(note["source_rule_ids"])
                if scoped:
                    notes.append(dict(note, source_rule_ids=sorted(scoped)))
            row["boundary_summaries"] = notes
            row["boundary_detail_only_rule_ids"] = sorted(refs.intersection(detail_only))
            row["boundary_text"] = "\n".join(note["summary"] for note in notes)
        return result

    def merge():
        require(_fit(backend, MERGE_TASK, payload),
                "Complete Rule and Boundary summaries exceed final merge context; nothing was dropped")
        try:
            return checked_infer(backend, MERGE_TASK, payload, decode)
        except ModelOutputError:
            LOGGER.warning("Global merge output incomplete; retrying with concise prose and the same sources")
            require(_fit(backend, COMPACT_MERGE_TASK, payload),
                    "Concise merge exceeds context; no summaries were dropped")
            return checked_infer(backend, COMPACT_MERGE_TASK, payload, decode)

    result = cache.get_or_compute("hierarchical_staged_global_theory", {
        "version": "theory-then-condition-features-v3", "tasks": [MERGE_TASK, COMPACT_MERGE_TASK], "payload": payload,
        "backend": backend.fingerprint,
    }, merge)
    if candidate_evidence is not None:
        result["boundary_candidate_review"] = candidates
    return result
