"""Natural-language discovery; executable rule conversion is deliberately absent."""
from collections import defaultdict, deque
import logging
import re
import time
from .action_grouping import group_actions
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION, checked_infer
from theory_discovery.core.schemas import ActionGroup, require, stable_id, to_jsonable
from .semantic_alignment import SemanticActionResolver
from .semantic_rules import INDUCTION_TASK, SIMILAR_CONTEXT_TASK, code_packet, _call as old_call
from .predictive_theory import _sample
from .natural_theory_validation import action_options_for_refs, selected_frozen_actions
from .scoped_conditions import assess_conditions
from theory_discovery.core.storage import Cache, EvidenceLedger, write_json
from .cross_group_policy import (
    VERSION as CROSS_GROUP_VERSION, BOUNDARY_PROMPT, RELATION_GUIDANCE, BoundaryCaseRoleError, BoundaryQuoteError,
    comparison_case_roles, output_shape, validate_comparison, synthesis_relations,
)
from theory_discovery.core.response_recovery import (ModelOutputError, recoverable_cached_call,
    output_size_hint, remember_smaller_output)

VERSION = 'natural-language-rules-boundaries-theory-v1'
MODE = 'natural_language_predictive_theory'
GROUPING_POLICY = 'incremental_approximate_method_type_v3_inherited_provenance'
LOGGER = logging.getLogger(__name__)
BOUNDARY_TASK = BOUNDARY_PROMPT + '\n' + PLAIN_LANGUAGE_INSTRUCTION
THEORY_TASK = RELATION_GUIDANCE + '\n' + """Summarize supplied rules into useful predictive statements. Keep each source rule's condition, boundary and action independent and traceable. Multiple theories may apply to one case; do not classify their action relationships. Preserve uncertainty; invent no facts, motives, conditions or actions.
Return {"branches":[{"situation":str,"source_rule_ids":[str],"interpretation":str,"contrast_rule_ids":[str],"supporting_case_ids":[str],"contradicting_case_ids":[str],"uncertainty":[str]}],"excluded_rules":[{"rule_id":str,"reason":str}],"open_questions":[str]}. Cover every useful rule or explicitly exclude it with a reason. Cite only supplied rule and case IDs.""" + '\n' + PLAIN_LANGUAGE_INSTRUCTION
PREDICTION_TASK = """Using only Before and the supplied frozen natural-language theory situations, describe the editing direction or supported combination justified by the rule texts, boundaries and uncertainty. Match approximate meanings. Do not invent a branch or action, infer unseen sources, or claim knowledge of the actual edit. Read all choices; frequency alone is not a selection rule. Abstain if transfer is unsupported.
Return {"abstain":bool,"branch_id":str|null,"source_rule_ids":[str],"strategy":str,"reason":str}. Identify a supplied theory situation and the source rule(s) supporting your strategy. Sources may span several frozen editing directions; a supported combined direction is allowed. Do not force one action or one source when the evidence supports a combination. Describe the concrete direction briefly, using only Before and the frozen theory; do not provide a full rewrite or invent facts. Or abstain with null, an empty source list and an empty strategy.""" + '\n' + PLAIN_LANGUAGE_INSTRUCTION


def strings(value, message, unique=True):
    require(isinstance(value, list) and all(isinstance(x, str) for x in value), message)
    require(not unique or len(value) == len(set(value)), message + ': duplicate IDs')
    return value


def fits(backend, task, payload):
    checker = getattr(backend, 'prompt_fits', None)
    return checker is None or checker(task, payload, repair_reserve=True)


def cached_call(backend, cache, stage, task, payload, decode):
    return recoverable_cached_call(backend, cache, stage, VERSION, task, payload, decode)


def citation_stats(all_ids, compared, support, contradictions):
    all_ids, compared = set(all_ids), set(compared)
    support, contradictions = set(support), set(contradictions)
    require(compared <= all_ids and (support | contradictions) <= compared, 'Invented comparison evidence')
    contested = support & contradictions
    support -= contested
    contradictions -= contested
    unclassified = compared - support - contradictions - contested
    return {'supporting_case_ids': sorted(support), 'contradicting_case_ids': sorted(contradictions),
        'contested_case_ids': sorted(contested), 'unclassified_case_ids': sorted(unclassified),
        'compared_case_ids': sorted(compared), 'not_compared_case_ids': sorted(all_ids - compared),
        'support_count': len(support), 'contradiction_count': len(contradictions),
        'eligible_cases': len(compared), 'conditional_action_frequency': None, 'coverage': None,
        'counting_basis': 'model_cited_evidence_with_recorded_comparison_scope',
        'unclassified_are_not_unknown_or_counterexamples': True}


def induce(observations, groups, conditions, config, backend, cache, group_filter=None):
    context = stable_id('cohort', {'observations': to_jsonable(observations), 'groups': to_jsonable(groups)})
    comparisons, rules = [], []
    all_ids = {o.case_id for o in observations}
    for group in groups:
        if group_filter is not None and group.group_id not in group_filter: continue
        members = [o for o in observations if o.case_id in group.member_case_ids]
        offset, local, packets = 0, {}, []
        while offset < len(members):
            size = len(members) - offset
            while True:
                chunk = members[offset:offset + size]
                payload = {**code_packet(chunk, conditions), 'group_id': group.group_id,
                    'action': group.normalized_action, 'member_case_ids': [o.case_id for o in chunk],
                    'full_group_size': len(members),
                    'condition_scope': 'Per-case observed features and locally assessed values; omitted conditions were not assessed on that case.'}
                if fits(backend, INDUCTION_TASK, payload): break
                require(size > 1, 'One group case exceeds context; no evidence was truncated')
                size = max(1, size // 2)
            shown = {o.case_id for o in chunk}
            def decode(raw):
                require(isinstance(raw, dict) and set(raw) == {'rules', 'comparison_memo', 'unresolved_questions'}
                        and isinstance(raw['rules'], list), 'Invalid natural-language rule response')
                require(isinstance(raw['comparison_memo'], str) and raw['comparison_memo'].strip(), 'Missing comparison memo')
                strings(raw['unresolved_questions'], 'Invalid rule questions', unique=False)
                for row in raw['rules']:
                    require(set(row) == {'rule_text', 'rationale', 'supporting_case_ids', 'contradicting_case_ids'},
                            'Natural rule must contain prose and evidence, not executable conditions')
                    require(all(isinstance(row[k], str) and row[k].strip() for k in ('rule_text', 'rationale')), 'Empty rule text or rationale')
                    for key in ('supporting_case_ids', 'contradicting_case_ids'):
                        require(set(strings(row[key], 'Invalid natural rule citations')) <= shown, 'Unknown within-group citation')
                    require(bool(row['supporting_case_ids']), 'A rule needs an actual within-group support')
                    require(not set(row['supporting_case_ids']) & set(row['contradicting_case_ids']), 'Conflicting rule citations')
                return raw
            LOGGER.info('Natural rule induction: %s, members %d-%d/%d; no outside cases',
                        group.group_id, offset + 1, offset + size, len(members))
            # Preserve the existing natural induction identity. Completed prose is reusable.
            result = old_call(backend, cache, 'natural_rule_induction', INDUCTION_TASK, payload, decode, context)
            # Cached results must pass the current contract too.
            result = decode(result)
            for row in result['rules']:
                rid = stable_id('rule', [VERSION, group.group_id, row['rule_text']])
                rule = local.setdefault(rid, {'rule_id': rid, 'natural_rule_text': row['rule_text'],
                    'induction_rationale': row['rationale'],
                    'action': dict(group.normalized_action, group_id=group.group_id), 'group_id': group.group_id,
                    'boundary_text': '', 'supporting_case_ids': [], 'contradicting_case_ids': [],
                    'induction_supporting_case_ids': [], 'induction_contradicting_case_ids': [],
                    'status': 'candidate', 'representation': 'natural_language'})
                for key in ('supporting_case_ids', 'contradicting_case_ids'):
                    rule['induction_' + key] = sorted(set(rule['induction_' + key]) | set(row[key]))
            packets.append({'member_case_ids': sorted(shown), 'comparison_memo': result['comparison_memo'],
                            'unresolved_questions': result['unresolved_questions']})
            offset += size
        for rule in local.values():
            stats = citation_stats(all_ids, group.member_case_ids, rule['induction_supporting_case_ids'],
                                   rule['induction_contradicting_case_ids'])
            rule['evidence_statistics'] = stats
            rule['supporting_case_ids'] = stats['supporting_case_ids']
            rule['contradicting_case_ids'] = stats['contradicting_case_ids']
            rule['construction_case_ids'] = stats['compared_case_ids']
        comparisons.append({'group_id': group.group_id, 'representation': 'natural_language',
            'natural_rules': list(local.values()), 'comparison_memos': packets,
            'open_code_case_ids_compared': sorted(group.member_case_ids)})
        rules.extend(local.values())
    return comparisons, rules


def retrieve(rule, observations, groups, conditions, backend, cache):
    memberships = defaultdict(list)
    for group in groups:
        if group.group_id != rule['group_id']:
            for cid in group.member_case_ids: memberships[cid].append(group.group_id)
    outside = [o for o in observations if o.case_id in memberships]
    index = {c.condition_id: c for c in conditions}
    selected, packets, offset = [], [], 0
    own = set(next(g for g in groups if g.group_id == rule['group_id']).member_case_ids)
    while offset < len(outside):
        size = len(outside) - offset
        while True:
            chunk = outside[offset:offset + size]
            used = {cid for o in chunk for cid in o.candidate_conditions}
            payload = {'target_context': {'rule_text': rule['natural_rule_text'],
                        'condition_description': rule['natural_rule_text']},
                'condition_definitions': [{'condition_id': cid, 'definition': index[cid].definition,
                        'value_type': index[cid].value_type} for cid in sorted(used)],
                'outside_condition_profiles': [{'case_id': o.case_id, 'before_text': o.before_text,
                    'other_group_ids': memberships[o.case_id], 'also_in_target_group': o.case_id in own,
                    'candidate_condition_ids': o.candidate_conditions, 'observed_conditions': o.condition_values} for o in chunk],
                'observable_context_policy': 'Before-only text supplements incomplete original condition proposals; do not infer an edit or use After.'}
            if fits(backend, SIMILAR_CONTEXT_TASK, payload): break
            require(size > 1, 'One outside Before/profile exceeds context; no evidence was truncated')
            size = max(1, size // 2)
        shown = {o.case_id for o in chunk}
        def decode(raw):
            require(isinstance(raw, dict) and 'similar_case_ids' in raw, 'Missing outside case selections')
            ids = raw['similar_case_ids']
            require(isinstance(ids, list) and all(isinstance(cid, str) for cid in ids), 'Invalid selected case IDs')
            require(set(ids) <= shown, 'Invented outside match')
            reason = raw.get('reason')
            if not isinstance(reason, str) or not reason.strip():
                reason = 'The model supplied a complete matched-case list without a separate explanation.'
            return {'similar_case_ids': list(dict.fromkeys(ids)), 'reason': reason}
        result = cached_call(backend, cache, 'natural_similar_outside', SIMILAR_CONTEXT_TASK, payload, decode)
        wanted = set(result['similar_case_ids'])
        selected.extend(o for o in chunk if o.case_id in wanted)
        packets.append({'considered_case_ids': sorted(shown), **result})
        offset += size
    return selected, {'selected_case_ids': [o.case_id for o in selected],
        'outside_condition_profiles_considered': len(outside), 'context_match_decisions': packets,
        'outside_group_ids_by_case': {o.case_id: memberships[o.case_id] for o in selected},
        'shared_with_target_case_ids': [o.case_id for o in selected if o.case_id in own],
        'fixed_case_limit': None, 'match_policy': 'all_approximately_similar_condition_types_before_only'}


def compare_boundary_packet(backend, cache, payload, decode):
    stage = "natural_boundary_packets"
    citation_errors = (BoundaryCaseRoleError, BoundaryQuoteError)
    try:
        return cached_call(backend, cache, stage, BOUNDARY_TASK, payload, decode)
    except ValueError as error:
        if not isinstance(error, citation_errors) and not isinstance(error.__cause__, citation_errors):
            raise

    _, own, outside = comparison_case_roles(payload)
    source_spans = {}
    case_spans = []
    for item in payload["before_evidence"]:
        case_id = item["case_id"]
        roles = [side for side, ids in (("own", own), ("outside", outside)) if case_id in ids]
        spans = []
        for piece in re.split(r"(?<=[.!?])\s+|\n+", item["before_text"]):
            quote = piece.strip()
            if not quote:
                continue
            span_id = "S" + str(len(source_spans) + 1)
            source_spans[span_id] = (case_id, quote, roles)
            spans.append(dict(span_id=span_id, before_text=quote))
        case_spans.append(dict(case_id=case_id, allowed_roles=roles, spans=spans))

    shape = output_shape()
    distinction = shape["distinctions"][0]
    for side in ("own", "outside"):
        distinction.pop(side + "_case_id")
        distinction.pop(side + "_quote")
        distinction[side + "_span_id"] = "supplied Before span ID"
    constrained = dict(payload, before_evidence=case_spans, output_shape=shape)
    task = BOUNDARY_TASK.replace(
        "Give exact Before quotes from both sides and explain why each distinction matters.",
        "For each new distinction choose one supplied own_span_id and one outside_span_id from "
        "different cases with the indicated allowed_roles. Python attaches their exact Before text. "
        "Explain why the distinction matters. Use only the supplied span IDs; do not write quote text. "
        "Use existing distinction IDs in boundary clauses when retaining prior supported limits. "
        "When no new distinction is supported, return an empty distinctions list.",
    )

    def decode_spans(raw):
        require(isinstance(raw, dict) and isinstance(raw.get("distinctions"), list), "Expected distinctions")
        rows = []
        for source in raw["distinctions"]:
            require(isinstance(source, dict) and set(source) == set(distinction),
                    "Return distinction fields with supplied span IDs")
            row = dict(source)
            for side in ("own", "outside"):
                span_id = row.pop(side + "_span_id")
                require(isinstance(span_id, str) and span_id in source_spans,
                        "Use an existing Before span ID")
                case_id, quote, roles = source_spans[span_id]
                require(side in roles, "Use a Before span from the correct comparison side")
                row[side + "_case_id"] = case_id
                row[side + "_quote"] = quote
            rows.append(row)
        return decode(dict(raw, distinctions=rows))

    if not fits(backend, task, constrained):
        raise ModelOutputError("Span-referenced boundary request needs a smaller evidence window")
    LOGGER.warning("Retrying boundary %s with exact Before span references", payload["rule"]["rule_id"])
    try:
        return cached_call(backend, cache, stage, task, constrained, decode_spans)
    except ValueError as error:
        raise ModelOutputError("Span-referenced boundary response did not pass validation") from error


def add_boundaries(rules, observations, groups, conditions, backend, cache):
    index = {o.case_id: o for o in observations};all_ids = set(index);boundaries = []
    for rule in rules:
        historical = rule.pop("cross_group_comparisons", None)
        if historical:rule["historical_cross_group_comparisons"]=historical
        selected, retrieval = retrieve(rule, observations, groups, conditions, backend, cache)
        # All supports cited during full within-group induction are retained, without a sample cap.
        supports = [index[cid] for cid in rule['induction_supporting_case_ids']]
        pairs = [(supports[i % len(supports)], selected[i % len(selected)])
                 for i in range(max(len(supports), len(selected)))] if selected else []
        offset, state, packets, support_cites, contra_cites = 0, '', [], set(rule['induction_supporting_case_ids']), set(rule.get('induction_contradicting_case_ids', []))
        seen_support, seen_outside = set(), set()
        distinction_state, clause_state = [], []
        distinction_history = {}
        while offset < len(pairs):
            size = len(pairs) - offset
            while True:
                window = pairs[offset:offset + size]
                own_chunk = list({a.case_id: a for a, b in window}.values())
                out_chunk = list({b.case_id: b for a, b in window}.values())
                cases = list({o.case_id: o for o in own_chunk + out_chunk}.values())
                payload = {**code_packet(cases, conditions),
                    'rule': {'rule_id': rule['rule_id'], 'rule_text': rule['natural_rule_text'], 'action': rule['action']},
                    'cross_group_policy': CROSS_GROUP_VERSION,
                    'previous_boundary_text': state,
                    'previous_distinctions': distinction_state, 'previous_boundary_clauses': clause_state,
                    'all_own_support_case_ids': [o.case_id for o in supports],
                    'all_matched_outside_case_ids': [o.case_id for o in selected],
                    'before_evidence': [{'case_id': o.case_id, 'before_text': o.before_text} for o in cases],
                    'other_actions_by_group': {g.group_id: g.normalized_action for g in groups if any(g.group_id in retrieval['outside_group_ids_by_case'][o.case_id] for o in out_chunk)},
                    'output_shape': output_shape(),
                    'previously_compared_case_ids': sorted(seen_support | seen_outside),
                    'own_support_case_ids': [o.case_id for o in own_chunk],
                    'matched_outside_case_ids': [o.case_id for o in out_chunk],
                    'outside_group_ids_by_case': {o.case_id: retrieval['outside_group_ids_by_case'][o.case_id] for o in out_chunk}}
                hint = output_size_hint(backend, 'natural_boundary_packets', BOUNDARY_TASK, payload)
                if hint is not None and hint < size:
                    size = hint
                    continue
                if fits(backend, BOUNDARY_TASK, payload): break
                require(size > 1, 'One natural boundary comparison exceeds context; no evidence was truncated')
                size = max(1, size // 2)
            shown = {o.case_id for o in cases} | seen_support | seen_outside
            def decode(raw):
                return validate_comparison(raw, payload, index)
            LOGGER.info('Natural boundary comparison: %s, %d own supports, %d outside cases; window %d-%d/%d',
                        rule['rule_id'], len(own_chunk), len(out_chunk), offset + 1, offset + size, len(pairs))
            try:
                result = compare_boundary_packet(backend, cache, payload, decode)
            except ModelOutputError as exc:
                if size <= 1: raise
                remember_smaller_output(backend, 'natural_boundary_packets', BOUNDARY_TASK, payload, size, exc)
                continue
            record = {'supporting_case_ids': payload['own_support_case_ids'],
                'outside_case_ids': payload['matched_outside_case_ids'], 'previous_boundary_text': state, **result}
            # Keep coverage fields separate from the model's evidence citations.
            record['own_support_case_ids'] = payload['own_support_case_ids']
            record['matched_outside_case_ids'] = payload['matched_outside_case_ids']
            packets.append(record);state = result['boundary_text']
            carried = {d['id']: d for d in distinction_state}
            carried.update({d['id']: d for d in result['distinctions']})
            distinction_history.update(carried)
            active = {k for c in result['boundary_clauses'] for k in c['distinction_ids']}
            distinction_state = [d for k,d in carried.items() if k in active];clause_state = result['boundary_clauses']
            support_cites.update(result['supporting_case_ids']);contra_cites.update(result['contradicting_case_ids'])
            seen_support.update(payload['own_support_case_ids']);seen_outside.update(payload['matched_outside_case_ids'])
            offset += size
        require(seen_outside == {o.case_id for o in selected}, 'Boundary missed matched outside cases')
        require(not selected or seen_support == {o.case_id for o in supports}, 'Boundary sampled own supports')
        # Historical boundary citations/coverage are not evidence for the new policy.
        compared = set(next(g for g in groups if g.group_id == rule['group_id']).member_case_ids) | seen_outside
        stats = citation_stats(all_ids, compared, support_cites, contra_cites)
        rule.update(cross_group_policy=CROSS_GROUP_VERSION, boundary_distinctions=list(distinction_history.values()), boundary_clauses=clause_state,
                    boundary_text=state, evidence_statistics=stats, construction_case_ids=stats['compared_case_ids'],
                    supporting_case_ids=stats['supporting_case_ids'], contradicting_case_ids=stats['contradicting_case_ids'])
        boundaries.append({'rule_id': rule['rule_id'], 'group_id': rule['group_id'], 'natural_rule_text': rule['natural_rule_text'],
            'boundary_text': state, 'context_retrieval': retrieval, 'cross_group_policy': CROSS_GROUP_VERSION,
            'boundary_distinctions': list(distinction_history.values()), 'boundary_clauses': clause_state,
            'available_support_case_ids': [o.case_id for o in supports],
            'boundary_comparison_packets': packets, 'boundary_support_sampling': False,
            'evidence_statistics': stats, 'evidence_count_note': 'Citations across recorded comparisons; no executable condition-frequency estimate.'})
    return boundaries


def validate_theory(theory):
    require(theory.get('generation_mode') == MODE and theory.get('rule_representation') == 'natural_language', 'Natural-language theory required')
    all_ids = set(theory['construction_case_ids'])
    require(len(all_ids) == len(theory['construction_case_ids']), 'Duplicate construction cases')
    index = {r['rule_id']: r for r in theory['rules']}
    require(len(index) == len(theory['rules']), 'Duplicate natural rules')
    covered = set();branch_ids = set()
    for rule in theory['rules']:
        require(not {'conditions', 'exceptions', 'when'} & set(rule), 'Executable conditions are forbidden in natural rules')
        require(isinstance(rule['natural_rule_text'], str) and rule['natural_rule_text'].strip(), 'Empty natural rule')
        require(isinstance(rule['boundary_text'], str), 'Invalid natural boundary')
        stats = rule['evidence_statistics']
        partitions = [set(stats[k]) for k in ('supporting_case_ids', 'contradicting_case_ids',
            'contested_case_ids', 'unclassified_case_ids', 'not_compared_case_ids')]
        require(set().union(*partitions) == all_ids and sum(map(len, partitions)) == len(all_ids), 'Invalid natural evidence partition')
        require(set(stats['compared_case_ids']) == all_ids - partitions[-1]
                and stats['eligible_cases'] == len(stats['compared_case_ids']), 'Wrong comparison denominator')
        require(stats['support_count'] == len(partitions[0]) and stats['contradiction_count'] == len(partitions[1]), 'Wrong citation counts')
        require(stats['conditional_action_frequency'] is None, 'Natural rules must not claim executable applicability rates')
    for branch in theory['branches']:
        require(branch['branch_id'] not in branch_ids, 'Duplicate natural theory branch');branch_ids.add(branch['branch_id'])
        refs = strings(branch['source_rule_ids'], 'Invalid branch source rules')
        require(refs and set(refs) <= set(index), 'Invented natural source rule')
        expected_options = action_options_for_refs(refs, index)
        if 'action_options' in branch:
            require(branch['action_options'] == expected_options, 'Theory changed a frozen action alternative')
            require(branch.get('action') == (expected_options[0]['action'] if len(expected_options) == 1 else None),
                    'A multi-action theory cannot silently default to its first action')
        else:
            require(len(expected_options) == 1 and branch['action'] == expected_options[0]['action'],
                    'A fused theory must preserve all source action alternatives')
        require(branch['source_rules'] == [{'rule_id': rid, 'rule_text': index[rid]['natural_rule_text'],
                'boundary_text': index[rid]['boundary_text']} for rid in refs], 'Theory changed a frozen source rule')
        require(not {'when', 'conditions'} & set(branch), 'Executable branch expression is forbidden')
        require(set(branch['evidence_citations']['supporting_case_ids']) <= all_ids
                and set(branch['evidence_citations']['contradicting_case_ids']) <= all_ids, 'Unknown branch evidence')
        covered.update(refs)
    excluded = {r['rule_id'] for r in theory['excluded_theory_rules']}
    require(not covered & excluded and covered | excluded == set(index), 'Source rule missing from theory or exclusions')
    return True


def synthesize(rules, observations, groups, conditions, backend, cache):
    index = {o.case_id: o for o in observations};all_ids = set(index)
    queues = defaultdict(deque)
    for rule in rules: queues[rule['group_id']].append(rule)
    ordered = []
    while any(queues.values()):
        for gid in sorted(queues):
            if queues[gid]: ordered.append(queues[gid].popleft())
    offset, branches, exclusions, questions = 0, [], [], []
    while offset < len(ordered):
        size = len(ordered) - offset
        while True:
            chunk = ordered[offset:offset + size]
            cited = {cid for r in chunk for k in ('supporting_case_ids', 'contradicting_case_ids') for cid in _sample(r[k])}
            packet = code_packet([index[cid] for cid in sorted(cited)], conditions)
            payload = {**packet, 'rules': [{k: r[k] for k in ('rule_id', 'action', 'natural_rule_text',
                'boundary_text', 'induction_rationale', 'supporting_case_ids', 'contradicting_case_ids',
                'cross_group_policy', 'boundary_clauses') if k in r} for r in chunk],
                'evidence_scope': 'Actual compared case-linked evidence. All rule conditions and boundaries are natural language; do not produce executable expressions.'}
            hint = output_size_hint(backend, 'natural_theory_packets', THEORY_TASK, payload)
            if hint is not None and hint < size:
                size = hint
                continue
            if fits(backend, THEORY_TASK, payload): break
            require(size > 1, 'One natural theory packet exceeds context; no evidence was truncated')
            size = max(1, size // 2)
        source = {r['rule_id']: r for r in chunk};shown = {c['case_id'] for c in payload['open_codes']}
        def decode(raw):
            from .natural_theory_validation import decode_theory_payload
            return decode_theory_payload(raw, chunk, payload)
        LOGGER.info('Natural theory synthesis: %d source rules', len(chunk))
        try:
            result = cached_call(backend, cache, 'natural_theory_packets', THEORY_TASK, payload, decode)
        except ModelOutputError as exc:
            if size <= 1: raise
            remember_smaller_output(backend, 'natural_theory_packets', THEORY_TASK, payload, size, exc)
            continue
        # Revalidate cached successful packets too, without rewriting their envelopes.
        result = decode(result)
        for row in result['branches']:
            refs = row['source_rule_ids']
            compared = set().union(*(set(source[rid]['construction_case_ids']) for rid in refs))
            stats = citation_stats(all_ids, compared,
                set().union(*(set(source[rid]['supporting_case_ids']) for rid in refs)),
                set().union(*(set(source[rid]['contradicting_case_ids']) for rid in refs)))
            options = action_options_for_refs(refs, source)
            branches.append({k: row[k] for k in ('situation', 'source_rule_ids', 'interpretation', 'contrast_rule_ids', 'uncertainty')} | {
                'branch_id': stable_id('branch', [VERSION, sorted(refs), row['situation']]),
                'action': options[0]['action'] if len(options) == 1 else None,
                'action_options': options,
                'source_rules': [{'rule_id': rid, 'rule_text': source[rid]['natural_rule_text'],
                    'boundary_text': source[rid]['boundary_text']} for rid in refs],
                'evidence_citations': {k: row[k] for k in ('supporting_case_ids', 'contradicting_case_ids')},
                'evidence_citation_provenance': row['citation_provenance'],
                'evidence_statistics': stats, 'status': 'candidate'})
        exclusions.extend(result['excluded_rules']);questions.extend(result['open_questions']);offset += size
    theory = {'schema_version': '1', 'generation_mode': MODE, 'rule_representation': 'natural_language',
        'version': VERSION, 'theory_action_representation': 'all_frozen_source_action_options_v2', 'rules': rules, 'branches': branches, 'excluded_theory_rules': exclusions,
        'open_questions': list(dict.fromkeys(questions)), 'construction_case_ids': sorted(all_ids),
        'conditions': to_jsonable(conditions), 'rule_format_conversion_performed': False,
        'separate_model_audit': False, 'unsupported_new_conditions_or_actions_allowed': False,
        'prediction_basis': 'Qwen reads frozen natural-language rules, boundaries and situations using Before only.'}
    validate_theory(theory)
    return theory


def predict_natural(before, theory, backend, cache, theory_hash):
    from theory_discovery.prediction.theory_prediction import decode_choice
    require(isinstance(before, str), 'Prediction accepts Before text only')
    validate_theory(theory)
    def relaxed_choice(raw, chunk):
        require(isinstance(raw, dict), 'Missing frozen-theory choice')
        clean = {k: raw.get(k) for k in ('abstain', 'branch_id', 'source_rule_ids', 'reason')}
        if not isinstance(clean['reason'], str) or not clean['reason'].strip():
            clean['reason'] = 'The model supplied a frozen-theory decision without a separate explanation.'
        decision = decode_choice(clean, chunk)
        if not decision['abstain']:
            branch = next(b for b in chunk if b['branch_id'] == decision['branch_id'])
            selected_frozen_actions(branch, decision['source_rule_ids'])
        strategy = raw.get('strategy')
        decision['strategy'] = strategy if isinstance(strategy, str) and strategy.strip() else None
        return decision
    index = {r['rule_id']: r for r in theory['rules']}
    choices = [{k: b[k] for k in ('branch_id', 'situation', 'interpretation', 'source_rule_ids',
        'source_rules', 'action', 'uncertainty')} | {
        'action_options': b.get('action_options', action_options_for_refs(b['source_rule_ids'], index))}
        for b in theory['branches']]
    finalists = choices;selected = None
    while finalists:
        winners, offset = [], 0
        while offset < len(finalists):
            size = min(4, len(finalists) - offset)
            while True:
                chunk = finalists[offset:offset + size]
                payload = {'before': before, 'theory_situations': chunk}
                hint = output_size_hint(backend, 'natural_heldout_choices', PREDICTION_TASK,
                                        dict(payload, frozen_theory_sha256=theory_hash))
                if hint is not None and hint < size:
                    size = hint
                    continue
                if fits(backend, PREDICTION_TASK, payload): break
                require(size > 1, 'One natural theory situation exceeds context; Before was not truncated')
                size = max(1, size // 2)
            try:
                decision = cached_call(backend, cache, 'natural_heldout_choices', PREDICTION_TASK,
                                      dict(payload, frozen_theory_sha256=theory_hash), lambda raw: relaxed_choice(raw, chunk))
            except ModelOutputError as exc:
                if size <= 1: raise
                remember_smaller_output(backend, 'natural_heldout_choices', PREDICTION_TASK,
                                       dict(payload, frozen_theory_sha256=theory_hash), size, exc)
                continue
            if not decision['abstain']:
                winners.append(next(c for c in chunk if c['branch_id'] == decision['branch_id']))
                selected = decision
            offset += size
        if not winners: break
        if len(winners) == 1:
            source_actions = selected_frozen_actions(winners[0], selected['source_rule_ids'])
            strategy = selected.get('strategy') or winners[0]['interpretation']
            return {'strategy': strategy, 'source_action_options': source_actions,
                'abstained': False, 'branch_id': selected['branch_id'], 'source_rule_ids': selected['source_rule_ids'],
                'reason': selected['reason'], 'applicable_branch_count': len(choices), 'unknown_branch_ids': [],
                'rule_representation': 'natural_language'}
        require(len(winners) < len(finalists), 'Natural theory comparisons cannot reduce choices within context')
        finalists = winners
    return {'strategy': '', 'abstained': True, 'branch_id': None, 'source_rule_ids': [],
        'reason': 'The frozen natural-language theory did not justify a concrete choice.',
        'applicable_branch_count': len(choices), 'unknown_branch_ids': [], 'rule_representation': 'natural_language'}


def run_framework(pairs, config, backend, *, synthetic=False):
    from .structured_observation import extract_observations
    config.validate()
    require(backend is not None, 'Natural-language discovery requires a model backend')
    require(config.allow_real_data or synthetic and all(p.metadata.get('synthetic') for p in pairs), 'Unapproved real-data execution')
    require(len(pairs) <= config.max_observations and len({p.case_id for p in pairs}) == len(pairs)
            and all(p.split == 'discovery' for p in pairs), 'Invalid discovery cohort')
    started = time.monotonic();cache = Cache(config.cache_dir);output = __import__('pathlib').Path(config.output_dir)
    LOGGER.info('Stage 1: reuse concrete actions and original Before-only candidates')
    observations, catalog = extract_observations(pairs, config, cache, backend)
    conditions = list(catalog.conditions.values());fingerprint = backend.fingerprint
    resolver = SemanticActionResolver(backend, config.llm.batch_size)
    def grouping():
        return {'groups': to_jsonable(group_actions(observations, resolver)), 'semantic_alignment': resolver.history}
    LOGGER.info('Stage 2: reuse editing-method groups before condition assessment')
    grouped = cache.get_or_compute('action_groups', {'observations': to_jsonable(observations),
        'backend': fingerprint, 'grouping_policy': GROUPING_POLICY}, grouping)
    groups = [ActionGroup(**g) for g in grouped['groups']]
    write_json(output / 'action_groups.json', to_jsonable(groups))
    LOGGER.info('Action grouping complete: %d cases, %d groups', len(observations), len(groups))
    assess_conditions(observations, conditions, backend, cache, workers=config.workers)
    write_json(output / 'stage_progress.json', {'stage': 'natural_rules', 'source_cases': len(observations), 'action_groups': len(groups)})
    LOGGER.info('Stage 3: natural-language rules; no executable condition conversion')
    comparisons, rules = induce(observations, groups, conditions, config, backend, cache)
    write_json(output / 'natural_rules_before_boundaries.json', rules)
    write_json(output / 'stage_progress.json', {'stage': 'natural_boundaries', 'source_rules': len(rules)})
    LOGGER.info('Stage 4: natural-language boundaries from all matched outside cases and all own supports')
    boundaries = add_boundaries(rules, observations, groups, conditions, backend, cache)
    write_json(output / 'natural_rules.json', rules);write_json(output / 'boundary_discovery.json', boundaries)
    write_json(output / 'stage_progress.json', {'stage': 'natural_theory',
        'completed_boundary_rules': len(boundaries), 'source_rules': len(rules)})
    LOGGER.info('Stage 5: synthesize natural-language rules and boundaries into predictive theory')
    theory = synthesize(rules, observations, groups, conditions, backend, cache)
    ledger = EvidenceLedger(o.case_id for o in observations)
    for obs in observations: ledger.record('structured_observation', 'source_observation', [obs.case_id], to_jsonable(obs))
    for rule in rules: ledger.record('natural_rules', 'rule_and_boundary', rule['construction_case_ids'], rule)
    artifacts = {'observations': to_jsonable(observations), 'conditions': to_jsonable(conditions),
        'action_groups': to_jsonable(groups), 'condition_comparisons': comparisons, 'boundary_discovery': boundaries,
        'theory': theory, 'evidence_ledger': ledger.to_dict(), 'semantic_alignment': {
            'action_alignment': grouped['semantic_alignment'], 'condition_alignment': [],
            'global_condition_alignment_enabled': False, 'rule_format_conversion_performed': False}}
    for name, value in artifacts.items(): write_json(output / (name + '.json'), value)
    stats = {'processed_observations': len(observations), 'discovered_methods': len(groups), 'candidate_rules': len(rules),
        'runtime_seconds': time.monotonic() - started, 'cache_hits': cache.hits, 'cache_misses': cache.misses,
        'rule_representation': 'natural_language', 'rule_format_conversion_performed': False,
        'global_condition_alignment_performed': False, 'global_condition_matrix_performed': False,
        'conditional_action_frequency_computed': False, 'citation_counts_only': True}
    write_json(output / 'run_statistics.json', stats)
    write_json(output / 'run_manifest.json', {'status': 'complete', 'version': VERSION, 'config': config.to_dict(),
        'synthetic': synthetic, 'case_ids': [p.case_id for p in pairs], 'statistics': stats, 'artifacts': list(artifacts)})
    write_json(output / 'stage_progress.json', {'stage': 'complete', 'source_rules': len(rules),
        'theory_statements': len(theory['branches']), 'rule_representation': 'natural_language'})
    return artifacts
