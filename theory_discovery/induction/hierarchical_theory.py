"""Method-group integration followed by a global theory, using prose rules only.

This module is opt-in. Existing natural_pipeline jobs and caches are untouched.
No observation text, open codes, condition catalog or held-out inputs enter prompts.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import logging
from pathlib import Path

from ._support.natural_pipeline import citation_stats, validate_theory
from ._support.natural_theory_validation import action_options_for_refs
from ._support.cross_group_policy import RELATION_GUIDANCE, synthesis_relations
from theory_discovery.core.response_recovery import recoverable_cached_call
from theory_discovery.core.schemas import require, stable_id
from theory_discovery.core.storage import write_json
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION

LOGGER = logging.getLogger(__name__)
VERSION = 'method-group-then-global-prose-v1'
OUTPUT_SCHEMA = RELATION_GUIDANCE + '\n' + (
    'Return {"account":str,"branches":[{"situation":str,"interpretation":str,'
    '"boundary_text":str,"source_rule_ids":[str],"contrast_rule_ids":[str],'
    '"uncertainty":[str]}],"excluded_rules":[{"rule_id":str,"reason":str}],'
    '"open_questions":[str]}. Cite source rule IDs, not case IDs. '
    'Represent every rule in a branch or exclude it with a reason. Preserve source actions, '
    'boundaries and uncertainty; invent no facts, motives or evidence. '
) + '\n' + PLAIN_LANGUAGE_INSTRUCTION
GROUP_TASK = (
    'Integrate rules and boundaries within this editing-method group. Explain when the method applies, '
    'how its variants relate and what limits it. Summarize shared patterns while retaining each independent source rule; '
    'do not just paraphrase each rule. Let the sources determine the number of patterns. '
    'Use existing rule accounts, without reanalysing cases or requesting original text. '
    + OUTPUT_SCHEMA
)
GLOBAL_TASK = (
    'Build a theory from all supplied group accounts and linked rules. Explain shared patterns, '
    'context differences and why different methods apply. Retain independent conditions, boundaries and actions across groups; do not concatenate '
    'accounts. Allow multiple rules to apply in one case. Preserve each rule with its own conditions and action. '
    'Do not force one action or a fixed number of propositions. Use no new case evidence. '
    'The account should explain relationships, not list methods. '
    + OUTPUT_SCHEMA
)
COMPACT_TASK = (
    'Prepare this already-integrated method-group account for the global integration. '
    'Compress repetitive prose while preserving each source reference, important context, '
    'action distinction, exclusion/boundary and unresolved tension. Do not add evidence. '
    'The prose word budget is supplied; reference IDs do not count toward it. '
    + OUTPUT_SCHEMA
)


def _strings(value, label):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    require(isinstance(value, list) and all(isinstance(x, str) for x in value), label)
    return list(dict.fromkeys(value))


def _decode(raw, allowed, *, allow_unassigned=False):
    require(isinstance(raw, dict) and isinstance(raw.get('branches'), list),
            'Integration requires a complete branches list')
    result = {'branches': [], 'excluded_rules': [],
              'open_questions': _strings(raw.get('open_questions'), 'Invalid integration questions')}
    covered, excluded = set(), set()
    for original in raw['branches']:
        require(isinstance(original, dict), 'Invalid integrated proposition')
        row = {key: original.get(key) for key in ('situation', 'interpretation')}
        require(all(isinstance(v, str) and v.strip() for v in row.values()),
                'Integrated proposition needs situation and interpretation')
        refs = _strings(original.get('source_rule_ids'), 'Invalid integration source IDs')
        require(refs and set(refs) <= allowed, 'Integration invented a source rule')
        row['source_rule_ids'] = refs
        row['contrast_rule_ids'] = _strings(original.get('contrast_rule_ids'), 'Invalid contrast IDs')
        require(set(row['contrast_rule_ids']) <= allowed, 'Integration invented a contrast rule')
        row['uncertainty'] = _strings(original.get('uncertainty'), 'Invalid uncertainty')
        boundary = original.get('boundary_text')
        row['boundary_text'] = boundary if isinstance(boundary, str) else ''
        result['branches'].append(row)
        covered.update(refs)
    exclusions = raw.get('excluded_rules') or []
    require(isinstance(exclusions, list), 'Invalid integration exclusions')
    for row in exclusions:
        require(isinstance(row, dict), 'Invalid source exclusion')
        rid = row.get('rule_id')
        require(rid in allowed and rid not in excluded, 'Unknown or duplicate integration exclusion')
        reason = row.get('reason')
        result['excluded_rules'].append({'rule_id': rid, 'reason': reason if isinstance(reason, str)
                                        and reason.strip() else 'Explicitly excluded by the integrator.'})
        excluded.add(rid)
    require(not covered & excluded and (allow_unassigned or covered | excluded == allowed),
            'Integration must account for every supplied source rule')
    if allow_unassigned:
        result['unassigned_rule_ids'] = sorted(allowed - covered - excluded)
    account = raw.get('account')
    result['account'] = (account if isinstance(account, str) and account.strip()
                         else ' '.join(b['interpretation'] for b in result['branches']))
    return result


def _source_view(rule):
    # Evidence IDs/counts are inherited in code after synthesis, not sent back to the LLM.
    view = {key: deepcopy(rule[key]) for key in
            ('rule_id', 'action', 'natural_rule_text', 'boundary_text')}
    if rule.get('cross_group_policy'):
        view['cross_group_policy'] = rule['cross_group_policy']
    return view


def _fit(backend, task, payload):
    checker = getattr(backend, 'prompt_fits', None)
    return checker is None or checker(task, payload, repair_reserve=True)


def _call(backend, cache, stage, task, payload, allowed):
    require(_fit(backend, task, payload),
            'Whole integration input exceeds context; no rules or groups were dropped')
    decode = lambda raw: _decode(raw, allowed)
    result = recoverable_cached_call(backend, cache, stage, VERSION, task, payload, decode)
    return decode(result)


def global_payload(accounts):
    # Every group is present in ONE global input. No case-level evidence or raw rules.
    return {'group_theories': [
        {key: deepcopy(group[key]) for key in
         ('group_id', 'action', 'account', 'branches', 'open_questions')}
        for group in accounts if group['branches']],
        'evidence_scope': 'Previously established rule/boundary accounts only. No new case analysis.'}


def _branch(row, source, all_ids):
    refs = row['source_rule_ids']
    compared = set().union(*(set(source[rid]['construction_case_ids']) for rid in refs))
    support = set().union(*(set(source[rid]['supporting_case_ids']) for rid in refs))
    contradictions = set().union(*(set(source[rid]['contradicting_case_ids']) for rid in refs))
    stats = citation_stats(all_ids, compared, support, contradictions)
    options = action_options_for_refs(refs, source)
    return {
        **{key: deepcopy(row[key]) for key in ('situation', 'interpretation', 'source_rule_ids',
                                             'contrast_rule_ids', 'uncertainty', 'boundary_text')},
        **{key: deepcopy(row[key]) for key in ('boundary_summaries', 'boundary_detail_only_rule_ids')
           if key in row},
        'branch_id': stable_id('branch', [VERSION, sorted(refs), row['situation']]),
        'action': options[0]['action'] if len(options) == 1 else None,
        'action_options': options,
        'source_group_ids': sorted({source[rid]['group_id'] for rid in refs}),
        'source_rules': [{'rule_id': rid, 'rule_text': source[rid]['natural_rule_text'],
                          'boundary_text': source[rid]['boundary_text']} for rid in refs],
        'evidence_citations': {key: stats[key] for key in
                              ('supporting_case_ids', 'contradicting_case_ids')},
        'evidence_citation_provenance': {
            'source_rule_ids': refs,
            'allowed_evidence_scope': 'inherited_source_rule_evidence_not_reexamined',
            'direct_open_code_case_ids': [],
            'new_case_assessment_performed': False},
        'evidence_statistics': stats, 'status': 'candidate'}


def synthesize_hierarchical(rules, construction_case_ids, backend, cache, *,
                            output=None, workers=4, staged_global_backend=None, candidate_evidence=None):
    """Reuse completed rules/boundaries, integrate each group, then all groups.

    output is a dedicated version directory; callers must never pass active outputs.
    Existing caches are versioned by prompts and this method's VERSION.
    """
    require(rules and workers >= 1, 'Expected completed rules and positive workers')
    source = {r['rule_id']: r for r in rules}
    require(len(source) == len(rules), 'Duplicate source rule')
    all_ids = set(construction_case_ids)
    require(len(all_ids) == len(construction_case_ids), 'Duplicate construction case')
    grouped = defaultdict(list)
    for rule in rules:
        require(rule['group_id'] == rule['action']['group_id'], 'Rule/action group mismatch')
        grouped[rule['group_id']].append(rule)
    progress = {'method': VERSION, 'stage': 'group_integration', 'source_rules': len(rules),
                'total_groups': len(grouped), 'completed_groups': 0}
    def save_progress(**updates):
        progress.update(updates)
        if output is not None:
            write_json(Path(output) / 'stage_progress.json', progress)
    save_progress()
    payloads = {}
    for gid in sorted(grouped):
        group_rules = grouped[gid]
        payload = {'group_id': gid, 'action': deepcopy(group_rules[0]['action']),
                   'rules': [_source_view(r) for r in group_rules]}
        # All rules of one method group stay together, never round-robin across groups.
        require(_fit(backend, GROUP_TASK, payload),
                'One entire method group exceeds context; no source rule was truncated')
        payloads[gid] = payload
    def integrate(gid):
        LOGGER.info('Within-group integration: %s, %d source rules', gid, len(grouped[gid]))
        result = _call(backend, cache, 'hierarchical_group_theories', GROUP_TASK,
                       payloads[gid], {r['rule_id'] for r in grouped[gid]})
        return {'group_id': gid, 'action': deepcopy(grouped[gid][0]['action']), **result}
    by_group = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(grouped))) as pool:
        pending = {pool.submit(integrate, gid): gid for gid in sorted(grouped)}
        for future in as_completed(pending):
            group = future.result()
            by_group[group['group_id']] = group
            save_progress(completed_groups=len(by_group))
            LOGGER.info('Integrated method groups: %d/%d', len(by_group), len(grouped))
    accounts = [by_group[gid] for gid in sorted(by_group)]
    if output is not None:
        write_json(Path(output) / 'group_theories.json', accounts)
    group_exclusions = [row for group in accounts for row in group['excluded_rules']]
    eligible = set(source) - {row['rule_id'] for row in group_exclusions}
    payload = global_payload(accounts)
    compactions = []
    # Compact group accounts if needed; every source and every group remains represented.
    # Never substitute independent partial "global" batches for a global integration.
    for budget in ((160, 90) if staged_global_backend is None else ()):
        if _fit(backend, GLOBAL_TASK, payload):
            break
        compacted = []
        for group in accounts:
            allowed = {rid for row in group['branches'] for rid in row['source_rule_ids']}
            if not allowed:
                compacted.append(group)
                continue
            compact_payload = {'group_theory': group, 'prose_word_budget': budget}
            result = _call(backend, cache, 'hierarchical_group_compaction', COMPACT_TASK,
                           compact_payload, allowed)
            require(not result['excluded_rules'], 'Compaction cannot remove a source rule')
            compacted.append({**group, **result, 'excluded_rules': group['excluded_rules']})
        accounts = compacted
        payload = global_payload(accounts)
        compactions.append({'prose_word_budget': budget, 'groups_retained': len(accounts)})
    require(not eligible or staged_global_backend is not None or _fit(backend, GLOBAL_TASK, payload),
            'All group accounts exceed global context after compaction; no group was dropped')
    save_progress(stage='global_integration', completed_groups=len(accounts))
    LOGGER.info('Global integration: all %d method groups, %d source rules',
                len(accounts), len(eligible))
    if eligible and staged_global_backend is not None:
        from ._support.staged_global_theory import VERSION as staged_version, complete_staged_global_theory
        result = complete_staged_global_theory(staged_global_backend, cache, eligible, source,
                                               output=output, save_progress=save_progress,
                                               candidate_evidence=candidate_evidence)
    elif eligible:
        from ._support.global_theory_recovery import complete_global_theory
        result = complete_global_theory(backend, cache, payload, eligible, source, output=output)
    else:
        result = {'account': 'No supported source rules remained after within-group integration.',
                  'branches': [], 'excluded_rules': [], 'open_questions': []}
    theory = {
        'schema_version': '1', 'generation_mode': 'natural_language_predictive_theory',
        'version': VERSION, 'rule_representation': 'natural_language',
        'theory_action_representation': 'all_frozen_source_action_options_v2',
        'integration_method': 'within_method_group_then_all_groups',
        'global_explanation': result['account'],
        'group_theories': accounts, 'rules': deepcopy(rules),
        'branches': [_branch(row, source, all_ids) for row in result['branches']],
        'excluded_theory_rules': group_exclusions + result['excluded_rules'],
        'open_questions': list(dict.fromkeys(result['open_questions'] +
                              [q for group in accounts for q in group['open_questions']])),
        'construction_case_ids': sorted(all_ids), 'conditions': [],
        'rule_format_conversion_performed': False, 'separate_model_audit': False,
        'unsupported_new_conditions_or_actions_allowed': False,
        'prediction_basis': 'Qwen reads frozen integrated situations and their source-rule boundaries using Before only.',
        'integration_input_policy': 'existing_rule_and_boundary_prose_and_rule_IDs_only',
        'case_level_open_codes_supplied': False, 'original_text_supplied': False,
        'integration_metadata': {
            'source_rule_count': len(rules), 'groups_with_rules': len(grouped),
            'global_input_group_count': len(payload['group_theories']),
            'compactions': compactions,
            'source_rules_per_branch': [len(row['source_rule_ids']) for row in result['branches']]}}
    if eligible and staged_global_backend is not None:
        theory['integration_metadata']['global_synthesis'] = staged_version
    if "boundary_candidate_review" in result:
        theory["boundary_candidate_review"] = result["boundary_candidate_review"]
    validate_theory(theory)
    if output is not None:
        write_json(Path(output) / 'theory.json', theory)
        write_json(Path(output) / 'group_theories_for_global.json', accounts)
        write_json(Path(output) / 'integration_summary.json', theory['integration_metadata'])
    save_progress(stage='complete', theory_branches=len(theory['branches']))
    return theory
