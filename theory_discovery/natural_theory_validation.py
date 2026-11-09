"""Validate theory citations against all evidence actually supplied in the packet."""
from copy import deepcopy
import re
from .schemas import require


def _list(value, label):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    require(isinstance(value, list) and all(isinstance(x, str) for x in value), label)
    return value


def decode_theory_payload(raw, rules, payload):
    require(isinstance(raw, dict) and isinstance(raw.get('branches'), list)
            and isinstance(raw.get('excluded_rules'), list),
            'Missing theory branches or source-rule accounting')
    source = {r['rule_id']: r for r in rules}
    shown = {row['case_id'] for row in payload.get('open_codes', [])}
    result = {'branches': [], 'excluded_rules': [],
              'open_questions': _list(raw.get('open_questions'), 'Invalid theory questions')}
    covered, excluded = set(), set()
    for original in raw['branches']:
        require(isinstance(original, dict), 'Invalid theory branch')
        row = deepcopy(original)
        refs = list(dict.fromkeys(_list(row.get('source_rule_ids'), 'Invalid theory source rules')))
        require(refs and set(refs) <= set(source), 'Invented theory source rule')
        row['source_rule_ids'] = refs
        for key in ('situation', 'interpretation'):
            require(isinstance(row.get(key), str) and row[key].strip(), 'Empty theory prose')
        row['contrast_rule_ids'] = list(dict.fromkeys(_list(row.get('contrast_rule_ids'), 'Invalid contrast rules')))
        require(set(row['contrast_rule_ids']) <= set(source), 'Unknown contrast rule')
        linked = {cid for rid in refs for key in ('supporting_case_ids', 'contradicting_case_ids')
                  for cid in source[rid].get(key, [])}
        text_linked = set()
        for rid in refs:
            rule = source[rid]
            text = ' '.join(str(rule.get(key, '')) for key in
                            ('natural_rule_text', 'boundary_text', 'induction_rationale'))
            embedded = set(re.findall(r'\bcase-\d{4}\b', text))
            # These IDs were explicitly provided in source prose and really compared.
            # An invented ID in prose does not create new evidence.
            text_linked.update(embedded & set(rule.get('construction_case_ids', [])))
        allowed = shown | linked | text_linked
        for key in ('supporting_case_ids', 'contradicting_case_ids'):
            row[key] = list(dict.fromkeys(_list(row.get(key), 'Invalid theory citations')))
            unknown = set(row[key]) - allowed
            require(not unknown, 'Theory cites evidence absent from supplied examples and referenced source rules: '
                    + ', '.join(sorted(unknown)))
        require(row['supporting_case_ids'], 'Theory branch requires supporting evidence')
        row['uncertainty'] = _list(row.get('uncertainty'), 'Invalid theory uncertainty')
        cited = set(row['supporting_case_ids']) | set(row['contradicting_case_ids'])
        row['citation_provenance'] = {
            'direct_open_code_case_ids': sorted(cited & shown),
            'source_rule_list_only_case_ids': sorted((cited & linked) - shown),
            'source_rule_text_only_case_ids': sorted((cited & text_linked) - shown - linked),
            'allowed_evidence_scope': 'shown_open_codes_or_cited_source_rule_lists_and_verified_prose',
            'source_rule_ids': refs}
        result['branches'].append(row)
        covered.update(refs)
    for original in raw['excluded_rules']:
        require(isinstance(original, dict), 'Invalid theory exclusion')
        rid = original.get('rule_id')
        require(rid in source and rid not in excluded, 'Unknown or duplicate excluded source rule')
        reason = original.get('reason')
        if not isinstance(reason, str) or not reason.strip():
            reason = 'The model explicitly excluded this rule without a separate explanation.'
        result['excluded_rules'].append({'rule_id': rid, 'reason': reason})
        excluded.add(rid)
    require(not covered & excluded and covered | excluded == set(source),
            'Every natural rule needs a theory statement or explicit exclusion')
    return result


def action_options_for_refs(refs, source):
    """Retain every original direction; never assign a fused branch its first action."""
    import json
    options = {}
    for rid in refs:
        action = source[rid]['action']
        key = json.dumps(action, sort_keys=True, ensure_ascii=False)
        option = options.setdefault(key, {'action': deepcopy(action), 'source_rule_ids': []})
        option['source_rule_ids'].append(rid)
    return list(options.values())


def selected_frozen_actions(branch, refs):
    options = branch.get('action_options')
    if options is None:
        options = [{'action': branch['action'], 'source_rule_ids': branch['source_rule_ids']}]
    available = {rid for option in options for rid in option['source_rule_ids']}
    require(refs and set(refs) <= available, 'Prediction cites an unknown frozen source action')
    return [deepcopy(option) for option in options if set(refs) & set(option['source_rule_ids'])]
