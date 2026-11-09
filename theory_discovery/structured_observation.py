"""Stage 1: local diff, semantic action interpretation, separate predicates."""
from .llm import PLAIN_LANGUAGE_INSTRUCTION, checked_infer

STAGE_INSTRUCTION = PLAIN_LANGUAGE_INSTRUCTION
import difflib
import re
from .conditions import CONDITION_TASK, CONDITION_REVIEW_TASK, ConditionCatalog, ConditionDetector, review_condition_quality
from .schemas import Action, EvidenceSpan, Observation, require, stable_id, to_jsonable, resolve_span

ACTION_TASK = 'Describe concrete edits from Before/After and diff: reusable technique, target and transformation. Allow multiple actions, cover every diff and infer no intent.\nReturn {"actions":[{"family":"deletion"|"insertion"|"replacement"|"reordering"|"multi_span","method":str,"target_feature":str,"transformation":str,"scope":"word"|"phrase"|"claim"|"multi_span","target_expression":str,"diff_indices":[int],"uncertainty_flags":[str]}]}. Python links case IDs and diff evidence; omit quotes/offsets.' + "\n" + PLAIN_LANGUAGE_INSTRUCTION


def changed_spans(before, after):
    """Word/punctuation diff adapted from former wnc.common; whitespace fallback."""
    a = list(re.finditer(r'\w+|[^\w\s]', before))
    b = list(re.finditer(r'\w+|[^\w\s]', after))
    matcher = difflib.SequenceMatcher(None, [t.group() for t in a], [t.group() for t in b], autojunk=False)

    def bounds(tokens, lo, hi, text):
        if lo == hi:
            pos = tokens[lo].start() if lo < len(tokens) else len(text)
            return pos, pos
        return tokens[lo].start(), tokens[hi - 1].end()

    result = []
    for op, i, j, k, l in matcher.get_opcodes():
        if op == 'equal':
            continue
        s, e = bounds(a, i, j, before)
        t, u = bounds(b, k, l, after)
        result.append({'operation': op, 'before_start': s, 'before_end': e,
                       'after_start': t, 'after_end': u,
                       'before_text': before[s:e], 'after_text': after[t:u]})
    if not result and before != after:
        for op, i, j, k, l in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            if op != 'equal':
                result.append({'operation': op, 'before_start': i, 'before_end': j,
                               'after_start': k, 'after_end': l, 'before_text': before[i:j], 'after_text': after[k:l]})
    # Annotate unambiguous exact moves; preserve raw deletion/insertion provenance.
    for i, span in enumerate(result):
        if span['operation'] != 'delete':
            continue
        partners = [j for j, other in enumerate(result) if other['operation'] == 'insert'
                    and span['before_text'] == other['after_text'] and bool(span['before_text'].strip())]
        sources = [d for d in result if d['operation'] == 'delete' and d['before_text'] == span['before_text']]
        if len(partners) == len(sources) == 1:
            span['move_partner'] = partners[0]
            result[partners[0]]['move_partner'] = i
    return result


def mechanical_actions(pair, diff):
    actions, used = [], set()
    for index, span in enumerate(diff):
        if index in used:
            continue
        indices = sorted({index, span.get('move_partner', index)})
        used.update(indices)
        family = 'reordering' if len(indices) == 2 else {'delete': 'deletion', 'insert': 'insertion', 'replace': 'replacement'}[span['operation']]
        evidence = []
        for idx in indices:
            edit = diff[idx]
            for side in ('before', 'after'):
                if edit[side + '_text']:
                    evidence.append(EvidenceSpan(side, edit[side + '_start'], edit[side + '_end'], edit[side + '_text']))
        actions.append(Action(stable_id('action', [pair.case_id, indices, family]), family, family,
                              'uninterpreted text span', family, 'multi_span' if len(indices) > 1 else 'phrase',
                              span['before_text'] or span['after_text'], evidence, indices,
                              semantic=False, uncertainty_flags=['semantic_interpretation_required']))
    return actions


def interpret_actions(pair, diff, backend):
    if backend is None:
        return mechanical_actions(pair, diff)
    payload = {'before_text': pair.before_text, 'after_text': pair.after_text, 'diff': diff}
    attempts = getattr(getattr(backend, 'config', None), 'retries', 2) + 1
    for attempt in range(attempts):
        actions = checked_infer(backend, ACTION_TASK, payload, lambda raw: decode_actions(pair, diff, raw))
        if not hasattr(backend, 'config'):
            return actions
        def review_decode(raw):
            require(set(raw) == {'acceptable', 'issues'} and type(raw['acceptable']) is bool
                    and isinstance(raw['issues'], list), 'Invalid action-quality review')
            require(raw['acceptable'] or bool(raw['issues']), 'Rejected action review needs issues')
            return raw
        review = checked_infer(backend, ACTION_REVIEW_TASK,
            {'before_text': pair.before_text, 'after_text': pair.after_text, 'diff': diff, 'actions': to_jsonable(actions)}, review_decode)
        if review['acceptable']:
            return actions
        payload = dict(payload, quality_feedback={'issues': review['issues'], 'previous_actions': to_jsonable(actions), 'revision_attempt': attempt + 1})
    raise ValueError('Concrete action quality unresolved: ' + '; '.join(review['issues']))


def decode_actions(pair, diff, raw):
    require(set(raw) == {'actions'} and isinstance(raw['actions'], list), 'Invalid action response')
    actions = []
    for index, item in enumerate(raw['actions']):
        data = dict(item)
        supplied_ids = data.pop('evidence_case_ids', [])
        require(isinstance(supplied_ids, list) and set(supplied_ids) <= {pair.case_id}, 'Unknown action evidence case ID')
        data.pop('evidence_spans', None)
        indices = data.get('diff_indices', [])
        require(bool(indices) and all(type(i) is int and 0 <= i < len(diff) for i in indices), 'Invalid source diff reference')
        spans = []
        for i in indices:
            for side in ('before', 'after'):
                edit = diff[i]
                if edit[side + '_text']:
                    spans.append(EvidenceSpan(side, edit[side + '_start'], edit[side + '_end'], edit[side + '_text']))
        action = Action(action_id=stable_id('action', [pair.case_id, index, item]),
            **dict(data, evidence_spans=spans, evidence_case_ids=[pair.case_id]))
        require(action.family in ('deletion', 'insertion', 'replacement', 'reordering', 'multi_span'), 'Invalid action family')
        require(action.scope in ('word', 'phrase', 'claim', 'multi_span'), 'Invalid action scope')
        require(all(isinstance(v, str) and v.strip() for v in (action.method, action.target_feature, action.transformation, action.target_expression)), 'Incomplete concrete action')
        simple_method = re.sub(r'[_\W]+', ' ', action.method.casefold()).strip()
        simple_target = re.sub(r'[_\W]+', ' ', action.target_feature.casefold()).strip()
        generic_methods = {'move', 'remove', 'add', 'word swap', 'replace', 'replacement', 'deletion', 'insertion', 'reorder', 'edit', 'change'}
        generic_targets = {'word', 'phrase', 'adjective', 'noun', 'sentence', 'punctuation', 'text', 'meaning', 'expression'}
        require(not (simple_method in generic_methods and simple_target in generic_targets),
                'Method and target are too generic: name the reusable editing technique and linguistic role, using only the supplied evidence')
        require(bool(action.evidence_spans) and bool(action.diff_indices), 'Action lacks evidence')
        require(all(type(i) is int and 0 <= i < len(diff) for i in action.diff_indices), 'Invalid diff reference')
        for span in action.evidence_spans:
            span.validate(pair)
        require(any(any(span.start <= diff[i][span.side + '_end'] and span.end >= diff[i][span.side + '_start']
                        for i in action.diff_indices) for span in action.evidence_spans), 'Action evidence does not touch edit')
        actions.append(action)
    covered = {i for a in actions for i in a.diff_indices}
    require(covered == set(range(len(diff))), 'Unrepresented edits; retain them instead of silently dropping')
    return actions


def extract_observations(pairs, config, cache, backend=None, *, retain_unchanged=False):
    from concurrent.futures import ThreadPoolExecutor
    import logging
    index = {p.case_id: p for p in pairs}
    require(len(index) == len(pairs), 'Duplicate case IDs')
    catalog = ConditionCatalog(index, config.pre_context_keys)
    fingerprint = backend.fingerprint if backend is not None else 'mechanical-only-v1'
    def extract_one(item):
        pair_index, pair = item
        logging.getLogger(__name__).info('Extracting observation %d/%d: %s', pair_index + 1, len(pairs), pair.case_id)
        require(pair.split == 'discovery', 'Only discovery pairs allowed')
        context = pair.before(config.pre_context_keys)
        diff = changed_spans(pair.before_text, pair.after_text)
        if retain_unchanged and pair.before_text == pair.after_text:
            evidence = EvidenceSpan('before', 0, len(pair.before_text), pair.before_text)
            actions = [Action(stable_id('action', [pair.case_id, 'no_change']), 'no_change',
                              'keep wording unchanged', 'existing wording', 'retain original wording',
                              'multi_span', pair.before_text, [evidence], [], semantic=False)]
        else:
            raw_actions = cache.get_or_compute('actions', {'pair': to_jsonable(pair), 'backend': fingerprint},
                lambda: interpret_actions(pair, diff, backend))
            actions = [Action(**dict(a, evidence_spans=[EvidenceSpan(**span) for span in a['evidence_spans']])) for a in raw_actions]
        for action in actions:
            action.evidence_case_ids = [pair.case_id]
        proposals = {'conditions': []}
        if backend is not None:
            action_hints = condition_action_hints(actions)
            proposals = cache.get_or_compute('condition_proposals',
                {'context': context.to_dict(), 'observed_actions': action_hints,
                 'task': CONDITION_TASK, 'review_task': CONDITION_REVIEW_TASK, 'backend': fingerprint},
                lambda: discover_relevant_conditions(backend, context, pair, config.pre_context_keys,
                                                     action_hints=action_hints))
        return pair, context, actions, proposals
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        first_pass = list(pool.map(extract_one, enumerate(pairs)))
    extracted = []
    from .schemas import Condition
    for pair, context, actions, proposals in first_pass:
        candidates = []
        for item in proposals['conditions']:
            item = dict(item)
            item.pop('evidence_spans', None)
            spans = []
            candidates.append(catalog.add(Condition('', evidence_by_case={pair.case_id: spans}, **item)))
        extracted.append((pair, context, actions, candidates))
    # Keep original per-case features. No global semantic condition alignment or matrix.
    observations=[]
    for pair, context, actions, candidates in extracted:
        spans=[span for action in actions for span in action.evidence_spans]
        flags=sorted({flag for action in actions for flag in action.uncertainty_flags})
        if not actions or (retain_unchanged and pair.before_text == pair.after_text):
            flags.append('no_text_change')
        if not candidates:flags.append('no_supported_candidate_conditions')
        observations.append(Observation(pair.case_id,pair.before_text,pair.after_text,actions,candidates,{},
            spans,flags,context.pre_context,condition_evidence={}))
    return observations,catalog


def validate_condition_proposals(raw, pair, allowlist, *, require_candidates=False):
    from .schemas import Condition
    require(set(raw) == {'conditions'} and isinstance(raw['conditions'], list), 'Invalid condition proposals')
    if require_candidates:
        require(1 <= len(raw['conditions']) <= 4,
                'Return one to four observable Before candidates. Use a concrete wording or structural cue if meaning is uncertain.')
    output = []
    for item in raw['conditions']:
        item = dict(item)
        item.pop('evidence_spans', None)
        supplied_ids = item.pop('evidence_case_ids', [])
        require(isinstance(supplied_ids, list) and set(supplied_ids) <= {pair.case_id}, 'Unknown condition evidence case ID')
        spans = []
        condition = Condition('', evidence_by_case={pair.case_id: spans}, **item)
        condition.validate({pair.case_id: pair}, allowlist)
        output.append(dict(item, detector=condition.detector, evidence_spans=to_jsonable(spans)))
    return {'conditions': output}


def condition_action_hints(actions):
    # The Before itself is supplied in full. Include short verified Before targets
    # without repeating whole deleted passages or supplying replacement text.
    return [dict(family=action.family, method=action.method, target_feature=action.target_feature,
                 scope=action.scope, before_targets=list(dict.fromkeys(
                     span.quote for span in action.evidence_spans
                     if span.side == 'before' and 0 < len(span.quote) <= 160)))
            for action in actions]


SEMANTIC_CONDITION_REPAIR_TASK = """The prior candidate reply used an invalid detector format. Re-propose one to four observable Before features related to the supplied action hints. Return only {"conditions":[{"name":str,"definition":str,"value_type":"boolean"|"number"|"string","discovery_note":str}]} with those exact fields; Python assigns the semantic detector.
Choose the value type for the actual feature. Presence, absence, or attribution of a review quotation is a yes/no property of its content. For edits to review quotations, look for the quoted evaluation, its attribution, or a claim of acclaim when visible. Whole-text length alone cannot establish the presence of a review. Ground the feature in Before and do not invent motives or unseen evidence.""" + "\n" + PLAIN_LANGUAGE_INSTRUCTION


def propose_conditions(backend, payload, pair, allowlist, *, guided):
    def decode(raw):
        return validate_condition_proposals(raw, pair, allowlist, require_candidates=guided)

    try:
        return checked_infer(backend, CONDITION_TASK, payload, decode)
    except ValueError as error:
        detail = str(error.__cause__ or error)
        detector_errors = ('Unexpected detector parameters', 'Unsupported detector',
                           'contains must be boolean', 'regex must be boolean',
                           'text_length must be numeric', 'Empty contains detector',
                           'Invalid regex', 'Context key not declared pre-edit')
        if not guided or not detail.startswith(detector_errors):
            raise

    return checked_infer(backend, SEMANTIC_CONDITION_REPAIR_TASK,
                         dict(payload, detector_validation_error=detail),
                         lambda raw: validate_semantic_condition_candidates(raw, pair, allowlist))


def validate_semantic_condition_candidates(raw, pair, allowlist):
    require(isinstance(raw, dict) and set(raw) == {'conditions'}
            and isinstance(raw['conditions'], list), 'Invalid semantic condition candidates')
    fields = {'name', 'definition', 'value_type', 'discovery_note'}
    require(all(isinstance(item, dict) and set(item) == fields for item in raw['conditions']),
            'Return candidate fields only. Python supplies the detector.')
    return validate_condition_proposals(
        {'conditions': [dict(item, detector={'kind': 'semantic'}) for item in raw['conditions']]},
        pair, allowlist, require_candidates=True)


QUALITY_REPROPOSAL_TASK = """Open-code one to four observable candidate features of Before. Use the observed action and edit_extent to focus attention. Treat candidates as observations for later comparison. For WHOLE_EXCERPT_REMOVED, examine the kind of content and how it is presented, such as comparative praise, an evaluative claim, attribution, or the excerpt's discourse role. Local wording can be evidence of a passage-level pattern. For UNCHANGED, code visible characteristics of the retained wording, such as explicit attribution, descriptive factual language, or structure. Do not invent an edit or infer why it happened. For PARTIAL_EDIT, focus on the changed target and its context in Before. Return only {"conditions":[{"name":str,"definition":str,"value_type":"boolean"|"number"|"string","discovery_note":str}]}. Python supplies the detector.""" + "\n" + PLAIN_LANGUAGE_INSTRUCTION

QUALITY_SELECTION_TASK = """Review candidate open codes individually. Retain the indices of observable Before features plausibly relevant for later comparison. A feature can qualify when the whole excerpt was removed or stayed unchanged; it need not justify the action or identify a replacement edit. For whole removal, consider content and framing of the removed passage. For unchanged text, consider visible features of the retained wording. Reject unsupported details, invented motives, and bare topic names. Return only {"retained_indices":[int],"issues":[str]}. Indices start at zero. If no candidate is supportable, return an empty list and explain what is missing.""" + "\n" + PLAIN_LANGUAGE_INSTRUCTION


def recover_condition_quality(backend, payload, pair, allowlist, attempts):
    if pair.before_text == pair.after_text:
        extent = 'UNCHANGED'
    elif not pair.after_text.strip():
        extent = 'WHOLE_EXCERPT_REMOVED'
    else:
        extent = 'PARTIAL_EDIT'
    scoped = dict(payload, edit_extent=extent)
    retry_payload = scoped
    for attempt in range(attempts):
        proposals = checked_infer(backend, QUALITY_REPROPOSAL_TASK, retry_payload,
            lambda raw: validate_semantic_condition_candidates(raw, pair, allowlist))
        candidates = proposals['conditions']

        def decode_selection(raw):
            require(isinstance(raw, dict) and set(raw) == {'retained_indices', 'issues'},
                    'Expected candidate indices and issues')
            indices = raw['retained_indices']
            require(isinstance(indices, list) and all(type(i) is int and 0 <= i < len(candidates) for i in indices)
                    and len(indices) == len(set(indices)), 'Use distinct supplied zero-based candidate indices')
            issues = raw['issues']
            require(isinstance(issues, list) and all(isinstance(issue, str) for issue in issues)
                    and (indices or any(issue.strip() for issue in issues)), 'Explain an empty selection')
            return raw

        review = checked_infer(backend, QUALITY_SELECTION_TASK,
            dict(scoped, candidate_conditions=candidates), decode_selection)
        if review['retained_indices']:
            return {'conditions': [candidates[i] for i in review['retained_indices']]}
        retry_payload = dict(scoped, review_issues=review['issues'], revision_attempt=attempt + 1)
    raise ValueError('Candidate condition quality unresolved for ' + pair.case_id + ': ' + '; '.join(review['issues']))


def discover_relevant_conditions(backend, context, pair, allowlist, *, action_hints=None):
    base_payload = context.to_dict()
    if action_hints is not None:
        base_payload['observed_actions'] = action_hints
    payload = base_payload
    attempts = getattr(getattr(backend, 'config', None), 'retries', 2) + 1
    for attempt in range(attempts):
        proposals = propose_conditions(backend, payload, pair, allowlist, guided=action_hints is not None)
        # Unconfigured synthetic fixtures exercise structural contracts directly.
        if not hasattr(backend, 'config'):
            return proposals
        review = review_condition_quality(backend, context, proposals, action_hints=action_hints)
        if review['acceptable']:
            return proposals
        payload = dict(base_payload, quality_feedback={
            'issues': review['issues'], 'rejected_candidate_conditions': proposals['conditions'], 'revision_attempt': attempt + 1,
            'instruction': ('Return one to four observable Before candidates related to the action hints. '
                            'Use a concrete wording or structural cue if meaning is uncertain.'
                            if action_hints is not None else
                            'Replace irrelevant identity/topic predicates with supported reusable editing-relevant features; return [] if none can be supported.')})
    if action_hints is not None:
        return recover_condition_quality(backend, base_payload, pair, allowlist, attempts)
    raise ValueError('Candidate condition quality unresolved: ' + '; '.join(review['issues']))


ACTION_REVIEW_TASK = 'Check that actions describe the supplied pair/diff and cover every edit. Reject invented intent, unsupported or duplicate actions, and vague meaning-change labels. Allow paraphrases and multiple actions. Case-linked evidence is sufficient; do not judge quote formatting.\nReturn {"acceptable":bool,"issues":[str]}.' + "\n" + PLAIN_LANGUAGE_INSTRUCTION
