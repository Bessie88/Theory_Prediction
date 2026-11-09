"""Meaning-based alignment for actions and conditions, preserving original evidence."""
from copy import deepcopy
from .llm import PLAIN_LANGUAGE_INSTRUCTION, checked_infer
from .schemas import Condition, require, stable_id, to_jsonable

ACTION_ALIGNMENT_TASK = """Group edits by approximately the same method, despite different wording, targets or word/phrase scope. Preserve meaningful direction, such as broadening versus narrowing. Create a category only for a different method.
Return {"assignments":[{"incoming_id":str,"existing_group_id":str|null,"canonical":{"family":str,"method":str,"target_feature":str,"scope":str,"semantic":true},"uncertainty_flags":[str],"rationale":str}]}.
Assign each incoming ID once to a supplied category or null for a new one. Equivalent new methods may share a category. Copy existing canonical descriptions; name new methods briefly. Keep semantic=true: it records model interpretation, including spelling/punctuation, not a change in meaning. Original actions stay preserved.""" + "\n" + PLAIN_LANGUAGE_INSTRUCTION

CONDITION_ALIGNMENT_TASK = 'Align conditions with approximately the same observable meaning; wording need not match.\nKeep differences that change when the condition holds. Preserve editing-relevant features, not names or topic keywords.\nReturn {"assignments":[{"incoming_id":str,"existing_condition_id":str|null,\n"canonical":{"name":str,"definition":str,"value_type":"boolean"|"number"|"string"},"rationale":str}]}.\nAssign every incoming ID once, using supplied existing IDs only. Copy existing definitions; new paraphrases may share\none reusable Before-only definition. Choose its value_type to match the canonical meaning, not the source label: whether/presence is boolean, a measured quantity is number, a category is string. You may unify inconsistent source types when their observable meaning is equivalent; retain different meanings such as presence versus magnitude. Re-evaluate every Before using the canonical definition. No action labels, case-specific exceptions, or empirical claims.' + "\n" + PLAIN_LANGUAGE_INSTRUCTION


class SemanticActionResolver:
    def __init__(self, backend, batch_size=4, initial_groups=None):
        self.backend, self.batch_size = backend, batch_size
        self.history = []
        self.initial_groups = initial_groups or {}

    def resolve(self, refs):
        def operation_id(ref):
            return stable_id('operation', {'signature': ref['signature'], 'transformation': ref.get('transformation', '')})
        unique = {operation_id(r): {'signature': r['signature'], 'transformation': r.get('transformation', '')} for r in refs}
        incoming = [{'incoming_id': key, **value} for key, value in sorted(unique.items()) if value['signature']['semantic']]
        groups, assignments = dict(self.initial_groups), {}
        for offset in range(0, len(incoming), self.batch_size):
            chunk = incoming[offset:offset + self.batch_size]
            raw = alignment_infer(self.backend, ACTION_ALIGNMENT_TASK, {'existing_groups': [
                {'group_id': gid, 'canonical': desc} for gid, desc in sorted(groups.items())], 'incoming_methods': chunk})
            require(set(raw) == {'assignments'}, 'Invalid action alignment response')
            items = raw['assignments']
            ids = [x['incoming_id'] for x in items]
            require(len(ids) == len(set(ids)) and set(ids) == {x['incoming_id'] for x in chunk}, 'Action alignment lost/invented methods')
            anchors = set(groups)
            for item in items:
                iid, gid, canonical = item['incoming_id'], item['existing_group_id'], item['canonical']
                require(set(canonical) == set(unique[iid]['signature']) and canonical['semantic'] is True, 'Invalid canonical action')
                if gid is not None:
                    require(gid in anchors, 'Invalid action anchor')
                    canonical = groups[gid]
                else:
                    gid = stable_id('group', canonical)
                    groups.setdefault(gid, canonical)
                assignments[iid] = canonical
                self.history.append({**item, 'assigned_group_id': gid, 'source_action_ids': [r['action_id'] for r in refs if operation_id(r) == iid]})
            import logging
            logging.getLogger(__name__).info('Editing-method grouping progress: %d/%d distinct operation descriptions; %d method types',
                offset+len(chunk),len(incoming),len(groups))
        return {r['action_id']: assignments.get(operation_id(r), r['signature']) for r in refs}


def align_conditions(catalog, backend, batch_size=4):
    originals = list(catalog.conditions.values())
    catalog.original_conditions = to_jsonable(originals)
    catalog.alignment_history = []
    aliases, canonical = {}, {}
    for offset in range(0, len(originals), batch_size):
        chunk = originals[offset:offset + batch_size]
        raw = alignment_infer(backend, CONDITION_ALIGNMENT_TASK, {'existing_conditions': [
            {'condition_id': cid, 'canonical': {'name': c.name, 'definition': c.definition, 'value_type': c.value_type}}
            for cid, c in sorted(canonical.items())], 'incoming_conditions': [
            {'incoming_id': c.condition_id, 'name': c.name, 'definition': c.definition,
             'value_type': c.value_type, 'detector': c.detector} for c in chunk]})
        require(set(raw) == {'assignments'}, 'Invalid condition alignment response')
        items = raw['assignments']
        ids = [x['incoming_id'] for x in items]
        require(len(ids) == len(set(ids)) and set(ids) == {c.condition_id for c in chunk}, 'Condition alignment lost/invented predicates')
        by_id, anchors = {c.condition_id: c for c in chunk}, set(canonical)
        for item in items:
            original, desc = by_id[item['incoming_id']], item['canonical']
            require(set(desc) == {'name', 'definition', 'value_type'}, 'Invalid canonical condition')
            require(desc['value_type'] in ('boolean', 'number', 'string'), 'Invalid canonical condition type')
            existing_id = item['existing_condition_id']
            if existing_id is not None:
                require(existing_id in anchors, 'Unknown condition anchor')
                target = canonical[existing_id]
            else:
                candidate = Condition('', desc['name'], desc['definition'], {'kind': 'semantic'},
                                      {k: list(v) for k, v in original.evidence_by_case.items()}, desc['value_type'],
                                      'Semantic generalization; re-evaluate on every before context.')
                candidate.condition_id = stable_id('condition', {'definition': candidate.definition.strip(),
                    'detector': candidate.detector, 'value_type': candidate.value_type})
                target = canonical.setdefault(candidate.condition_id, candidate)
            for cid, spans in original.evidence_by_case.items():
                target.evidence_by_case.setdefault(cid, []).extend(s for s in spans if s not in target.evidence_by_case.get(cid, []))
            target.validate(catalog.pairs, catalog.allowlist)
            aliases[original.condition_id] = target.condition_id
            catalog.alignment_history.append({**item, 'assigned_condition_id': target.condition_id,
                'source_value_type': original.value_type, 'canonical_value_type': target.value_type,
                'type_decision': 'qwen_canonical_meaning_re_evaluated_before_only'})
        import logging
        logging.getLogger(__name__).info('Condition alignment progress: %d/%d source conditions, %d canonical conditions',
            offset + len(chunk), len(originals), len(canonical))
    catalog.conditions = canonical
    return aliases


def alignment_infer(backend, task, payload):
    """Use the original full comparison when it fits; otherwise examine every anchor window."""
    fits = getattr(backend, 'prompt_fits', None)
    if fits is None or fits(task, payload, repair_reserve=True):
        return _alignment_checked(backend, task, payload)
    is_action = 'incoming_methods' in payload
    incoming_key = 'incoming_methods' if is_action else 'incoming_conditions'
    existing_key = 'existing_groups' if is_action else 'existing_conditions'
    reference_key = 'existing_group_id' if is_action else 'existing_condition_id'
    anchor_key = 'group_id' if is_action else 'condition_id'
    incoming, anchors = payload[incoming_key], payload[existing_key]
    if len(incoming) > 1:
        assignments = []
        for item in incoming:
            result = alignment_infer(backend, task, {**payload, incoming_key: [item]})
            assignments.extend(result['assignments'])
        return {'assignments': assignments}
    require(bool(incoming) and bool(anchors), 'Single alignment item exceeds context; no source text was truncated')
    # Pack all anchors in order. No retrieval filter or fixed taxonomy is introduced.
    windows = []
    offset = 0
    while offset < len(anchors):
        low, high, size = 1, len(anchors) - offset, 0
        while low <= high:
            middle = (low + high) // 2
            candidate = {**payload, existing_key: anchors[offset:offset + middle]}
            if fits(task, candidate, repair_reserve=True):
                size = middle
                low = middle + 1
            else:
                high = middle - 1
        require(size > 0, 'Single alignment anchor exceeds context; no source text was truncated')
        windows.append(anchors[offset:offset + size])
        offset += size
    decisions = [_alignment_checked(backend, task, {**payload, existing_key: window})['assignments'][0]
                 for window in windows]
    selected = {d[reference_key] for d in decisions if d[reference_key] is not None}
    if not selected:
        return {'assignments': [decisions[0]]}
    if len(selected) == 1:
        return {'assignments': [next(d for d in decisions if d[reference_key] in selected)]}
    candidates = [anchor for anchor in anchors if anchor[anchor_key] in selected]
    require(len(candidates) < len(anchors), 'Context cannot compare two alignment anchors; no source was truncated')
    return alignment_infer(backend, task, {**payload, existing_key: candidates})


def _resolve_batch_references(raw, payload, is_action):
    """Resolve the model's explicit same-batch links, without guessing semantic matches."""
    from copy import deepcopy
    result = deepcopy(raw)
    incoming_key = 'incoming_methods' if is_action else 'incoming_conditions'
    existing_key = 'existing_groups' if is_action else 'existing_conditions'
    anchor_key = 'group_id' if is_action else 'condition_id'
    reference_key = 'existing_group_id' if is_action else 'existing_condition_id'
    order = {row['incoming_id']: i for i, row in enumerate(payload[incoming_key])}
    rows = {row['incoming_id']: row for row in result['assignments']}
    anchors = {row[anchor_key]: row['canonical'] for row in payload[existing_key]}
    for row in result['assignments']:
        reference = row[reference_key]
        if reference is None or reference in anchors or reference in rows:
            continue
        matches = [key for key, canonical in anchors.items() if canonical == row['canonical']]
        require(len(matches) == 1, 'Unknown semantic anchor; no unique exact supplied description')
        row.update({reference_key: matches[0], 'model_reference_id': reference,
                    'reference_resolution': 'unique_exact_supplied_description'})
    def resolve(start):
        trail = []
        current = start
        while True:
            if current in anchors:
                return current, anchors[current], 'existing_anchor'
            require(current in rows, 'Unknown semantic anchor')
            if current in trail:
                # The model explicitly linked the cycle as equivalent. Select only its
                # displayed wording deterministically; no semantic similarity is inferred.
                cycle = trail[trail.index(current):]
                representative = min(cycle, key=lambda key: order[key])
                return None, rows[representative]['canonical'], 'same_batch_cycle'
            trail.append(current)
            row = rows[current]
            target = row[reference_key]
            if target is None or target == current:
                return None, row['canonical'], 'same_batch_new'
            current = target
    # Resolve against the untouched graph before replacing any references.
    resolutions = {iid: resolve(iid) for iid in rows}
    for iid, row in rows.items():
        original_reference = row[reference_key]
        if original_reference is not None and original_reference not in anchors:
            target, canonical, source = resolutions[iid]
            row.update({reference_key: target, 'canonical': deepcopy(canonical),
                        'model_reference_id': original_reference, 'reference_resolution': source})
    return result


def _alignment_checked(backend, task, payload):
    is_action = 'incoming_methods' in payload
    incoming = payload['incoming_methods' if is_action else 'incoming_conditions']
    expected = {x['incoming_id'] for x in incoming}
    anchor_key = 'group_id' if is_action else 'condition_id'
    anchors = {x[anchor_key] for x in payload['existing_groups' if is_action else 'existing_conditions']}
    reference_key = 'existing_group_id' if is_action else 'existing_condition_id'
    def decode(raw):
        require(set(raw) == {'assignments'} and isinstance(raw['assignments'], list), 'Invalid semantic assignments')
        raw = deepcopy(raw)  # Do not rewrite a cached model response.
        returned = [x['incoming_id'] for x in raw['assignments']]
        require(len(returned) == len(set(returned)) and set(returned) == expected, 'Missing/duplicate/invented semantic assignments')
        for item in raw['assignments']:
            required = {'family', 'method', 'target_feature', 'scope', 'semantic'} if is_action else {'name', 'definition', 'value_type'}
            require(isinstance(item['canonical'], dict) and set(item['canonical']) == required, 'Invalid canonical fields')
            canonical = item['canonical']
            text_fields = required - {'semantic'} if is_action else {'name', 'definition'}
            require(all(isinstance(canonical[k], str) and canonical[k].strip() for k in text_fields), 'Empty/non-text canonical description')
            if is_action:
                iid = item['incoming_id']
                require(type(canonical['semantic']) is bool,
                        f'Action {iid}: canonical.semantic must be a JSON boolean')
                require(isinstance(item['uncertainty_flags'], list)
                        and all(isinstance(flag, str) for flag in item['uncertainty_flags']),
                        f'Action {iid}: uncertainty_flags must be a list of strings')
                if canonical['semantic'] is False:
                    source = next(x for x in incoming if x['incoming_id'] == iid)
                    require(source.get('signature', {}).get('semantic') is True,
                            f'Action {iid}: cannot inherit interpreted provenance from a non-interpreted source')
                    # This flag identifies model interpretation versus the local diff
                    # fallback. It is not a judgment of whether an edit changes meaning.
                    # Keep the model decision and its original flag in the audit trail.
                    item['model_semantic_value'] = False
                    item['semantic_resolution'] = 'inherited_interpreted_action_provenance'
                    canonical['semantic'] = True
            else:
                require(canonical['value_type'] in ('boolean', 'number', 'string'), 'Invalid canonical condition type')
        resolved = _resolve_batch_references(raw, payload, is_action)
        if is_action:
            action_anchors = {row['group_id']: row['canonical'] for row in payload['existing_groups']}
            require(all(row['canonical']['semantic'] is True and
                        (row[reference_key] is None or action_anchors[row[reference_key]]['semantic'] is True)
                        for row in resolved['assignments']),
                    'Action anchor mixes interpreted and mechanical provenance')
        return resolved
    if hasattr(backend, 'reserve_work_calls'):
        backend.reserve_work_calls((backend.config.retries + 1) ** 2, 'semantic_alignment_batch')
    return checked_infer(backend, task, payload, decode)
