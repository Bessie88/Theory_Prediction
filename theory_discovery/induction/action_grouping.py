"""Stage 2: deterministic operation hierarchy, optional explicit disambiguation."""
from theory_discovery.core.llm import PLAIN_LANGUAGE_INSTRUCTION

STAGE_INSTRUCTION = PLAIN_LANGUAGE_INSTRUCTION
import re
import unicodedata
from theory_discovery.core.schemas import ActionGroup, require, stable_id


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def action_signature(action):
    # No original sentence, target word, condition, or topic participates in grouping.
    return {'family': action.family, 'method': normalize(action.method),
            'target_feature': normalize(action.target_feature), 'scope': action.scope,
            'semantic': action.semantic}


def group_actions(observations, resolver=None):
    """resolver.resolve(refs) may return grounded action_id -> canonical signature.

    Meaning-equivalent descriptions can differ in family/scope wording. Original
    action objects retain those distinctions. Mechanical and semantic status cannot mix.
    """
    refs = [{'case_id': o.case_id, 'action_id': a.action_id, 'signature': action_signature(a), 'transformation': a.transformation}
            for o in observations for a in o.actions]
    resolved = resolver.resolve(refs) if resolver is not None else {}
    require(set(resolved) <= {r['action_id'] for r in refs}, 'Resolver invented action references')
    buckets = {}
    for obs in observations:
        for action in obs.actions:
            signature = action_signature(action)
            canonical = resolved.get(action.action_id, signature)
            require(set(canonical) == set(signature), 'Invalid canonical action signature')
            require(canonical['semantic'] == signature['semantic'], 'Resolver mixed interpreted and mechanical actions')
            require(bool(canonical['method']) and bool(canonical['target_feature']), 'Empty action normalization')
            gid = stable_id('group', canonical)
            bucket = buckets.setdefault(gid, {'signature': canonical, 'members': {}, 'refs': [], 'variations': set(), 'flags': set(), 'original_signatures': []})
            if signature not in bucket['original_signatures']:
                bucket['original_signatures'].append(signature)
            bucket['members'][obs.case_id] = obs
            bucket['refs'].append({'case_id': obs.case_id, 'action_id': action.action_id})
            bucket['variations'].add(action.method)
            bucket['flags'].update(action.uncertainty_flags)
            if resolver is not None:
                for decision in getattr(resolver, 'history', []):
                    if action.action_id in decision.get('source_action_ids', []):
                        bucket['flags'].update(decision['uncertainty_flags'])
    groups = []
    for gid, bucket in sorted(buckets.items()):
        members = bucket['members']
        representatives = [{'case_id': cid, 'before_text': members[cid].before_text, 'after_text': members[cid].after_text,
                            'action_refs': [r for r in bucket['refs'] if r['case_id'] == cid]} for cid in sorted(members)[:3]]
        groups.append(ActionGroup(gid, bucket['signature'], sorted(members), bucket['refs'], representatives,
                                  len(members), sorted(bucket['variations']), sorted(bucket['flags']), bucket['original_signatures']))
    return groups


def has_action(observation, group):
    # Supports held-out observed actions without requiring their IDs in a training group.
    signatures = [group.normalized_action, *group.member_action_signatures]
    return observation.case_id in group.member_case_ids or any(action_signature(a) in signatures for a in observation.actions)


def align_to_frozen_groups(observations, frozen_groups, backend, batch_size=4):
    """Future held-out outcome annotation using frozen meanings; retain novel groups.

    Historical pairs are allowed only here for independent outcome annotation.
    Never invoke this operation in the before-only prediction path.
    """
    from .semantic_alignment import SemanticActionResolver
    resolver = SemanticActionResolver(backend, batch_size, initial_groups={g.group_id: g.normalized_action for g in frozen_groups})
    return group_actions(observations, resolver)
