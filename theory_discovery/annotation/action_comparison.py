"""One frozen label comparison protocol; no inference, evaluation run or taxonomy."""
from dataclasses import dataclass
from theory_discovery.core.workflow_contracts import Record, ActionDefinition, ActionInstance, PredictionStatus, AnnotationStatus
from theory_discovery.core.schemas import require, stable_id
import json
import re

NOVEL_ACTION_VERSION = "open-action-v1"


def _label_text(value):
    return re.sub(r"[\s_-]+", " ", value.strip().casefold()) if isinstance(value, str) else value


def action_meaning_key(definition):
    return tuple(_label_text(getattr(definition, key)) for key in
                 ("family", "method", "target_feature", "direction", "scope"))


def novel_action_id(definition):
    return stable_id("novel_action", action_meaning_key(definition))


def novel_action_definition(action):
    """Novel labels carry their canonical definition, not a private unexplained ID."""
    if not action.action_id.startswith("novel_action_"):
        return None
    try:
        definition = ActionDefinition.from_dict(json.loads(action.description))
        require(definition.version == NOVEL_ACTION_VERSION and definition.action_id == action.action_id ==
                novel_action_id(definition), "Novel Action identity/definition differs")
        return definition
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Novel Action needs a canonical semantic definition") from exc


def permits_novel_actions(protocol):
    return protocol.comparison.other_rule == "CONFIRMED_NEW_IS_MISS"


@dataclass(frozen=True)
class ActionComparisonProtocol(Record):
    version: str = "action-set-comparison-v2"
    unit: str = "ACTION_ID_SET_PER_CASE"
    partial_rule: str = "NONEMPTY_SET_INTERSECTION"
    other_rule: str = "CONFIRMED_NEW_IS_MISS"
    target_rule: str = "LABEL_LEVEL_ONLY"
    established_before_later: bool = False

    def validate(self):
        require((self.version, self.other_rule) in (("action-set-comparison-v1", "UNRESOLVED"),
                ("action-set-comparison-v2", "CONFIRMED_NEW_IS_MISS")) and
                (self.unit, self.partial_rule, self.target_rule) ==
                ("ACTION_ID_SET_PER_CASE", "NONEMPTY_SET_INTERSECTION", "LABEL_LEVEL_ONLY"),
                "Unsupported comparison protocol")


@dataclass(frozen=True)
class ActionGuideline(Record):
    action_id: str
    inclusion: tuple[str, ...]
    exclusion: tuple[str, ...]
    evidence_guidance: tuple[str, ...]


@dataclass(frozen=True)
class ActionProtocolSettings(Record):
    guidelines_version: str = "unconfigured"
    guidelines: tuple[ActionGuideline, ...] = ()
    general_guidelines: tuple[str, ...] = ()
    comparison: ActionComparisonProtocol = ActionComparisonProtocol()


@dataclass(frozen=True)
class FrozenActionProtocol(Record):
    catalog_version: str
    actions: tuple[ActionDefinition, ...]
    guidelines_version: str
    guidelines: tuple[ActionGuideline, ...]
    general_guidelines: tuple[str, ...]
    comparison: ActionComparisonProtocol
    protocol_hash: str

    def validate(self):
        ids = [a.action_id for a in self.actions]
        require(len(set(ids)) == len(ids) and "OTHER" in ids, "Common catalog must explicitly include OTHER")
        require(set(ids) == {g.action_id for g in self.guidelines} and
                len(ids) == len(self.guidelines), "One frozen guideline per common label required")
        require(self.general_guidelines and self.guidelines_version not in ("", "unconfigured") and
                self.comparison.established_before_later, "Freeze guidelines/comparison before Later use")


def protocol_hash(protocol):
    return stable_id("action_annotation_protocol", {
        k: v for k, v in protocol.to_dict().items() if k != "protocol_hash"})


def validate_protocol(protocol):
    require(protocol.protocol_hash == protocol_hash(protocol), "Action protocol changed")


@dataclass(frozen=True)
class ActionAgreement(Record):
    category: str
    predicted_ids: tuple[str, ...]
    observed_ids: tuple[str, ...]
    shared_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]
    reason: str


def compare_action_sets(predicted, observed, protocol):
    validate_protocol(protocol)
    allowed = {a.action_id for a in protocol.actions}
    left = {a.action_id for a in predicted}
    right = {a.action_id for a in observed}
    for action in (*predicted, *observed):
        require(action.action_id in allowed or
                (permits_novel_actions(protocol) and novel_action_definition(action) is not None),
                "Comparison cannot use an unexplained private label")
    common = left & right
    category = ("UNRESOLVED" if "OTHER" in left | right else
                "EXACT" if left == right else "PARTIAL" if common else "NO_MATCH")
    return ActionAgreement(category, tuple(sorted(left)), tuple(sorted(right)), tuple(sorted(common)),
        tuple(sorted(right - left)), tuple(sorted(left - right)),
        "OTHER does not identify a unique operation." if category == "UNRESOLVED" else
        "Frozen label-set comparison; no semantic intent or target-span equivalence inferred.")


def provisional_agreement(prediction, annotation, protocol):
    """Sampling support ONLY: agreement with unverified annotations is not correctness."""
    if prediction.prediction_status is not PredictionStatus.PREDICTED:
        return "UNRESOLVED"
    if annotation.annotation_status not in (AnnotationStatus.ANNOTATED, AnnotationStatus.NO_CHANGE):
        return "UNRESOLVED"
    require(annotation.protocol_hash == protocol.protocol_hash, "Annotation protocol mismatch")
    return compare_action_sets(prediction.predicted_actions, annotation.observed_actions, protocol).category


def compare_condition_actions(predicted_ids, observed_ids, *, condition=True, observed_complete=True, predicted_complete=True):
    """Primary any-hit score for ONE frozen Condition, never pooled across Conditions."""
    require(condition is None or type(condition) is bool,"Condition truth must be Boolean/Unknown")
    left=set(predicted_ids)-{"OTHER"};right=set(observed_ids)-{"OTHER"}
    require(all(isinstance(k,str) and k for k in left|right),"Action IDs must be nonempty strings")
    shared=left&right
    if condition is False:status,passed="NOT_APPLICABLE",None
    elif condition is None:status,passed="CONDITION_UNKNOWN",None
    elif shared:status,passed="PASS_ANY_ACTION",True
    elif not predicted_complete or "OTHER" in predicted_ids:status,passed="PREDICTED_ACTION_UNKNOWN",None
    elif not observed_complete or "OTHER" in observed_ids:status,passed="OBSERVED_ACTION_UNKNOWN",None
    elif not left:status,passed="NO_PREDICTED_ACTION",False
    else:status,passed="FAIL_NO_SHARED_ACTION",False
    return dict(status=status,passed=passed,shared_action_ids=sorted(shared),
        predicted_action_ids=sorted(left),observed_action_ids=sorted(right),
        extra_action_ids=sorted(left-right),missing_action_ids=sorted(right-left),
        scoring_unit="ONE_CONDITION_ONE_CASE",criterion="NONEMPTY_INTERSECTION")
