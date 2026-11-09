"""Future revision preparation. No source is configured/loaded at import."""
from hashlib import sha256
from theory_discovery.evaluation.evaluation_contracts import EvaluationContext
from theory_discovery.core.workflow_contracts import (
    CaseIdentity, Period, PredictionCase, OutcomeCase, ChronologicalSplitManifest, MetadataItem,
)
from theory_discovery.core.revision_history import load_local_revisions, SENTENCE_PAIR_UNITS, unique_revision_metadata
from theory_discovery.core.chronological_split import split_editor_revisions
from theory_discovery.core.prediction_units import (RevisionBefore, BuiltinBeforeTargeter, validate_units, align_outcomes,
                                provided_sentence_unit, AlignedOutcome)
from theory_discovery.core.stage_interfaces import (
    DataPreparationInput, DataPreparationOutput, SplitArtifact, EarlyRecords,
    EarlyBeforeRecords, LaterRecords, OutcomeRecords, NeutralActionCatalog,
)
from theory_discovery.core.schemas import require, stable_id


def prepare_records(inputs: DataPreparationInput, revisions, targeter=None) -> DataPreparationOutput:
    split = split_editor_revisions(revisions, inputs.editor, inputs.split, inputs.source.ordering)
    provided = inputs.unit.kind == "provided_sentence_pair"
    require(not provided or targeter is None, "Provided pairs cannot be resegmented by a targeter")
    if not provided:
        targeter = targeter or BuiltinBeforeTargeter(inputs.unit.kind, inputs.unit.version)
    early_revisions = unique_revision_metadata(split.early)
    later_revisions = unique_revision_metadata(split.later)
    early, early_before, later, outcomes = [], [], [], []
    identities = []
    for period, records in ((Period.EARLY, split.early), (Period.LATER, split.later)):
        for row in records:
            r = row.revision
            context = tuple(field for field in row.pre_context
                            if field.name in inputs.unit.pre_context_allowlist)
            before = RevisionBefore(r.editor, r.revision_id, r.parent_revision_id,
                                    row.before_text, row.parent_timestamp, context)
            require((row.text_unit in SENTENCE_PAIR_UNITS) == provided,
                    "Provided pairs require provided_sentence_pair; revision modes require full revisions")
            units = ((provided_sentence_unit(before, row.sentence_pair_id, inputs.unit.version),)
                     if provided else targeter.enumerate_units(before))
            validate_units(before, units)
            require(all(unit.policy_version == inputs.unit.version for unit in units),
                    "Targeter version differs from central configuration")
            fixed = []
            for unit in units:
                identity = CaseIdentity.create(r.editor, r.revision_id, unit.prediction_unit_id,
                                               inputs.unit.version, period)
                text = before.before_text[unit.start:unit.end]
                case = PredictionCase(identity, text, sha256(text.encode()).hexdigest(),
                                      before.parent_timestamp, before.pre_context)
                fixed.append(case)
                identities.append(identity)
            # Only this separate outcome path sees the target After.
            aligned = ((AlignedOutcome(units[0].prediction_unit_id, row.after_text, "ALIGNED"),)
                       if provided else align_outcomes(row.before_text, row.after_text, units))
            for unit, case, alignment in zip(units, fixed, aligned):
                provenance = r.provenance + (r.before_reference, r.after_reference)
                pair = OutcomeCase(case.identity, case.before_text, alignment.after_text, provenance,
                                   alignment.alignment_status,
                                   (MetadataItem("text_unit", "PROVIDED_EDITED_SENTENCE_PAIR" if provided else
                                      "BEFORE_TARGETED_SENTENCE" if inputs.unit.kind == "sentence_v1" else "WHOLE_REVISION"),
                                    MetadataItem("before_start", unit.start),
                                    MetadataItem("targeter_kind", inputs.unit.kind)))
                if period is Period.EARLY:
                    early.append(pair)
                    early_before.append(case)
                else:
                    later.append(case)
                    outcomes.append(pair)
    source_refs = tuple({(p.source_id, p.json_pointer): p
                         for row in split.early + split.later for p in row.revision.provenance}.values())
    manifest = ChronologicalSplitManifest(
        inputs.editor, split.cutoff,
        tuple(r.revision_id for r in early_revisions),
        tuple(r.revision_id for r in later_revisions),
        tuple(identities),
        (MetadataItem("cutoff_mode", "TIME" if inputs.split.cutoff_utc else "RATIO"),
         MetadataItem("requested_early_ratio", inputs.split.early_ratio),
         MetadataItem("ordering", inputs.source.ordering),
         MetadataItem("unit_kind", inputs.unit.kind), MetadataItem("unit_version", inputs.unit.version),
         MetadataItem("task", "ACTION_GIVEN_EDITED_SENTENCE" if provided else "BEFORE_TARGETED_UNITS")),
        source_refs,
        tuple(MetadataItem(r.revision_id, r.timestamp) for r in early_revisions + later_revisions))
    return DataPreparationOutput(
        EvaluationContext(early_revisions + later_revisions, len(early_revisions),
                          len(later_revisions), split.cutoff, inputs.unit.version, source_refs),
        SplitArtifact(manifest), EarlyRecords(tuple(early), early_revisions),
        EarlyBeforeRecords(tuple(early_before)), LaterRecords(tuple(later)), OutcomeRecords(tuple(outcomes)),
        NeutralActionCatalog(stable_id("neutral_catalog", [a.to_dict() for a in inputs.supplied_vocabulary]),
                             inputs.supplied_vocabulary))


def run(inputs: DataPreparationInput) -> DataPreparationOutput:
    """Explicit future handler. Empty source configuration fails before any read."""
    return prepare_records(inputs, load_local_revisions(inputs.source))
