"""Before-only comparison to the preserved per-feature reference arm; not gold."""
from theory_discovery.core.workflow_contracts import ObservationStatus
from theory_discovery.core.stage_interfaces import FeatureBatch
from theory_discovery.integrated.applicability import run as apply
from theory_discovery.core.stage_interfaces import ApplicabilityInput,TheoryPredictionInput,NPOVPredictionInput
from theory_discovery.integrated.theory_predictor import run as predict_theory
from theory_discovery.integrated.npov import predict as predict_policy

def compare(study,old,new):
    left={o.feature_id:o for o in old.features.observations}
    right={o.feature_id:o for o in new.features.observations}
    both=[k for k in left if left[k].status is ObservationStatus.KNOWN and right[k].status is ObservationStatus.KNOWN]
    changed=[k for k in both if left[k].extracted_value!=right[k].extracted_value]
    result={"known_overlap":len(both),"value_disagreements":changed,
            "reference_known_new_unknown":[k for k in left if left[k].status is ObservationStatus.KNOWN and right[k].status is not ObservationStatus.KNOWN],
            "prediction_comparison":{},"reference_is_not_gold":True}
    for name,output in (("reference",old),("open_coded",new)):
        f=output.features;a=apply(ApplicabilityInput(f,study,True))
        t=predict_theory(TheoryPredictionInput(f,a.theory_applicability,study,True))
        p=predict_policy(NPOVPredictionInput(f,a.npov_applicability,study,True))
        result["prediction_comparison"][name]={"theory":t.theory_predictions.to_dict(),"npov":p.npov_predictions.to_dict()}
    return result
