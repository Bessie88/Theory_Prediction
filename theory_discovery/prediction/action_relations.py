"""Code-only overlap checks, used only after concrete edits have been proposed."""
from theory_discovery.core.schemas import require, stable_id
from .rule_first_repairs import canonical_operation


def operation_identity(row):
    return stable_id("exact_validated_operation", canonical_operation(row))


def _target(row, before):
    target=row["target"];start,end=target["start"],target["end"]
    require(type(start) is int and type(end) is int and 0<=start<=end<=len(before),"Invalid operation target")
    require(target["side"]=="before" and before[start:end]==target["quote"],"Target must match exact Before")
    return start,end


def overlap_conflicts(candidates, before):
    """No pair LLM. Disjoint targets need no check; ambiguous overlap stays Unknown."""
    targets=[_target(row,before) for row in candidates];result=[]
    for i,left in enumerate(candidates):
        a,b=targets[i]
        for j in range(i+1,len(candidates)):
            right=candidates[j];c,d=targets[j]
            overlaps=(max(a,c)<min(b,d) if a<b and c<d else
                      a==c if a==b and c==d else c<=a<=d if a==b else a<=c<=b)
            if not overlaps:continue
            left_id,right_id=operation_identity(left),operation_identity(right)
            relation,reason="UNKNOWN","Overlapping edits cannot be resolved from their exact splice parameters."
            if left_id==right_id:
                relation,reason="COMPATIBLE","Identical concrete operation; execute once and retain all sources."
            else:
                lf=left["frozen_action"].get("family");rf=right["frozen_action"].get("family")
                lr="" if lf=="deletion" and left.get("replacement") is None else left.get("replacement")
                rr="" if rf=="deletion" and right.get("replacement") is None else right.get("replacement")
                if (a,b)==(c,d) and a<b and lf in ("deletion","replacement") and rf in ("deletion","replacement") and isinstance(lr,str) and isinstance(rr,str):
                    if lr==rr:relation,reason="COMPATIBLE","Same exact replacement; execute once."
                    else:relation,reason="INCOMPATIBLE","Different replacements require the same original span; they cannot both be applied."
            result.append(dict(left_action_id=left.get("action_id",left.get("source_action_id")),
                right_action_id=right.get("action_id",right.get("source_action_id")),
                left_operation_id=left_id,right_operation_id=right_id,relation=relation,
                scope="OVERLAPPING_CONCRETE_TARGETS_ONLY",origin="CODE_ONLY",reason=reason))
    return result
