"""Score held-out predictions and compare observed action rates."""


def any_action_hit(prediction, observed):
    status, predicted, conditions = prediction
    if status != "PREDICTED" or observed is None:
        return None
    if predicted & observed:
        return True
    return None if None in conditions.values() else False


def compare_rates(earlier, later, condition_id, action_id):
    """Each cohort contains one (condition measurements, observed actions) row per sample."""
    def count(cohort):
        usable = [actions for conditions, actions in cohort
                  if conditions[condition_id] is True and actions is not None]
        return sum(action_id in actions for actions in usable), len(usable)

    old_count, old_total = count(earlier)
    new_count, new_total = count(later)
    if min(old_total, new_total) < 5:
        return (old_count, old_total), (new_count, new_total), None
    delta_pp = (new_count / new_total - old_count / old_total) * 100
    return (old_count, old_total), (new_count, new_total), delta_pp
