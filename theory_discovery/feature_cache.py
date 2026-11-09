"""Content-addressed feature cache adapter over existing shared storage infrastructure."""
from typing import Protocol
from .workflow_contracts import FeatureObservation
from .storage import Cache


class FeatureCache(Protocol):
    def get_or_compute(self, key: dict, compute) -> FeatureObservation:
        ...


class SharedFeatureCache:
    """Side cache for observations, not a separate experiment/artifact registry."""
    def __init__(self, directory):
        self.cache = Cache(directory)

    def get_or_compute(self, key, compute):
        raw = self.cache.get_or_compute("integrated_feature_observations_v1", key, compute)
        return FeatureObservation.from_dict(raw)


class NoFeatureCache:
    def get_or_compute(self, key, compute):
        return compute()
