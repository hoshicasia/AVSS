import torch
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    """
    STOI Metric
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, est_source, true_source, mixture):
        """
        Args:
            est_source: Estimated source
            true_source: True source
            mixture: Mixture signal
        Returns:
            stoi: STOI score
        """
        stoi_est = self.stoi(est_source, true_source)
        return stoi_est.item()

    @staticmethod
    def stoi(source, target):
        """
        Compute STOI between source and target using torchmetrics.
        """
        stoi = ShortTimeObjectiveIntelligibility()
        return stoi(source, target)
