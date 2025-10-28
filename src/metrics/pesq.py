import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

from src.metrics.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    """
    PESQ Metric
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
            pesq: PESQ score
        """
        pesq_est = self.pesq(est_source, true_source)
        return pesq_est.item()

    @staticmethod
    def pesq(source, target):
        """
        Compute PESQ between source and target using torchmetrics.
        """
        pesq = PerceptualEvaluationSpeechQuality()
        return pesq(source, target)
