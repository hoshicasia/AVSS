import torch
from torchmetrics.audio import SignalDistortionRatio

from src.metrics.base_metric import BaseMetric


class SDRiMetric(BaseMetric):
    """
    SDRi Metric
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
            sdr: SDR improvement
        """
        sdr_est = self.sdr(est_source, true_source)
        sdr_mix = self.sdr(mixture, true_source)
        return (sdr_est - sdr_mix).item()

    @staticmethod
    def sdr(source, target):
        """
        Compute scale-invariant SDR between source and target using torchmetrics.
        """
        sdr = SignalDistortionRatio()
        return sdr(source, target)
