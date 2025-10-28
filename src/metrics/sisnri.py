import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.metrics.base_metric import BaseMetric


class SISNRiMetric(BaseMetric):
    """
    SI-SNRi Metric
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
            si_snri: SI-SNR improvement
        """
        si_snr_est = self.si_snr(est_source, true_source)
        si_snr_mix = self.si_snr(mixture, true_source)
        return (si_snr_est - si_snr_mix).item()

    @staticmethod
    def si_snr(source, target):
        """
        Compute scale-invariant SNR between source and target using torchmetrics.
        """
        si_snr = ScaleInvariantSignalNoiseRatio()
        return si_snr(source, target)
