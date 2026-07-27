import torch

from src.metrics.base_metric import BaseMetric


class ClassificationAccuracy(BaseMetric):
    """Adapter between the template metric API and a torchmetrics metric."""

    def __init__(self, metric, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, logits, labels, **batch):
        return self.metric(logits.argmax(dim=-1), labels)
