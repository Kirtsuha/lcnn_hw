import numpy as np

from src.metrics.base_metric import BaseMetric


def compute_eer(bonafide_scores, spoof_scores):
    """Return EER and threshold; higher scores must mean bonafide."""
    bonafide_scores = np.asarray(bonafide_scores)
    spoof_scores = np.asarray(spoof_scores)
    if not bonafide_scores.size or not spoof_scores.size:
        raise ValueError("EER requires both bonafide and spoof scores")

    scores = np.concatenate((bonafide_scores, spoof_scores))
    labels = np.concatenate(
        (np.ones(bonafide_scores.size), np.zeros(spoof_scores.size))
    )
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    false_rejections = np.concatenate(
        ([0.0], np.cumsum(labels) / bonafide_scores.size)
    )
    false_acceptances = np.concatenate(
        (
            [1.0],
            (
                spoof_scores.size
                - (np.arange(1, scores.size + 1) - np.cumsum(labels))
            )
            / spoof_scores.size,
        )
    )
    thresholds = np.concatenate(([scores[order[0]] - 1e-3], scores[order]))
    index = np.argmin(np.abs(false_rejections - false_acceptances))
    return (
        float((false_rejections[index] + false_acceptances[index]) / 2),
        float(thresholds[index]),
    )


class EqualErrorRate(BaseMetric):
    aggregate = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self._scores = []
        self._labels = []

    def update(self, logits, labels, **batch):
        self._scores.append(logits[:, 1].detach().cpu().numpy())
        self._labels.append(labels.detach().cpu().numpy())

    def compute(self):
        scores = np.concatenate(self._scores)
        labels = np.concatenate(self._labels)
        return 100.0 * compute_eer(scores[labels == 1], scores[labels == 0])[0]

    def __call__(self, logits, labels, **batch):
        self.update(logits, labels)
        return self.compute()
