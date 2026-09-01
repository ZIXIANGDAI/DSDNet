import numpy as np


class BinarySegmentationMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def update(self, prediction, target):
        valid = target != 255
        prediction = prediction[valid].astype(bool)
        target = target[valid].astype(bool)
        self.tp += np.logical_and(prediction, target).sum()
        self.fp += np.logical_and(prediction, ~target).sum()
        self.fn += np.logical_and(~prediction, target).sum()

    def compute(self):
        eps = 1e-8
        return {
            "dice": float(2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)),
            "iou": float(self.tp / (self.tp + self.fp + self.fn + eps)),
            "recall": float(self.tp / (self.tp + self.fn + eps)),
            "precision": float(self.tp / (self.tp + self.fp + eps)),
        }
