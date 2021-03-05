import torch

from lhotse import CutSet


class Standardize:
    def __init__(self, cuts: CutSet):
        stats = cuts.compute_global_feature_stats()
        self.norm_means = stats["norm_means"]
        self.norm_stds = stats["norm_stds"]

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.norm_means) / self.norm_stds

    def inverse(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.norm_stds + self.norm_means
