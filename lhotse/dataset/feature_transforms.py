import torch

from typing import Union

from lhotse import CutSet
from lhotse.utils import Pathlike


class GlobalMVN(torch.nn.Module):
    """Apply global mean and variance normalization"""

    def __init__(self, norm_means, norm_stds):
        super().__init__()
        self.register_buffer("norm_means", torch.as_tensor(norm_means))
        self.register_buffer("norm_stds", torch.as_tensor(norm_stds))

    @classmethod
    def from_cuts(cls, cuts: CutSet) -> "GlobalMVN":
        stats = cuts.compute_global_feature_stats()
        return cls(stats["norm_means"], stats["norm_stds"])

    @classmethod
    def from_file(cls, stats_file: Pathlike) -> "GlobalMVN":
        stats = torch.load(stats_file)
        return cls(stats["norm_means"], stats["norm_stds"])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.norm_means) / self.norm_stds

    def inverse(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.norm_stds + self.norm_means
