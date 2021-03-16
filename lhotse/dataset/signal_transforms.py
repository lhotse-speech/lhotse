from typing import Optional

import torch

from lhotse import CutSet
from lhotse.utils import Pathlike


class GlobalMVN(torch.nn.Module):
    """Apply global mean and variance normalization"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("norm_means", torch.zeros(feature_dim))
        self.register_buffer("norm_stds", torch.ones(feature_dim))

    @classmethod
    def from_cuts(cls, cuts: CutSet, max_cuts: Optional[int] = None) -> "GlobalMVN":
        stats = cuts.compute_global_feature_stats(max_cuts)
        stats = {name: torch.as_tensor(value) for name, value in stats.items()}
        feature_dim, = stats["norm_means"].shape
        global_mvn = cls(feature_dim)
        global_mvn.load_state_dict(stats)
        return global_mvn

    @classmethod
    def from_file(cls, stats_file: Pathlike) -> "GlobalMVN":
        stats = torch.load(stats_file)
        feature_dim, = stats["norm_means"].shape
        global_mvn = cls(feature_dim)
        global_mvn.load_state_dict(stats)
        return global_mvn

    def to_file(self, stats_file: Pathlike):
        torch.save(self.state_dict(), stats_file)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.norm_means) / self.norm_stds

    def inverse(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.norm_stds + self.norm_means
