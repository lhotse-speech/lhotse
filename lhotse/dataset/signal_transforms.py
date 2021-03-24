import random
from typing import Optional

import numpy as np
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
        stats = cuts.compute_global_feature_stats(max_cuts=max_cuts)
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


class SpecAugment(torch.nn.Module):
    """
    SpecAugment performs three augmentations:
    - time warping of the feature matrix
    - masking of ranges of features (frequency bands)
    - masking of ranges of frames (time)

    The current implementation works with batches, but processes each example separately
    in a loop rather than simultaneously to achieve different augmentation parameters for
    each example.
    """

    def __init__(
            self,
            time_warp_factor: Optional[int] = 80,
            num_feature_masks: int = 1,
            features_mask_size: int = 13,
            num_frame_masks: int = 1,
            frames_mask_size: int = 70,
            p=1.0,
    ):
        super().__init__()
        self.time_warp_factor = time_warp_factor
        self.num_feature_masks = num_feature_masks
        self.features_mask_size = features_mask_size
        self.num_frame_masks = num_frame_masks
        self.frames_mask_size = frames_mask_size
        self.p = p

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes SpecAugment for a matrix or a batch of matrices.

        :param features: features tensor of shape (T, F), or a batch of them with shape (B, T, F).
        :return: a tensor of shape (T, F), or a batch of them with shape (B, T, F)
        """
        # A single sample rather than a batch.
        if len(features.shape) == 2:
            return self._forward_single(features)

        # Loop over different examples in the batch to get different
        # augmentation for each example.
        for example_idx in range(features.size(0)):
            features[example_idx] = self._forward_single(features[example_idx])
        return features

    def _forward_single(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to a single feature matrix of shape (T, F).
        """
        if random.random() > self.p:
            # Randomly choose whether this transform is applied
            return features
        from torchaudio.functional import mask_along_axis
        mean = features.mean()
        if self.time_warp_factor is not None and self.time_warp_factor >= 1:
            features = time_warp(features, factor=self.time_warp_factor)
        for _ in range(self.num_feature_masks):
            features = mask_along_axis(
                features.unsqueeze(0),
                mask_param=self.features_mask_size,
                mask_value=mean,
                axis=2
            ).squeeze(0)
        for _ in range(self.num_frame_masks):
            features = mask_along_axis(
                features.unsqueeze(0),
                mask_param=self.frames_mask_size,
                mask_value=mean,
                axis=1
            ).squeeze(0)
        return features


def time_warp(features: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Time warping as described in the SpecAugment paper.
    Implementation based on Espresso:
    https://github.com/freewym/espresso/blob/master/espresso/tools/specaug_interpolate.py#L51

    :param features: input tensor of shape ``(B, T, F)``
    :param factor: time warping parameter.
    :return: a warped tensor of shape ``(B, T, F)``
    """
    t = features.size(1)
    if t - factor <= factor + 1:
        return features
    center = np.random.randint(factor + 1, t - factor)
    warped = np.random.randint(center - factor, center + factor + 1)
    if warped == center:
        return features
    features = features.unsqueeze(0).unsqueeze(0)
    left = torch.nn.functional.interpolate(
        features[:, :, :center, :], size=(warped, features.size(3)),
        mode="bicubic", align_corners=False,
    )
    right = torch.nn.functional.interpolate(
        features[:, :, center:, :], size=(t - warped, features.size(3)),
        mode="bicubic", align_corners=False,
    )
    return torch.cat((left, right), dim=2).squeeze(0).squeeze(0)
