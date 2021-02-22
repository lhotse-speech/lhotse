from typing import Iterable, Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from lhotse import CutSet
from lhotse.cut import MixedCut


def collate_features(cuts: CutSet) -> torch.Tensor:
    """
    Load features for all the cuts and return them as a batch in a torch tensor.
    The output shape is ``(batch, time, features)``.
    The cuts will be padded with silence if necessary.
    """
    assert all(cut.has_features for cut in cuts)
    cuts = maybe_pad(cuts)
    first_cut = next(iter(cuts))
    features = torch.empty(len(cuts), first_cut.num_frames, first_cut.num_features)
    for idx, cut in enumerate(cuts):
        features[idx] = torch.from_numpy(cut.load_features())
    return features


def collate_audio(cuts: CutSet) -> torch.Tensor:
    """
    Load audio samples for all the cuts and return them as a batch in a torch tensor.
    The output shape is ``(batch, time)``.
    The cuts will be padded with silence if necessary.
    """
    assert all(cut.has_recording for cut in cuts)
    cuts = maybe_pad(cuts)
    first_cut = next(iter(cuts))
    audio = torch.empty(len(cuts), first_cut.num_samples)
    for idx, cut in enumerate(cuts):
        audio[idx] = torch.from_numpy(cut.load_audio()[0])
    return audio


def collate_multi_channel_features(cuts: CutSet) -> torch.Tensor:
    """
    Load features for all the cuts and return them as a batch in a torch tensor.
    The cuts have to be of type ``MixedCut`` and their tracks will be interpreted as individual channels.
    The output shape is ``(batch, channel, time, features)``.
    The cuts will be padded with silence if necessary.
    """
    assert all(cut.has_features for cut in cuts)
    assert all(isinstance(cut, MixedCut) for cut in cuts)
    cuts = maybe_pad(cuts)
    # Output tensor shape: (B, C, T, F) -> (batch_size, num_channels, num_frames, num_features)
    first_cut = next(iter(cuts))
    # TODO: make MixedCut more friendly to use with multi channel audio;
    #  discount PaddingCuts in "tracks" when specifying the number of channels
    features = torch.empty(len(cuts), len(first_cut.tracks), first_cut.num_frames, first_cut.num_features)
    for idx, cut in enumerate(cuts):
        features[idx] = torch.from_numpy(cut.load_features(mixed=False))
    return features


def collate_multi_channel_audio(cuts: CutSet) -> torch.Tensor:
    """
    Load audio samples for all the cuts and return them as a batch in a torch tensor.
    The cuts have to be of type ``MixedCut`` and their tracks will be interpreted as individual channels.
    The output shape is ``(batch, channel, time)``.
    The cuts will be padded with silence if necessary.
    """
    assert all(cut.has_recording for cut in cuts)
    assert all(isinstance(cut, MixedCut) for cut in cuts)
    cuts = maybe_pad(cuts)
    first_cut = next(iter(cuts))
    audio = torch.empty(len(cuts), len(first_cut.tracks), first_cut.num_samples)
    for idx, cut in enumerate(cuts):
        audio[idx] = torch.from_numpy(cut.load_audio())
    return audio


def collate_vectors(
        tensors: Iterable[Union[torch.Tensor, np.ndarray]],
        padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
        matching_shapes: bool = False
) -> torch.Tensor:
    """
    Convert an iterable of 1-D tensors (of possibly various lengths)
    into a single stacked tensor.

    :param tensors: an iterable of 1-D tensors.
    :param padding_value: the padding value inserted to make all tensors have the same length.
    :param matching_shapes: when ``True``, will fail when input tensors have different shapes.
    :return: a tensor with shape ``(B, L)`` where ``B`` is the number of input tensors and
        ``L`` is the number of items in the longest tensor.
    """
    tensors = [t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    longest = max(tensors, key=lambda t: t.shape[0])
    if matching_shapes:
        assert all(t.shape == longest.shape for t in tensors), \
            "All tensors must have the same shape when matching_shapes is set to True."
    result = longest.new_ones(len(tensors), longest.shape[0]) * padding_value
    for i, t in enumerate(tensors):
        result[i, :t.shape[0]] = t
    return result


def collate_matrices(
        tensors: Iterable[Union[torch.Tensor, np.ndarray]],
        padding_value: Union[int, float] = 0,
        matching_shapes: bool = False
) -> torch.Tensor:
    """
    Convert an iterable of 2-D tensors (of possibly various first dimension, but consistent second dimension)
    into a single stacked tensor.

    :param tensors: an iterable of 2-D tensors.
    :param padding_value: the padding value inserted to make all tensors have the same length.
    :param matching_shapes: when ``True``, will fail when input tensors have different shapes.
    :return: a tensor with shape ``(B, L, F)`` where ``B`` is the number of input tensors,
        ``L`` is the largest found shape[0], and ``F`` is equal to shape[1].
    """
    tensors = [t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in tensors]
    assert all(len(t.shape) == 2 for t in tensors), "Expected only 2-D input tensors."
    longest = max(tensors, key=lambda t: t.shape[0])
    if matching_shapes:
        assert all(t.shape == longest.shape for t in tensors), \
            "All tensors must have the same shape when matching_shapes is set to True."
    result = longest.new_ones(len(tensors), *longest.shape) * padding_value
    for i, t in enumerate(tensors):
        result[i, :t.shape[0]] = t
    return result


def maybe_pad(cuts: CutSet) -> CutSet:
    """Check if all cuts' durations are equal and pad them to match the longest cut otherwise."""
    if len(set(c.duration for c in cuts)) == 1:
        # All cuts are of equal duration: nothing to do
        return cuts
    # Non-equal durations: silence padding
    return cuts.pad()
