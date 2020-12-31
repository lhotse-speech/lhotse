import torch

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


def maybe_pad(cuts: CutSet) -> CutSet:
    """Check if all cuts' durations are equal and pad them to match the longest cut otherwise."""
    if len(set(c.duration for c in cuts)) == 1:
        # All cuts are of equal duration: nothing to do
        return cuts
    # Non-equal durations: silence padding
    return cuts.pad()
