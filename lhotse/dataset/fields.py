"""
The *fields* defined in this file conform to "Protocols" defined at the top.
"Protocols" are like concepts in C++20, and you can think of them as an attempt
to specify an expected interface without forcing the user to inherit from a common base class.
It should make creating custom fields simpler for the user
(they can in fact be implemented even as a simple method).
"""
from decimal import ROUND_FLOOR

import torch
from torch.utils.data.dataloader import default_collate
from typing import Any, Dict, List, Sequence
from typing_extensions import Protocol, runtime_checkable

from lhotse import CutSet, SupervisionSegment
from lhotse.cut import AnyCut, MixedCut
from lhotse.utils import compute_num_frames, compute_num_samples


# Protocol (interface) definitions


class SignalField(Protocol):
    """
    Represents a piece of information contained in ``Recording`` or ``Features`` manifests.
    It could yield e.g. audio samples or feature matrices.
    It consumes a ``CutSet`` and collates all the Cut data internally.
    """

    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]: ...


class SupervisionField(Protocol):
    """
    Represents a piece of information contained in ``SupervisionSegment`` manifests.
    It could yield e.g. transcription text or speaker ID.
    It consumes individual supervision + cut pairs, and the data is collated later.
    """

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]: ...


@runtime_checkable
class RequiresCustomCollation(Protocol):
    """
    A field that defines ``collate()`` will be treated differently during collation -
    instead of using the default collation method in PyTorch, we will use the user-provided one.
    """

    def collate(self, items: Sequence, key: str) -> Any: ...


# Signal field implementations


class Audio:
    """
    Returns a single-channel audio field (if the cut is a MixedCut, it will perform the mix).

    Output:

    .. code-block:

        output = {'audio': <tensor, shape (B, T)>}
    """

    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        return {'audio': _collate_audio(cuts)}


class MultiChannelAudio:
    """
    Returns a multi-channel audio field (the cut has to be a MixedCut, but we won't mix its tracks).

    Output:

    .. code-block:

        output = {'multi_channel_audio': <tensor, shape (B, C, T)>}
    """

    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        assert all(isinstance(cut, MixedCut) for cut in cuts)
        return {'multi_channel_audio': _collate_multi_channel_audio(cuts)}


class Feats:
    """
    Returns a single-channel feature matrix field (if the cut is a MixedCut, it will perform the mix).

    Output:

    .. code-block:

        output = {'features': <tensor, shape (B, T, F)>}
    """

    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        return {'features': _collate_features(cuts)}


class MultiChannelFeats:
    """
    Returns a multi-channel feature matrix field (the cut has to be a MixedCut, but we won't mix its tracks).

    Output:

    .. code-block:

        output = {'multi_channel_features': <tensor, shape (B, C, T, F)>}
    """

    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        assert all(isinstance(cut, MixedCut) for cut in cuts)
        return {'multi_channel_features': _collate_multi_channel_features(cuts)}


# Supervision field implementations


class FeatureSpan:
    """
    Returns a field describing start frames and number of frames for the supervision.

    Output:

    .. code-block:

        output = {'start_frame': int, 'num_frames': int}
    """

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {
            'start_frame': compute_num_frames(
                supervision.start,
                frame_shift=cut.frame_shift,
                # Note: Rounding "floor" can sometimes result in one extra frame being included
                # in the left context; but it guarantees that we will never go out-of-bounds when
                # summing start_frame + num_frames.
                rounding=ROUND_FLOOR
            ),
            'num_frames': compute_num_frames(
                supervision.duration,
                frame_shift=cut.frame_shift
            )
        }


class AudioSpan:
    """
    Returns a field describing start samples and number of samples for the supervision.

    Output:

    .. code-block:

        output = {'start_sample': int, 'num_samples': int}
    """

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {
            'start_sample': compute_num_samples(
                supervision.start,
                sampling_rate=cut.sampling_rate,
                rounding=ROUND_FLOOR
            ),
            'num_samples': compute_num_samples(
                supervision.duration,
                sampling_rate=cut.sampling_rate
            )
        }


class Text:
    """
    Returns a field with the transcription text (as string).

    Output:

    .. code-block:

        output = {'text': str}
    """

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {'text': supervision.text}


class Characters:
    """
    Returns a field with the transcription text as a sequence of (string) characters.
    During collation, it will add `<PAD>` symbols at the end to match the longest
    sequence in the batch.

    Output:

    .. code-block:

        output = {'chars': List[str], 'chars_len': int}
    """

    def __init__(self):
        self.pad = '<PAD>'

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {'chars': list(supervision.text), 'chars_len': len(supervision.text)}

    def collate(self, items: List[str], key: str):
        assert key in ('chars', 'chars_len')
        if key == 'chars':
            items = sorted(items, key=len, reverse=True)
            for idx in range(1, len(items)):
                items[idx].extend([self.pad] * (len(items[0]) - len(items[idx])))
        elif key == 'chars_len':
            items = default_collate(items)
        return items


class CharacterIds:
    """
    Returns a field with the transcription text as a sequence of (int) character ids.
    During collation, it will add `<PAD>` symbol ids at the end to match the longest
    sequence in the batch.

    Output:

    .. code-block:

        output = {'char_ids': List[int], 'char_ids_len': int}
    """

    def __init__(self, cuts):
        self.pad = '<PAD>'
        self.id2sym = sorted({char for cut in cuts for s in cut.supervisions for char in s.text}) + [self.pad]
        self.sym2id = {v: k for k, v in enumerate(self.id2sym)}

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {'char_ids': [self.sym2id[c] for c in supervision.text], 'char_ids_len': len(supervision.text)}

    def collate(self, items: List[int], key: str):
        assert key in ('char_ids', 'char_ids_len')
        if key == 'char_ids':
            items = sorted(items, key=len, reverse=True)
            for idx in range(1, len(items)):
                items[idx].extend([self.sym2id[self.pad]] * (len(items[0]) - len(items[idx])))
            items = torch.tensor(items, dtype=torch.int32)
        elif key == 'chars_ids_len':
            items = default_collate(items)
        return items


class Speaker:
    """
    Returns a field with the speaker ID index.
    The indices are obtained by inspecting all speaker in the CutSet provided to ``__init__``.

    Output:

    .. code-block:

        output = {'speaker': int}
    """

    def __init__(self, cuts: CutSet):
        self.id2spk = sorted(cuts.speakers)
        self.spk2id = {v: k for k, v in enumerate(self.id2spk)}

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {'speaker': self.spk2id[supervision.speaker]}


class VoiceActivity:
    """
    Returns a field with a voice activity vector.
    It is a binary vector that has values corresponding to each frame/sample.

    Output:

    .. code-block:

        output = {'audio_is_voiced': <array, shape (T)>}
        output = {'frame_is_voiced': <array, shape (T)>}
    """

    def __init__(self, use_features: bool = True, use_audio: bool = False):
        if not use_audio and not use_features:
            raise ValueError("VoiceActivity requested but both use_audio and use_features set to False.")
        self.use_features = use_features
        self.use_audio = use_audio

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        output = {}
        if self.use_audio:
            output['audio_is_voiced'] = cut.supervisions_audio_mask()
        if self.use_features:
            output['frame_is_voiced'] = cut.supervisions_feature_mask()
        return output


def _collate_features(cuts: CutSet) -> torch.Tensor:
    assert all(cut.has_features for cut in cuts)
    first_cut = next(iter(cuts))
    features = torch.empty(len(cuts), first_cut.num_frames, first_cut.num_features)
    for idx, cut in enumerate(cuts):
        features[idx] = torch.from_numpy(cut.load_features())
    return features


def _collate_audio(cuts: CutSet) -> torch.Tensor:
    assert all(cut.has_recording for cut in cuts)
    first_cut = next(iter(cuts))
    audio = torch.empty(len(cuts), first_cut.num_samples)
    for idx, cut in enumerate(cuts):
        audio[idx] = torch.from_numpy(cut.load_audio()[0])
    return audio


def _collate_multi_channel_features(cuts: CutSet) -> torch.Tensor:
    assert all(cut.has_features for cut in cuts)
    assert all(isinstance(cut, MixedCut) for cut in cuts)
    # Output tensor shape: (B, C, T, F) -> (batch_size, num_channels, num_frames, num_features)
    first_cut = next(iter(cuts))
    # TODO: make MixedCut more friendly to use with multi channel audio;
    #  discount PaddingCuts in "tracks" when specifying the number of channels
    features = torch.empty(len(cuts), len(first_cut.tracks), first_cut.num_frames, first_cut.num_features)
    for idx, cut in enumerate(cuts):
        features[idx] = torch.from_numpy(cut.load_features(mixed=False))
    return features


def _collate_multi_channel_audio(cuts: CutSet) -> torch.Tensor:
    assert all(cut.has_recording for cut in cuts)
    assert all(isinstance(cut, MixedCut) for cut in cuts)
    first_cut = next(iter(cuts))
    audio = torch.empty(len(cuts), len(first_cut.tracks), first_cut.num_samples)
    for idx, cut in enumerate(cuts):
        audio[idx] = torch.from_numpy(cut.load_audio())
    return audio
