from decimal import ROUND_FLOOR, ROUND_HALF_DOWN

import torch
from torch.utils.data.dataloader import default_collate
from typing import Any, Dict, List, Sequence
from typing_extensions import Protocol, runtime_checkable

from lhotse import CutSet, SupervisionSegment
from lhotse.cut import AnyCut, MixedCut
from lhotse.utils import Seconds, compute_num_frames, compute_num_samples


class SignalField(Protocol):
    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]: ...


class Audio:
    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        return {'audio': _collate_audio(cuts)}


class MultiChannelAudio:
    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        return {'multi_channel_audio': _collate_multi_channel_audio(cuts)}


class Feats:
    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        return {'features': _collate_features(cuts)}


class MultiChannelFeats:
    def __call__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        return {'multi_channel_features': _collate_multi_channel_features(cuts)}


class SupervisionField(Protocol):
    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]: ...


@runtime_checkable
class RequiresCustomCollation(Protocol):
    def collate(self, items: Sequence, key: str) -> Any: ...


class FeatureSpan:
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
    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {'text': supervision.text}


class Characters:
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
    def __init__(self, cuts: CutSet):
        self.id2spk = sorted(cuts.speakers)
        self.spk2id = {v: k for k, v in enumerate(self.id2spk)}

    def __call__(self, supervision: SupervisionSegment, cut: AnyCut) -> Dict[str, Any]:
        return {'speaker': self.spk2id[supervision.speaker]}


class VoiceActivity:
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


def _asserted_num_frames(total_num_frames: int, start: Seconds, duration: Seconds, frame_shift: Seconds) -> int:
    """
    This closure with compute the num_frames, correct off-by-one errors in edge cases,
    and assert that the supervision does not exceed the feature matrix temporal dimension.
    """
    offset = compute_num_frames(start, frame_shift=frame_shift)
    num_frames = compute_num_frames(duration, frame_shift=frame_shift)
    diff = total_num_frames - (offset + num_frames)
    # Note: we tolerate off-by-ones because some mixed cuts could have one frame more
    # than their duration suggests (we will try to change this eventually).
    if diff == -1:
        num_frames -= 1
    assert offset + num_frames <= total_num_frames, \
        f"Unexpected num_frames ({offset + num_frames}) exceeding features time dimension for a supervision " \
        f"({total_num_frames}) when constructing a batch; please report this in Lhotse's GitHub issues, " \
        "ideally providing the Cut data that triggered this."
    return num_frames
