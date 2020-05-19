"""
This file is just a rough sketch for now.
"""
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from functools import partial
from itertools import chain
from math import isclose, ceil
from pathlib import Path
from typing import Union, List, Iterable, Dict, Optional
from uuid import uuid4

import lilcom
import numpy as np
import torch
import torchaudio
import yaml

from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Seconds, Milliseconds, Pathlike


@dataclass
class SpectrogramConfig:
    dither: float = 0.0
    window_type: str = "povey"
    frame_length: Milliseconds = 25.0
    frame_shift: Milliseconds = 10.0
    remove_dc_offset: bool = True
    round_to_power_of_two: bool = True
    energy_floor: float = 0.0
    min_duration: float = 0.0
    preemphasis_coefficient: float = 0.97
    blackman_coeff: float = 0.42
    raw_energy: bool = True
    snip_edges: bool = True


@dataclass
class MfccFbankCommonConfig:
    low_freq: float = 20.0
    high_freq: float = 0.0
    num_mel_bins: int = 23
    use_energy: bool = False
    vtln_low: float = 100.0
    vtln_high: float = -500.0
    vtln_warp: float = 1.0


@dataclass
class FbankSpecificConfig:
    use_log_fbank: bool = True


@dataclass
class MfccSpecificConfig:
    cepstral_lifter: float = 22.0
    num_ceps: int = 13


@dataclass
class FeatureExtractor:
    type: str = "mfcc"  # supported: mfcc/fbank/spectrogram
    spectrogram_config: SpectrogramConfig = SpectrogramConfig()
    mfcc_fbank_common_config: MfccFbankCommonConfig = MfccFbankCommonConfig()
    fbank_config: FbankSpecificConfig = FbankSpecificConfig()
    mfcc_config: MfccSpecificConfig = MfccSpecificConfig()

    def __post_init__(self):
        assert self.type in ('spectrogram', 'mfcc', 'fbank')

    @staticmethod
    def from_yaml(path: Pathlike) -> 'FeatureExtractor':
        with open(path) as f:
            return FeatureExtractor.from_dict(yaml.safe_load(f))

    @staticmethod
    def from_dict(data: Dict) -> 'FeatureExtractor':
        return FeatureExtractor(
            type=data['type'],
            spectrogram_config=SpectrogramConfig(**data.get('spectrogram_config', {})),
            mfcc_fbank_common_config=MfccFbankCommonConfig(**data.get('mfcc_fbank_common_config', {})),
            fbank_config=FbankSpecificConfig(**data.get('fbank_config', {})),
            mfcc_config=MfccSpecificConfig(**data.get('mfcc_config', {}))
        )

    def to_yaml(self, path: Pathlike):
        with open(path, 'w') as f:
            yaml.safe_dump(asdict(self), stream=f)

    def extract(self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int):
        params = asdict(self.spectrogram_config)
        params.update({"sample_frequency": sampling_rate})
        feature_fn = None
        if self.type == 'spectrogram':
            feature_fn = torchaudio.compliance.kaldi.spectrogram
        else:
            params.update(asdict(self.mfcc_fbank_common_config))
            if self.type == 'mfcc':
                params.update(asdict(self.mfcc_config))
                feature_fn = torchaudio.compliance.kaldi.mfcc
            elif self.type == 'fbank':
                params.update(asdict(self.fbank_config))
                feature_fn = torchaudio.compliance.kaldi.fbank

        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)

        return feature_fn(waveform=samples, **params)


@dataclass(order=True)
class Features:
    """
    Represents features extracted for some particular time range in a given recording and channel.
    It contains metadata about how it's stored: storage_type describes "how to read it", for now
    it supports numpy arrays serialized with np.save, as well as arrays compressed with lilcom;
    storage_path is the path to the file on the local filesystem.
    """
    recording_id: str
    channel_id: int
    start: Seconds
    duration: Seconds
    storage_type: str  # e.g. 'lilcom', 'numpy'
    storage_path: str

    def __post_init__(self):
        assert self.storage_type in ('lilcom', 'numpy')

    def load(self, root_dir: Optional[Pathlike] = None) -> np.ndarray:
        storage_path = self.storage_path if root_dir is None else Path(root_dir) / self.storage_path
        if self.storage_type == 'lilcom':
            with open(storage_path, 'rb') as f:
                return lilcom.decompress(f.read())
        if self.storage_type == 'numpy':
            return np.load(storage_path, allow_pickle=False)
        raise ValueError(f"Unknown storage_type: {self.storage_type}")

    @property
    def end(self):
        return self.start + self.duration


@dataclass
class FeatureSet:
    """
    Represents a feature manifest, and allows to read features for given recordings
    within particular channels and time ranges.
    It also keeps information about the feature extractor parameters used to obtain this set.
    When a given recording/time-range/channel is unavailable, raises a KeyError.
    """
    # TODO(pzelasko): we might need some efficient indexing structure,
    #                 e.g. Dict[<recording-id>, Dict[<channel-id>, IntervalTree]] (pip install intervaltree)
    feature_extractor: FeatureExtractor
    features: List[Features] = field(default_factory=lambda: list())

    def __post_init__(self):
        self.features = sorted(self.features)

    @staticmethod
    def from_yaml(path: Pathlike) -> 'FeatureSet':
        with open(path) as f:
            data = yaml.safe_load(f)
        return FeatureSet(
            feature_extractor=FeatureExtractor.from_dict(data['feature_extractor']),
            features=[Features(**feature_data) for feature_data in data['features']],
        )

    def to_yaml(self, path: Pathlike):
        with open(path, 'w') as f:
            yaml.safe_dump(asdict(self), stream=f)

    def load(
            self,
            recording_id: str,
            channel_id: int = 0,
            start: Seconds = 0.0,
            duration: Optional[Seconds] = None,
            root_dir: Optional[Pathlike] = None
    ) -> np.ndarray:
        if duration is not None:
            end = start + duration
        # TODO: naive linear search; will likely require optimization
        candidates = (
            f for f in self.features
            if f.recording_id == recording_id
               and f.channel_id == channel_id
               and f.start <= start < f.end
            # filter edge case: start 1.5, features available till 1.0, duration is None
        )
        if duration is not None:
            candidates = (f for f in candidates if f.end >= end)

        candidates = list(candidates)

        if not candidates:
            raise KeyError("No features available for the requested recording/channel/region.")

        # in case there is more than one candidate feature segment, select the best fit
        # by minimizing the MSE of the time markers...
        if duration is not None:
            feature_info = min(candidates, key=lambda f: (start - f.start) ** 2 + (end - f.end) ** 2)
        else:
            feature_info = min(candidates, key=lambda f: (start - f.start) ** 2)

        features = feature_info.load(root_dir)

        # in case we have features for longer segment than required, trim them
        if not isclose(start, feature_info.start):
            frames_to_trim = time_diff_to_num_frames(
                time_diff=start - feature_info.start,
                frame_length=self.feature_extractor.spectrogram_config.frame_length / 1000.0,
                frame_shift=self.feature_extractor.spectrogram_config.frame_shift / 1000.0
            )
            features = features[frames_to_trim:, :]
        if duration is not None and not isclose(end, feature_info.end):
            frames_to_trim = time_diff_to_num_frames(
                time_diff=feature_info.end - end,
                frame_length=self.feature_extractor.spectrogram_config.frame_length / 1000.0,
                frame_shift=self.feature_extractor.spectrogram_config.frame_shift / 1000.0
            )
            features = features[:-frames_to_trim, :]

        return features

    def __iter__(self) -> Iterable[Features]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def __add__(self, other: 'FeatureSet') -> 'FeatureSet':
        assert self.feature_extractor == other.feature_extractor
        return FeatureSet(feature_extractor=self.feature_extractor, features=self.features + other.features)


class FeatureSetBuilder:
    def __init__(
            self,
            feature_extractor: FeatureExtractor,
            output_dir: Pathlike,
            augmentation_manifest=None
    ):
        self.feature_extractor = feature_extractor
        self.output_dir = Path(output_dir)
        self.augmentation_manifest = augmentation_manifest  # TODO: implement and use

    def process_and_store_recordings(
            self,
            recordings: Iterable[Recording],
            segmentation: Optional[SupervisionSegment] = None,
            compressed: bool = True,
            lilcom_tick_power: int = -8,
            num_jobs: int = 1
    ):
        (self.output_dir / 'storage').mkdir(parents=True, exist_ok=True)
        do_work = partial(
            self._process_and_store_recording,
            segmentation=segmentation,
            compressed=compressed,
            lilcom_tick_power=lilcom_tick_power
        )
        with ProcessPoolExecutor(num_jobs) as ex:
            feature_set = FeatureSet(
                feature_extractor=self.feature_extractor,
                features=list(chain.from_iterable(ex.map(do_work, recordings)))
            )
        feature_set.to_yaml(self.output_dir / 'feature_manifest.yml')

    def _process_and_store_recording(
            self,
            recording: Recording,
            segmentation: Optional[SupervisionSegment] = None,
            compressed: bool = True,
            lilcom_tick_power: int = -8,
    ) -> List[Features]:
        results = []
        for channel in recording.channel_ids:
            output_features_path = (
                    self.output_dir / 'storage' / str(uuid4())
            ).with_suffix('.llc' if compressed else '.npy')

            samples = torch.from_numpy(recording.load_audio(channels=channel))

            # TODO: use augmentation manifest here
            feats = self.feature_extractor.extract(
                samples=samples,
                sampling_rate=recording.sampling_rate
            ).numpy()

            if compressed:
                # TODO: use segmentation manifest here
                serialized_feats = lilcom.compress(feats, tick_power=lilcom_tick_power)
                with open(output_features_path, 'wb') as f:
                    f.write(serialized_feats)
            else:
                np.save(output_features_path, feats, allow_pickle=False)

            results.append(Features(
                recording_id=recording.id,
                channel_id=channel,
                # TODO: revise start and duration with segmentation manifest info
                start=0.0,
                duration=recording.duration_seconds,
                storage_type='lilcom' if compressed else 'numpy',
                storage_path=str(output_features_path)
            ))
        return results


def time_diff_to_num_frames(time_diff: Seconds, frame_length: Seconds, frame_shift: Seconds) -> int:
    """Convert duration to an equivalent number of frames, so as to not exceed the duration."""
    return int(ceil((time_diff - frame_length) / frame_shift))
