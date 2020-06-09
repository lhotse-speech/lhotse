"""
This file is just a rough sketch for now.
"""
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from functools import partial
from itertools import chain
from math import isclose
from pathlib import Path
from typing import Union, List, Iterable, Dict, Optional, Tuple
from uuid import uuid4

import lilcom
import numpy as np
import torch
import torchaudio
import yaml

from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Seconds, Milliseconds, Pathlike, time_diff_to_num_frames, Decibels


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
    raw_energy: bool = True
    snip_edges: bool = False


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
    type: str = "fbank"  # supported: mfcc/fbank/spectrogram
    spectrogram_config: SpectrogramConfig = SpectrogramConfig()
    mfcc_fbank_common_config: MfccFbankCommonConfig = MfccFbankCommonConfig()
    fbank_config: FbankSpecificConfig = FbankSpecificConfig()
    mfcc_config: MfccSpecificConfig = MfccSpecificConfig()

    def __post_init__(self):
        if self.type not in ('spectrogram', 'mfcc', 'fbank'):
            raise ValueError(f"Unsupported feature type: '{self.type}'")

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

    # The Features object must know its frame length and shift to trim the features matrix when loading.
    frame_length: Milliseconds
    frame_shift: Milliseconds

    # Parameters related to storage - they define how to load the feature matrix.
    storage_type: str  # e.g. 'lilcom', 'numpy'
    storage_path: str

    def __post_init__(self):
        assert self.storage_type in ('lilcom', 'numpy')

    def load(
            self,
            root_dir: Optional[Pathlike] = None,
            start: Seconds = 0.0,
            duration: Optional[Seconds] = None,
    ) -> np.ndarray:
        # Load the features from the storage
        storage_path = self.storage_path if root_dir is None else Path(root_dir) / self.storage_path
        if self.storage_type == 'lilcom':
            with open(storage_path, 'rb') as f:
                features = lilcom.decompress(f.read())
        elif self.storage_type == 'numpy':
            features = np.load(storage_path, allow_pickle=False)
        else:
            raise ValueError(f"Unknown storage_type: {self.storage_type}")

        # In case the caller requested only a subset of features, trim them

        # Left trim
        if not isclose(start, self.start):
            frames_to_trim = time_diff_to_num_frames(
                time_diff=start - self.start,
                frame_length=self.frame_length / 1000.0,
                frame_shift=self.frame_shift / 1000.0
            )
            features = features[frames_to_trim:, :]

        # Right trim
        end = start + duration if duration is not None else None
        if duration is not None and not isclose(end, self.end):
            frames_to_trim = time_diff_to_num_frames(
                time_diff=self.end - end,
                frame_length=self.frame_length / 1000.0,
                frame_shift=self.frame_shift / 1000.0
            )
            features = features[:-frames_to_trim, :]

        return features

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

    def find(
            self,
            recording_id: str,
            channel_id: int = 0,
            start: Seconds = 0.0,
            duration: Optional[Seconds] = None,
    ) -> Features:
        """
        Find and return a Features object that best satisfies the search criteria.
        Raise a KeyError when no such object is available.
        """
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

        return feature_info

    def load(
            self,
            recording_id: str,
            channel_id: int = 0,
            start: Seconds = 0.0,
            duration: Optional[Seconds] = None,
            root_dir: Optional[Pathlike] = None
    ) -> np.ndarray:
        """
        Find a Features object that best satisfies the search criteria and load the features as a numpy ndarray.
        Raise a KeyError when no such object is available.
        """
        feature_info = self.find(
            recording_id=recording_id,
            channel_id=channel_id,
            start=start,
            duration=duration
        )
        features = feature_info.load(root_dir=root_dir, start=start, duration=duration)
        return features

    def __iter__(self) -> Iterable[Features]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def __add__(self, other: 'FeatureSet') -> 'FeatureSet':
        assert self.feature_extractor == other.feature_extractor
        # TODO: when validating, assert that there is no overlap between regions which have extracted features
        return FeatureSet(feature_extractor=self.feature_extractor, features=self.features + other.features)


class FeatureSetBuilder:
    def __init__(
            self,
            feature_extractor: FeatureExtractor,
            output_dir: Pathlike,
            root_dir: Optional[Pathlike] = None,
            augmentation_manifest=None
    ):
        self.root_dir = root_dir
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

            samples = torch.from_numpy(recording.load_audio(channels=channel, root_dir=self.root_dir))

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
                frame_length=self.feature_extractor.spectrogram_config.frame_length,
                frame_shift=self.feature_extractor.spectrogram_config.frame_shift,
                duration=recording.duration_seconds,
                storage_type='lilcom' if compressed else 'numpy',
                storage_path=str(output_features_path)
            ))
        return results


def overlay_fbank(
        left_feats: np.ndarray,
        right_feats: np.ndarray,
        snr: Optional[Decibels] = None,
        offset_right_by: Seconds = 0,
        frame_length: Optional[Milliseconds] = None,
        frame_shift: Optional[Milliseconds] = None,
        log_energy_floor: float = -1000.0
) -> np.ndarray:
    """
    Mix two log-mel energy feature matrices together to form a single signal.

    :param left_feats: The first feature matrix (assumed to be signal).
    :param right_feats: The second feature matrix (assumed to be noise).
    :param snr: float, decibels, optional: will rescale right_feats to meet the SNR criterion
    :param offset_right_by: float, seconds, optional: will shift right_feats forward in time
    :param frame_length: float, milliseconds, optional: required for offset_right_by
    :param frame_shift: float, milliseconds, optional: required for offset_right_by
    :param log_energy_floor: value used for padding frames when offsetting
    :return: np.ndarray; a mixed feature matrix of log-mel energies
    """
    # First deal with the offset
    if offset_right_by > 0:
        assert frame_length is not None and frame_shift is not None
        num_frames_offset = time_diff_to_num_frames(
            offset_right_by,
            frame_length=frame_length / 1000,
            frame_shift=frame_shift / 1000
        )
        num_feats = left_feats.shape[1]
        padded_left_feats = np.vstack([left_feats, log_energy_floor * np.ones((num_frames_offset, num_feats))])
        padded_right_feats = np.vstack([log_energy_floor * np.ones((num_frames_offset, num_feats)), right_feats])
    else:
        padded_left_feats, padded_right_feats = left_feats, right_feats

    # Then overlay - for SNR re-scale the right signal, otherwise logsumexp
    if snr is not None:
        # TODO: The SNR stuff here might be tricky. For prototyping assume the following:
        #  - always dealing with log mel filterbank feats
        #  - total signal energy can be approximated by summing individual bands energies
        #    (assumes no spectral leakage...) across time and frequencies
        #  - achieve the final SNR by determining the signal energy ratio and then scaling
        #    the signals to satisfy requested SNR
        left_energy, right_energy = [np.sum(np.exp(f)) for f in [left_feats, right_feats]]
        requested_right_energy = left_energy * (10.0 ** (-snr / 10))
        overlayed_feats = np.log(
            np.exp(padded_left_feats) +
            (requested_right_energy / right_energy) * np.exp(padded_right_feats)
        )
    else:
        overlayed_feats = np.log(np.exp(padded_left_feats) + np.exp(padded_right_feats))

    return overlayed_feats


def pad_shorter(
        left_feats: np.ndarray,
        right_feats: np.ndarray,
        log_energy_floor: float = -1000.0
) -> Tuple[np.ndarray, np.ndarray]:
    num_feats = left_feats.shape[1]
    if num_feats != right_feats.shape[1]:
        raise ValueError('Cannot pad feature matrices with different number of features.')

    size_diff = abs(left_feats.shape[0] - right_feats.shape[0])
    if not size_diff:
        return left_feats, right_feats

    if left_feats.shape[0] > right_feats.shape[0]:
        return left_feats, np.vstack([right_feats, log_energy_floor * np.ones((size_diff, num_feats))])
    else:
        return np.vstack([left_feats, log_energy_floor * np.ones((size_diff, num_feats))]), right_feats
