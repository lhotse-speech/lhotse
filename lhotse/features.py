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
from lhotse.utils import Seconds, Pathlike, Decibels, load_yaml, save_to_yaml


@dataclass
class SpectrogramConfig:
    # Note that `snip_edges` parameter is missing from config: in order to simplify the relationship between
    #  the duration and the number of frames, we are always setting `snip_edges` to False.
    dither: float = 0.0
    window_type: str = "povey"
    # Note that frame_length and frame_shift will be converted to milliseconds before torchaudio/Kaldi sees them
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    remove_dc_offset: bool = True
    round_to_power_of_two: bool = True
    energy_floor: float = 0.0
    min_duration: float = 0.0
    preemphasis_coefficient: float = 0.97
    raw_energy: bool = True


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
    spectrogram_config: SpectrogramConfig = field(default_factory=lambda: SpectrogramConfig())
    mfcc_fbank_common_config: MfccFbankCommonConfig = field(default_factory=lambda: MfccFbankCommonConfig())
    fbank_config: FbankSpecificConfig = field(default_factory=lambda: FbankSpecificConfig())
    mfcc_config: MfccSpecificConfig = field(default_factory=lambda: MfccSpecificConfig())

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

    def to_yaml(self, path: Pathlike, include_defaults: bool = False):
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(include_defaults=include_defaults), stream=f)

    def to_dict(self, include_defaults: bool = False) -> dict:
        config = asdict(self)
        if include_defaults:
            return config
        default_config = FeatureExtractor().to_dict(include_defaults=True)
        for key, subconfig in default_config.items():
            if key == 'type':
                continue  # we always want to preserve features type field
            for subkey, value in subconfig.items():
                if config[key][subkey] == value:
                    del config[key][subkey]
            if not config[key]:
                del config[key]
        return config

    def extract(self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int):
        params = asdict(self.spectrogram_config)
        params.update({"sample_frequency": sampling_rate})
        params['frame_shift'] *= 1000.0
        params['frame_length'] *= 1000.0
        params['snip_edges'] = False
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

    # Useful information about the features - their type (fbank, mfcc) and shape
    type: str
    num_frames: int
    num_features: int

    # Parameters related to storage - they define how to load the feature matrix.
    storage_type: str  # e.g. 'lilcom', 'numpy'
    storage_path: str

    @property
    def end(self) -> Seconds:
        return self.start + self.duration

    @property
    def frame_shift(self) -> Seconds:
        return round(self.duration / self.num_frames, ndigits=3)

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
            frames_to_trim = round((start - self.start) / self.frame_shift)
            features = features[frames_to_trim:, :]

        # Right trim
        end = start + duration if duration is not None else None
        if duration is not None and not isclose(end, self.end):
            frames_to_trim = round((self.end - end) / self.frame_shift)
            features = features[:-frames_to_trim, :]

        return features


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
        data = load_yaml(path)
        return FeatureSet(
            feature_extractor=FeatureExtractor.from_dict(data['feature_extractor']),
            features=[Features(**feature_data) for feature_data in data['features']],
        )

    def to_yaml(self, path: Pathlike):
        save_to_yaml(asdict(self), path)

    def find(
            self,
            recording_id: str,
            channel_id: int = 0,
            start: Seconds = 0.0,
            duration: Optional[Seconds] = None,
            leeway: Seconds = 0.05
    ) -> Features:
        """
        Find and return a Features object that best satisfies the search criteria.
        Raise a KeyError when no such object is available.

        :param recording_id: str, requested recording ID.
        :param channel_id: int, requested channel.
        :param start: float, requested start time in seconds for the feature chunk.
        :param duration: optional float, requested duration in seconds for the feature chunk.
            By default, return everything from the start.
        :param leeway: float, controls how strictly we have to match the requested start and duration criteria.
            It is necessary to keep a small positive value here (default 0.05s), as there might be differneces between
            the duration of recording/supervision segment, and the duration of features. The latter one is constrained
            to be a multiple of frame_shift, while the former can be arbitrary.
        :return: a Features object satisfying the search criteria.
        """
        if duration is not None:
            end = start + duration
        # TODO: naive linear search; will likely require optimization
        candidates = (
            f for f in self.features
            if f.recording_id == recording_id
               and f.channel_id == channel_id
               and f.start - leeway <= start < f.end + leeway
            # filter edge case: start 1.5, features available till 1.0, duration is None
        )
        if duration is not None:
            candidates = (f for f in candidates if f.end >= end - leeway)

        candidates = list(candidates)

        if not candidates:
            raise KeyError(
                f"No features available for recording '{recording_id}', channel {channel_id} in time range [{start}s,"
                f" {'end' if duration is None else duration}s]")

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
    """
    An extended constructor for the FeatureSet. Think of it as a class wrapper for a feature extraction script.
    It consumes an iterable of Recordings, extracts the features specified by the FeatureExtractor config,
    and saves stores them on the disk.

    Eventually, we plan to extend it with the capability to extract only the features in
    specified regions of recordings and to perform some time-domain data augmentation.
    """

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
    ) -> FeatureSet:
        (self.output_dir / 'storage').mkdir(parents=True, exist_ok=True)
        do_work = partial(
            self._process_and_store_recording,
            segmentation=segmentation,
            compressed=compressed,
            lilcom_tick_power=lilcom_tick_power
        )
        if num_jobs == 1:
            # Avoid spawning subprocesses for single threaded processing
            feature_infos = list(chain.from_iterable(map(do_work, recordings)))
        else:
            with ProcessPoolExecutor(num_jobs) as ex:
                feature_infos = list(chain.from_iterable(ex.map(do_work, recordings)))
        feature_set = FeatureSet(
            feature_extractor=self.feature_extractor,
            features=feature_infos
        )
        feature_set.to_yaml(self.output_dir / 'feature_manifest.yml.gz')
        return feature_set

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
                # We simplify the relationship between num_frames and duration - we guarantee that
                #  the duration is always num_frames * frame_shift
                duration=feats.shape[0] * self.feature_extractor.spectrogram_config.frame_shift,
                type=self.feature_extractor.type,
                num_frames=feats.shape[0],
                num_features=feats.shape[1],
                storage_type='lilcom' if compressed else 'numpy',
                storage_path=str(output_features_path)
            ))
        return results


class FbankMixer:
    """
    Utility class to mix multiple log-mel energy feature matrices into a single one.
    It pads the signals with low energy values to account for differing lengths and offsets.
    """

    def __init__(
            self,
            base_feats: np.ndarray,
            frame_shift: Seconds,
            log_energy_floor: float = -1000.0
    ):
        """
        :param base_feats: The features used to initialize the FbankMixer are a point of reference
            in terms of energy and offset for all features mixed into them.
        :param frame_shift: Required to correctly compute offset and padding during the mix.
        :param frame_length: Required to correctly compute offset and padding during the mix.
        :param log_energy_floor: The value used to pad the shorter features during the mix.
        """
        # The mixing output will be available in self.mixed_feats
        self.mixed_feats = base_feats
        # Keep a pre-computed energy value of the features that we initialize the Mixer with;
        # it is required to compute gain ratios that satisfy SNR during the mix.
        self.reference_energy = fbank_energy(base_feats)
        self.frame_shift = frame_shift
        self.log_energy_floor = log_energy_floor

    @property
    def num_features(self):
        return self.mixed_feats.shape[1]

    def add_to_mix(
            self,
            feats: np.ndarray,
            snr: Optional[Decibels] = None,
            offset: Seconds = 0.0
    ):
        """
        Add feature matrix of a new track into the mix.
        :param feats: A 2-d feature matrix to be mixed in.
        :param snr: Signal-to-noise ratio, assuming `feats` represents noise (positive SNR - lower `feats` energy,
            negative SNR - higher `feats` energy)
        :param offset: How many seconds to shift `feats` in time. For mixing, the signal will be padded before
            the start with low energy values.
        :return:
        """
        assert offset >= 0.0, "Negative offset in mixing is not supported."

        num_frames_offset = round(offset / self.frame_shift)
        current_num_frames = self.mixed_feats.shape[0]
        incoming_num_frames = feats.shape[0] + num_frames_offset
        mix_num_frames = max(current_num_frames, incoming_num_frames)

        existing_feats = self.mixed_feats
        feats_to_add = feats

        # When the existing frames are less than what we anticipate after the mix,
        # we need to pad after the end of the existing features mixed so far.
        if current_num_frames < mix_num_frames:
            existing_feats = np.vstack([
                self.mixed_feats,
                self.log_energy_floor * np.ones((mix_num_frames - current_num_frames, self.num_features))
            ])

        # When there is an offset, we need to pad before the start of the features we're adding.
        if offset > 0:
            feats_to_add = np.vstack([
                self.log_energy_floor * np.ones((num_frames_offset, self.num_features)),
                feats_to_add
            ])

        # When the features we're mixing in are shorter that the anticipated mix length,
        # we need to pad after their end.
        # Note: we're doing that non-efficiently, as it we potentially re-allocate numpy arrays twice,
        # during this padding and the  offset padding before. If that's a bottleneck, we'll optimize.
        if incoming_num_frames < mix_num_frames:
            feats_to_add = np.vstack([
                feats_to_add,
                self.log_energy_floor * np.ones((mix_num_frames - incoming_num_frames, self.num_features))
            ])

        # When SNR is requested, find what gain is needed to satisfy the SNR
        gain = 1.0
        if snr is not None:
            # TODO: The SNR stuff here might be tricky. For prototyping assume the following:
            #  - always dealing with log mel filterbank feats
            #  - total signal energy can be approximated by summing individual bands energies
            #    (assumes no spectral leakage...) across time and frequencies
            #  - achieve the final SNR by determining the signal energy ratio and then scaling
            #    the signals to satisfy requested SNR

            # Compute the added signal energy before it was padded
            added_feats_energy = fbank_energy(feats)
            target_energy = self.reference_energy * (10.0 ** (-snr / 10))
            gain = target_energy / added_feats_energy

        self.mixed_feats = np.log(np.exp(existing_feats) + gain * np.exp(feats_to_add))


def fbank_energy(fbank: np.ndarray) -> float:
    return float(np.sum(np.exp(fbank)))


def pad_shorter(
        left_feats: np.ndarray,
        right_feats: np.ndarray,
        log_energy_floor: float = -1000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the "shorter" feature matrix and pads it with low energy frames. Useful for overlaying the
    features when the real acoustic context is not available.
    """
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
