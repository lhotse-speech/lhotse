from abc import ABCMeta, abstractmethod
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import is_dataclass, asdict, dataclass, field
from functools import partial
from itertools import chain
from math import isclose
from pathlib import Path
from typing import Optional, Any, List, Iterable, Type

import lilcom
import numpy as np
import torch

from lhotse.audio import Recording
from lhotse.augmentation import WavAugmenter
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Seconds, Pathlike, load_yaml, save_to_yaml, uuid4


class FeatureExtractor(metaclass=ABCMeta):
    """
    The base class for all feature extractors in Lhotse.
    It is initialized with a config object, specific to a particular feature extraction method.
    The config is expected to be a dataclass so that it can be easily serialized.

    All derived feature extractors must implement at least the following:
    - a ``name`` class attribute (how are these features called, e.g. 'mfcc')
    - a ``config_type`` class attribute that points to the configuration dataclass type
    - the ``extract`` method,
    - the ``frame_shift`` property.

    Feature extractors that support feature-domain mixing should additionally specify two static methods:
    - ``compute_energy``, and
    - ``mix``.
    """
    name = None
    config_type = None

    def __init__(self, config: Optional[Any] = None):
        if config is None:
            config = self.config_type()
        assert is_dataclass(config), "The feature configuration object must be a dataclass."
        self.config = config

    @abstractmethod
    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray: ...

    @property
    @abstractmethod
    def frame_shift(self) -> Seconds: ...

    @staticmethod
    def mix(features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float) -> np.ndarray:
        """
        Perform feature-domain mix of two singals, ``a`` and ``b``, and return the mixed signal.

        :param features_a: Left-hand side (reference) signal.
        :param features_b: Right-hand side (mixed-in) signal.
        :param energy_scaling_factor_b: A scaling factor for ``features_b`` energy.
            It is used to achieve a specific SNR.
            E.g. to mix with an SNR of 10dB when both ``features_a`` and ``features_b`` energies are 100,
            the ``features_b`` signal energy needs to be scaled by 0.1.
            Since different features (e.g. spectrogram, fbank, MFCC) require different combination of
            transformations (e.g. exp, log, sqrt, pow) to allow mixing of two signals, the exact place
            where to apply ``energy_scaling_factor_b`` to the signal is determined by the implementer.
        :return: A mixed feature matrix.
        """
        raise ValueError('The feature extractor\'s "mix" operation is undefined.')

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        """
        Compute the total energy of a feature matrix. How the energy is computed depends on a
        particular type of features.
        It is expected that when implemented, ``compute_energy`` will never return zero.

        :param features: A feature matrix.
        :return: A positive float value of the signal energy.
        """
        raise ValueError('The feature extractor\'s "compute_energy" is undefined.')

    @classmethod
    def from_dict(cls, data: dict) -> 'FeatureExtractor':
        feature_type = data.pop('feature_type')
        extractor_type = get_extractor_type(feature_type)
        # noinspection PyUnresolvedReferences
        config = extractor_type.config_type(**data)
        return extractor_type(config)

    @classmethod
    def from_yaml(cls, path: Pathlike) -> 'FeatureExtractor':
        return cls.from_dict(load_yaml(path))

    def to_yaml(self, path: Pathlike):
        data = asdict(self.config)
        data['feature_type'] = self.name  # Insert the typename for config readability
        save_to_yaml(data, path=path)


FEATURE_EXTRACTORS = {}


def get_extractor_type(name: str) -> Type:
    """
    Return the feature extractor type corresponding to the given name.
    :param name: specifies which feature extractor should be used.
    :return: A feature extractors type.
    """
    return FEATURE_EXTRACTORS[name]


def create_default_feature_extractor(name: str) -> 'Optional[FeatureExtractor]':
    """
    Create a feature extractor object with a default configuration.
    :param name: specifies which feature extractor should be used.
    :return: A new feature extractor instance.
    """
    return get_extractor_type(name)()


def register_extractor(cls):
    """
    This decorator is used to register feature extractor classes in Lhotse so they can be easily created
    just by knowing their name.

    An example of usage:

    @register_extractor
    class MyFeatureExtractor:
        ...

    :param cls: A type (class) that is being registered.
    :return: Registered type.
    """
    FEATURE_EXTRACTORS[cls.name] = cls
    return cls


class TorchaudioFeatureExtractor(FeatureExtractor):
    """Common abstract base class for all torchaudio based feature extractors."""
    feature_fn = None

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        params = asdict(self.config)
        params.update({
            "sample_frequency": sampling_rate,
            "snip_edges": False
        })
        params['frame_shift'] *= 1000.0
        params['frame_length'] *= 1000.0
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        features = self.feature_fn(samples, **params)
        return features.numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift


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
    sampling_rate: int

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
        if start < self.start - 1e-5:
            raise ValueError(f"Cannot load features for recording {self.recording_id} starting from {start}s. "
                             f"The available range is ({self.start}, {self.end}) seconds.")
        if not isclose(start, self.start):
            frames_to_trim = round((start - self.start) / self.frame_shift)
            features = features[frames_to_trim:, :]

        # Right trim
        end = start + duration if duration is not None else None
        if duration is not None and not isclose(end, self.end):
            frames_to_trim = round((self.end - end) / self.frame_shift)
            # When duration is specified and very close to the original duration, frames_to_trim can be zero;
            # the conditional below is a safe-guard against these cases.
            if frames_to_trim:
                features = features[:-frames_to_trim, :]

        return features

    @staticmethod
    def from_dict(data: dict) -> 'Features':
        return Features(**data)


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
    features: List[Features] = field(default_factory=lambda: list())

    def __post_init__(self):
        self.features = sorted(self.features)

    @staticmethod
    def from_features(features: Iterable[Features]) -> 'FeatureSet':
        return FeatureSet(list(features))  # just for consistency with other *Sets

    @staticmethod
    def from_dicts(data: Iterable[dict]) -> 'FeatureSet':
        return FeatureSet(features=[Features.from_dict(feature_data) for feature_data in data])

    @staticmethod
    def from_yaml(path: Pathlike) -> 'FeatureSet':
        return FeatureSet.from_dicts(load_yaml(path))

    def to_yaml(self, path: Pathlike):
        data = [asdict(f) for f in self]
        save_to_yaml(data, path)

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
        return FeatureSet(features=self.features + other.features)


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
            augmenter: Optional[WavAugmenter] = None
    ):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.output_dir = Path(output_dir)
        self.augmenter = augmenter

    def process_and_store_recordings(
            self,
            recordings: Iterable[Recording],
            segmentation: Optional[SupervisionSegment] = None,
            compressed: bool = True,
            lilcom_tick_power: int = -5,
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
        feature_set = FeatureSet.from_features(feature_infos)
        feature_set.to_yaml(self.output_dir / 'feature_manifest.yml.gz')
        return feature_set

    def _process_and_store_recording(
            self,
            recording: Recording,
            segmentation: Optional[SupervisionSegment] = None,
            compressed: bool = True,
            lilcom_tick_power: int = -5,
    ) -> List[Features]:
        results = []
        for channel in recording.channel_ids:
            output_features_path = (
                    self.output_dir / 'storage' / str(uuid4())
            ).with_suffix('.llc' if compressed else '.npy')

            samples = recording.load_audio(channels=channel, root_dir=self.root_dir)

            if self.augmenter is not None:
                samples = self.augmenter.apply(samples)

            feats = self.feature_extractor.extract(samples=samples, sampling_rate=recording.sampling_rate)

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
                duration=feats.shape[0] * self.feature_extractor.frame_shift,
                type=self.feature_extractor.name,
                num_frames=feats.shape[0],
                num_features=feats.shape[1],
                sampling_rate=recording.sampling_rate,
                storage_type='lilcom' if compressed else 'numpy',
                storage_path=str(output_features_path)
            ))
        return results
