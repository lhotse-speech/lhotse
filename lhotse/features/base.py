from abc import ABCMeta, abstractmethod
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import is_dataclass, asdict, dataclass, field
from itertools import chain
from math import isclose
from pathlib import Path
from typing import Optional, Any, List, Iterable, Type, Union, Dict

import numpy as np
import torch

from lhotse.audio import Recording
from lhotse.augmentation import WavAugmenter
from lhotse.features.io import FeaturesWriter, get_reader
from lhotse.utils import Seconds, Pathlike, load_yaml, save_to_yaml, uuid4, JsonMixin, YamlMixin, fastcopy


class FeatureExtractor(metaclass=ABCMeta):
    """
    The base class for all feature extractors in Lhotse.
    It is initialized with a config object, specific to a particular feature extraction method.
    The config is expected to be a dataclass so that it can be easily serialized.

    All derived feature extractors must implement at least the following:

    * a ``name`` class attribute (how are these features called, e.g. 'mfcc')
    * a ``config_type`` class attribute that points to the configuration dataclass type
    * the ``extract`` method,
    * the ``frame_shift`` property.

    Feature extractors that support feature-domain mixing should additionally specify two static methods:

    * ``compute_energy``, and
    * ``mix``.

    By itself, the ``FeatureExtractor`` offers the following high-level methods
    that are not intended for overriding:

    * ``extract_from_samples_and_store``
    * ``extract_from_recording_and_store``

    These methods run a larger feature extraction pipeline that involves data augmentation and disk storage.
    """
    name = None
    config_type = None

    def __init__(self, config: Optional[Any] = None):
        if config is None:
            config = self.config_type()
        assert is_dataclass(config), "The feature configuration object must be a dataclass."
        self.config = config

    @abstractmethod
    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Defines how to extract features using a numpy ndarray of audio samples and the sampling rate.

        :return: a numpy ndarray representing the feature matrix.
        """
        pass

    @property
    @abstractmethod
    def frame_shift(self) -> Seconds: ...

    @abstractmethod
    def feature_dim(self, sampling_rate: int) -> int: ...

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

    def extract_from_samples_and_store(
            self,
            samples: np.ndarray,
            storage: FeaturesWriter,
            sampling_rate: int,
            offset: Seconds = 0,
            augmenter: Optional[WavAugmenter] = None,
    ):
        """
        Extract the features from an array of audio samples in a full pipeline:

        * optional audio augmentation;
        * extract the features;
        * save them to disk in a specified directory;
        * return a ``Features`` object with a description of the extracted features.

        Note, unlike in ``extract_from_recording_and_store``, the returned ``Features`` object
        might not be suitable to store in a ``FeatureSet``, as it does not reference any particular
        ``Recording``. Instead, this method is useful when extracting features from cuts - especially
        ``MixedCut`` instances, which may be created from multiple recordings and channels.

        :param samples: a numpy ndarray with the audio samples.
        :param sampling_rate: integer sampling rate of ``samples``.
        :param storage: a ``FeaturesWriter`` object that will handle storing the feature matrices.
        :param offset: an offset in seconds for where to start reading the recording - when used for
            ``Cut`` feature extraction, must be equal to ``Cut.start``.
        :param augmenter: an optional ``WavAugmenter`` instance to modify the waveform before feature extraction.
        :return: a ``Features`` manifest item for the extracted feature matrix (it is not written to disk).
        """
        if augmenter is not None:
            samples = augmenter.apply(samples)
        feats = self.extract(samples=samples, sampling_rate=sampling_rate)
        storage_key = store_feature_array(feats, storage=storage)
        return Features(
            start=offset,
            # We simplify the relationship between num_frames and duration - we guarantee that
            #  the duration is always num_frames * frame_shift
            duration=feats.shape[0] * self.frame_shift,
            type=self.name,
            num_frames=feats.shape[0],
            num_features=feats.shape[1],
            sampling_rate=sampling_rate,
            storage_type=storage.name,
            storage_path=str(storage.storage_path),
            storage_key=storage_key
        )

    def extract_from_recording_and_store(
            self,
            recording: Recording,
            storage: FeaturesWriter,
            offset: Seconds = 0,
            duration: Optional[Seconds] = None,
            channels: Union[int, List[int]] = None,
            augmenter: Optional[WavAugmenter] = None,
    ):
        """
        Extract the features from a ``Recording`` in a full pipeline:

        * load audio from disk;
        * optionally, perform audio augmentation;
        * extract the features;
        * save them to disk in a specified directory;
        * return a ``Features`` object with a description of the extracted features and the source data used.

        :param recording: a ``Recording`` that specifies what's the input audio.
        :param storage: a ``FeaturesWriter`` object that will handle storing the feature matrices.
        :param offset: an optional offset in seconds for where to start reading the recording.
        :param duration: an optional duration specifying how much audio to load from the recording.
        :param channels: an optional int or list of ints, specifying the channels;
            by default, all channels will be used.
        :param augmenter: an optional ``WavAugmenter`` instance to modify the waveform before feature extraction.
        :return: a ``Features`` manifest item for the extracted feature matrix.
        """
        samples = recording.load_audio(
            offset_seconds=offset,
            duration_seconds=duration,
            channels=channels,
        )
        if augmenter is not None:
            samples = augmenter.apply(samples)
        feats = self.extract(samples=samples, sampling_rate=recording.sampling_rate)
        storage_key = store_feature_array(feats, storage=storage)
        return Features(
            recording_id=recording.id,
            channels=channels if channels is not None else recording.channel_ids,
            # The start is relative to the beginning of the recording.
            start=offset,
            # We simplify the relationship between num_frames and duration - we guarantee that
            #  the duration is always num_frames * frame_shift
            duration=feats.shape[0] * self.frame_shift,
            type=self.name,
            num_frames=feats.shape[0],
            num_features=feats.shape[1],
            sampling_rate=recording.sampling_rate,
            storage_type=storage.name,
            storage_path=str(storage.storage_path),
            storage_key=storage_key
        )

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
    class MyFeatureExtractor: ...

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
        features = self.feature_fn(samples, **params).to(torch.float32)
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
    # Useful information about the features - their type (fbank, mfcc) and shape
    type: str
    num_frames: int
    num_features: int
    sampling_rate: int

    # Information about the time range of the features.
    start: Seconds
    duration: Seconds

    # Parameters related to storage - they define how to load the feature matrix.

    # Storage type defines which features reader type should be instantiated
    # e.g. 'lilcom_files', 'numpy_files', 'lilcom_hdf5'
    storage_type: str

    # Storage path is either the path to some kind of archive (like HDF5 file) or a path
    # to a directory holding files with feature matrices (exact semantics depend on storage_type).
    storage_path: str

    # Storage key is either the key used to retrieve a feautre matrix from an archive like HDF5,
    # or the name of the file in a directory (exact semantics depend on the storage_type).
    storage_key: str

    # Information which recording and channels were used to extract the features.
    # When ``recording_id`` and ``channels`` are ``None``, it means that the
    # features were extracted from a cut (e.g. a ``MixedCut``), which might have consisted
    # of multiple recordings.
    recording_id: Optional[str] = None
    channels: Optional[Union[int, List[int]]] = None

    @property
    def end(self) -> Seconds:
        return self.start + self.duration

    @property
    def frame_shift(self) -> Seconds:
        return round(self.duration / self.num_frames, ndigits=3)

    def load(
            self,
            start: Optional[Seconds] = None,
            duration: Optional[Seconds] = None,
    ) -> np.ndarray:
        # noinspection PyArgumentList
        storage = get_reader(self.storage_type)(self.storage_path)
        left_offset_frames, right_offset_frames = 0, None

        if start is None:
            start = self.start
        # In case the caller requested only a sub-span of the features, trim them.
        # Left trim
        if start < self.start - 1e-5:
            raise ValueError(f"Cannot load features for recording {self.recording_id} starting from {start}s. "
                             f"The available range is ({self.start}, {self.end}) seconds.")
        if not isclose(start, self.start):
            left_offset_frames = round((start - self.start) / self.frame_shift)

        # Right trim
        end = start + duration if duration is not None else None
        if duration is not None and not isclose(end, self.end):
            # Note the "minus" sign below before round - we're slicing a numpy array, e.g. a[20:-100]
            right_offset_frames = -round((self.end - end) / self.frame_shift)
            # When duration is specified and very close to the original duration, right_offset_frames can be zero;
            # the conditional below is a safe-guard against these cases.
            if right_offset_frames == 0:
                right_offset_frames = None

        # Load and return the features (subset) from the storage
        return storage.read(
            self.storage_key,
            left_offset_frames=left_offset_frames,
            right_offset_frames=right_offset_frames
        )

    def with_path_prefix(self, path: Pathlike) -> 'Features':
        return fastcopy(self, storage_path=str(Path(path) / self.storage_path))

    @staticmethod
    def from_dict(data: dict) -> 'Features':
        return Features(**data)


@dataclass
class FeatureSet(JsonMixin, YamlMixin):
    """
    Represents a feature manifest, and allows to read features for given recordings
    within particular channels and time ranges.
    It also keeps information about the feature extractor parameters used to obtain this set.
    When a given recording/time-range/channel is unavailable, raises a KeyError.
    """
    features: List[Features] = field(default_factory=lambda: list())

    def __post_init__(self):
        self.features = sorted(self.features)

    @staticmethod
    def from_features(features: Iterable[Features]) -> 'FeatureSet':
        return FeatureSet(list(features))  # just for consistency with other *Sets

    @staticmethod
    def from_dicts(data: Iterable[dict]) -> 'FeatureSet':
        return FeatureSet(features=[Features.from_dict(feature_data) for feature_data in data])

    def to_dicts(self) -> List[dict]:
        return [asdict(f) for f in self]

    def with_path_prefix(self, path: Pathlike) -> 'FeatureSet':
        return FeatureSet.from_features(f.with_path_prefix(path) for f in self)

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
        candidates = self._index_by_recording_id_and_cache()[recording_id]
        candidates = (
            f for f in candidates
            if f.channels == channel_id
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

    # This is a cache that significantly speeds up repeated ``find()`` queries.
    _features_by_recording_id: Optional[Dict[str, List[Features]]] = None

    def _index_by_recording_id_and_cache(self):
        if self._features_by_recording_id is None:
            from cytoolz import groupby
            self._features_by_recording_id = groupby(lambda feat: feat.recording_id, self)
        return self._features_by_recording_id

    def load(
            self,
            recording_id: str,
            channel_id: int = 0,
            start: Seconds = 0.0,
            duration: Optional[Seconds] = None,
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
        features = feature_info.load(start=start, duration=duration)
        return features

    def __repr__(self) -> str:
        return f'FeatureSet(len={len(self)})'

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
            storage: FeaturesWriter,
            augmenter: Optional[WavAugmenter] = None
    ):
        self.feature_extractor = feature_extractor
        self.storage = storage
        self.augmenter = augmenter

    def process_and_store_recordings(
            self,
            recordings: Iterable[Recording],
            output_manifest: Optional[Pathlike] = None,
            num_jobs: int = 1
    ) -> FeatureSet:
        if num_jobs == 1:
            # Avoid spawning subprocesses for single threaded processing
            feature_infos = list(chain.from_iterable(map(self._process_and_store_recording, recordings)))
        else:
            with ProcessPoolExecutor(num_jobs) as ex:
                feature_infos = list(chain.from_iterable(ex.map(self._process_and_store_recording, recordings)))
        feature_set = FeatureSet.from_features(feature_infos)
        if output_manifest is not None:
            feature_set.to_json(output_manifest)
        return feature_set

    def _process_and_store_recording(
            self,
            recording: Recording,
    ) -> List[Features]:
        results = []
        for channel in recording.channel_ids:
            results.append(self.feature_extractor.extract_from_recording_and_store(
                recording=recording,
                storage=self.storage,
                channels=channel,
                augmenter=self.augmenter,
            ))
        return results


def store_feature_array(
        feats: np.ndarray,
        storage: FeaturesWriter,
) -> str:
    """
    Store ``feats`` array on disk, using ``lilcom`` compression by default.

    :param feats: a numpy ndarray containing features.
    :param storage: a ``FeaturesWriter`` object to use for array storage.
    :return: a path to the file containing the stored array.
    """
    feats_id = str(uuid4())
    storage_key = storage.write(feats_id, feats)
    return storage_key
