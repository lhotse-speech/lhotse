import logging
import multiprocessing
import pickle
import warnings
from abc import ABCMeta, abstractmethod
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from itertools import chain
from math import isclose
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from lhotse.audio import Recording
from lhotse.augmentation import AugmentFn
from lhotse.features.io import FeaturesWriter, get_reader
from lhotse.serialization import Serializable, load_yaml, save_to_yaml
from lhotse.utils import (Pathlike, Seconds, asdict_nonull, compute_num_frames, exactly_one_not_null, fastcopy,
                          ifnone, split_sequence,
                          uuid4)


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
    def frame_shift(self) -> Seconds:
        ...

    @abstractmethod
    def feature_dim(self, sampling_rate: int) -> int:
        ...

    @staticmethod
    def mix(features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float) -> np.ndarray:
        """
        Perform feature-domain mix of two signals, ``a`` and ``b``, and return the mixed signal.

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
        raise ValueError('The feature extractor\'s "mix" operation is undefined. '
                         'It does not support feature-domain mix, consider computing the features '
                         'after, rather than before mixing the cuts.')

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        """
        Compute the total energy of a feature matrix. How the energy is computed depends on a
        particular type of features.
        It is expected that when implemented, ``compute_energy`` will never return zero.

        :param features: A feature matrix.
        :return: A positive float value of the signal energy.
        """
        raise ValueError('The feature extractor\'s "compute_energy" operation is undefined. '
                         'It does not support feature-domain mix, consider computing the features '
                         'after, rather than before mixing the cuts.')

    def extract_from_samples_and_store(
            self,
            samples: np.ndarray,
            storage: FeaturesWriter,
            sampling_rate: int,
            offset: Seconds = 0,
            channel: Optional[int] = None,
            augment_fn: Optional[AugmentFn] = None,
    ) -> 'Features':
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
        :param channel: an optional channel number to insert into ``Features`` manifest.
        :param augment_fn: an optional ``WavAugmenter`` instance to modify the waveform before feature extraction.
        :return: a ``Features`` manifest item for the extracted feature matrix (it is not written to disk).
        """
        from lhotse.qa import validate_features
        if augment_fn is not None:
            samples = augment_fn(samples, sampling_rate)
        duration = round(samples.shape[1] / sampling_rate, ndigits=8)
        feats = self.extract(samples=samples, sampling_rate=sampling_rate)
        storage_key = store_feature_array(feats, storage=storage)
        manifest = Features(
            start=offset,
            duration=duration,
            type=self.name,
            num_frames=feats.shape[0],
            num_features=feats.shape[1],
            frame_shift=self.frame_shift,
            sampling_rate=sampling_rate,
            channels=channel,
            storage_type=storage.name,
            storage_path=str(storage.storage_path),
            storage_key=storage_key
        )
        validate_features(manifest, feats_data=feats)
        return manifest

    def extract_from_recording_and_store(
            self,
            recording: Recording,
            storage: FeaturesWriter,
            offset: Seconds = 0,
            duration: Optional[Seconds] = None,
            channels: Union[int, List[int]] = None,
            augment_fn: Optional[AugmentFn] = None,
    ) -> 'Features':
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
        :param augment_fn: an optional ``WavAugmenter`` instance to modify the waveform before feature extraction.
        :return: a ``Features`` manifest item for the extracted feature matrix.
        """
        from lhotse.qa import validate_features
        samples = recording.load_audio(
            offset=offset,
            duration=duration,
            channels=channels,
        )
        if augment_fn is not None:
            samples = augment_fn(samples, recording.sampling_rate)
        feats = self.extract(samples=samples, sampling_rate=recording.sampling_rate)
        storage_key = store_feature_array(feats, storage=storage)
        manifest = Features(
            recording_id=recording.id,
            channels=channels if channels is not None else recording.channel_ids,
            # The start is relative to the beginning of the recording.
            start=offset,
            duration=recording.duration,
            type=self.name,
            num_frames=feats.shape[0],
            num_features=feats.shape[1],
            frame_shift=self.frame_shift,
            sampling_rate=recording.sampling_rate,
            storage_type=storage.name,
            storage_path=str(storage.storage_path),
            storage_key=storage_key
        )
        validate_features(manifest, feats_data=feats)
        return manifest

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

    def extract(self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> np.ndarray:
        params = asdict(self.config)
        params.update({
            "sample_frequency": sampling_rate,
            "snip_edges": False
        })
        params['frame_shift'] *= 1000.0
        params['frame_length'] *= 1000.0
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        # Torchaudio Kaldi feature extractors expect the channel dimension to be first.
        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
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
    frame_shift: Seconds
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
            left_offset_frames = compute_num_frames(start - self.start, frame_shift=self.frame_shift,
                                                    sampling_rate=self.sampling_rate)
        # Right trim
        if duration is not None:
            right_offset_frames = left_offset_frames + compute_num_frames(duration, frame_shift=self.frame_shift,
                                                                          sampling_rate=self.sampling_rate)

        # Load and return the features (subset) from the storage
        return storage.read(
            self.storage_key,
            left_offset_frames=left_offset_frames,
            right_offset_frames=right_offset_frames
        )

    def with_path_prefix(self, path: Pathlike) -> 'Features':
        return fastcopy(self, storage_path=str(Path(path) / self.storage_path))

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    @staticmethod
    def from_dict(data: dict) -> 'Features':
        # The "storage_type" check is to ensure that the "data" dict actually contains
        # the data for a "Features" object, and not something else.
        # Some Lhotse utilities try to "guess" what is the right object type via trial-and-error,
        # and would have created a false alarm here.
        if 'frame_shift' not in data and 'storage_type' in data:
            warnings.warn('The "frame_shift" field was not found in a feature manifest; '
                          'we\'ll try to infer it for now, but you should recreate the manifests.')
            data['frame_shift'] = round(data['duration'] / data['num_frames'], ndigits=3)
        return Features(**data)


class FeatureSet(Serializable, Sequence[Features]):
    """
    Represents a feature manifest, and allows to read features for given recordings
    within particular channels and time ranges.
    It also keeps information about the feature extractor parameters used to obtain this set.
    When a given recording/time-range/channel is unavailable, raises a KeyError.
    """

    def __init__(self, features: List[Features] = None) -> None:
        self.features = sorted(ifnone(features, []))

    def __eq__(self, other: 'FeatureSet') -> bool:
        return self.features == other.features

    @staticmethod
    def from_features(features: Iterable[Features]) -> 'FeatureSet':
        return FeatureSet(list(features))  # just for consistency with other *Sets

    @staticmethod
    def from_dicts(data: Iterable[dict]) -> 'FeatureSet':
        return FeatureSet(features=[Features.from_dict(feature_data) for feature_data in data])

    def to_dicts(self) -> Iterable[dict]:
        return (f.to_dict() for f in self)

    def with_path_prefix(self, path: Pathlike) -> 'FeatureSet':
        return FeatureSet.from_features(f.with_path_prefix(path) for f in self)

    def split(self, num_splits: int, shuffle: bool = False, drop_last: bool = False) -> List['FeatureSet']:
        """
        Split the :class:`~lhotse.FeatureSet` into ``num_splits`` pieces of equal size.

        :param num_splits: Requested number of splits.
        :param shuffle: Optionally shuffle the recordings order first.
        :param drop_last: determines how to handle splitting when ``len(seq)`` is not divisible
            by ``num_splits``. When ``False`` (default), the splits might have unequal lengths.
            When ``True``, it may discard the last element in some splits to ensure they are
            equally long.
        :return: A list of :class:`~lhotse.FeatureSet` pieces.
        """
        return [
            FeatureSet.from_features(subset) for subset in
            split_sequence(self, num_splits=num_splits, shuffle=shuffle, drop_last=drop_last)
        ]

    def subset(self, first: Optional[int] = None, last: Optional[int] = None) -> 'FeatureSet':
        """
        Return a new ``FeatureSet`` according to the selected subset criterion.
        Only a single argument to ``subset`` is supported at this time.

        :param first: int, the number of first supervisions to keep.
        :param last: int, the number of last supervisions to keep.
        :return: a new ``FeatureSet`` with the subset results.
        """
        assert exactly_one_not_null(first, last), "subset() can handle only one non-None arg."

        if first is not None:
            assert first > 0
            if first > len(self):
                logging.warning(f'FeatureSet has only {len(self)} items but first {first} required; '
                                f'not doing anything.')
                return self
            return FeatureSet.from_features(self.features[:first])

        if last is not None:
            assert last > 0
            if last > len(self):
                logging.warning(f'FeatureSet has only {len(self)} items but last {last} required; '
                                f'not doing anything.')
                return self
            return FeatureSet.from_features(self.features[-last:])

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
            It is necessary to keep a small positive value here (default 0.05s), as there might be differences between
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

    def compute_global_stats(self, storage_path: Optional[Pathlike] = None) -> Dict[str, np.ndarray]:
        """
        Compute the global means and standard deviations for each feature bin in the manifest.
        It follows the implementation in scikit-learn:
        https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/utils/extmath.py#L715
        which follows the paper:
        "Algorithms for computing the sample variance: analysis and recommendations", by Chan, Golub, and LeVeque.

        :param storage_path: an optional path to a file where the stats will be stored with pickle.
        :return a dict of ``{'norm_means': np.ndarray, 'norm_stds': np.ndarray}`` with the
            shape of the arrays equal to the number of feature bins in this manifest.
        """
        return compute_global_stats(feature_manifests=self, storage_path=storage_path)

    def __repr__(self) -> str:
        return f'FeatureSet(len={len(self)})'

    def __iter__(self) -> Iterable[Features]:
        return iter(self.features)

    def __getitem__(self, i: int) -> Features:
        return self.features[i]

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
            augment_fn: Optional[AugmentFn] = None
    ):
        self.feature_extractor = feature_extractor
        self.storage = storage
        self.augment_fn = augment_fn

    def process_and_store_recordings(
            self,
            recordings: Sequence[Recording],
            output_manifest: Optional[Pathlike] = None,
            num_jobs: int = 1
    ) -> FeatureSet:
        if num_jobs == 1:
            # Avoid spawning subprocesses for single threaded processing
            feature_set = FeatureSet.from_features(
                tqdm(
                    chain.from_iterable(
                        map(self._process_and_store_recording, recordings)
                    ),
                    total=len(recordings),
                    desc='Extracting and storing features'
                )
            )
        else:
            with ProcessPoolExecutor(num_jobs, mp_context=multiprocessing.get_context('spawn')) as ex:
                feature_set = FeatureSet.from_features(
                    tqdm(
                        chain.from_iterable(
                            ex.map(self._process_and_store_recording, recordings)
                        ),
                        total=len(recordings),
                        desc='Extracting and storing features in parallel'
                    )
                )
        if output_manifest is not None:
            feature_set.to_file(output_manifest)
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
                augment_fn=self.augment_fn,
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


def compute_global_stats(
        feature_manifests: Iterable[Features],
        storage_path: Optional[Pathlike] = None
) -> Dict[str, np.ndarray]:
    """
    Compute the global means and standard deviations for each feature bin in the manifest.
    It performs only a single pass over the data and iteratively updates the estimate of the
    means and variances.

    We follow the implementation in scikit-learn:
    https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/utils/extmath.py#L715
    which follows the paper:
    "Algorithms for computing the sample variance: analysis and recommendations", by Chan, Golub, and LeVeque.

    :param feature_manifests: an iterable of ``Features`` objects.
    :param storage_path: an optional path to a file where the stats will be stored with pickle.
    :return a dict of ``{'norm_means': np.ndarray, 'norm_stds': np.ndarray}`` with the
        shape of the arrays equal to the number of feature bins in this manifest.
    """
    feature_manifests = iter(feature_manifests)
    first = next(feature_manifests)
    total_sum = np.zeros((first.num_features,), dtype=np.float64)
    total_unnorm_var = np.zeros((first.num_features,), dtype=np.float64)
    total_frames = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        for features in chain([first], feature_manifests):
            # Read the features
            arr = features.load().astype(np.float64)
            # Update the sum for the means
            curr_sum = arr.sum(axis=0)
            updated_total_sum = total_sum + curr_sum
            # Update the number of frames
            curr_frames = arr.shape[0]
            updated_total_frames = total_frames + curr_frames
            # Update the unnormalized variance
            total_over_curr_frames = total_frames / curr_frames
            curr_unnorm_var = np.var(arr, axis=0) * curr_frames
            if total_frames > 0:
                total_unnorm_var = (
                        total_unnorm_var + curr_unnorm_var +
                        total_over_curr_frames / updated_total_frames *
                        (total_sum / total_over_curr_frames - curr_sum) ** 2)
            else:
                total_unnorm_var = curr_unnorm_var
            total_sum = updated_total_sum
            total_frames = updated_total_frames
    stats = {
        'norm_means': total_sum / total_frames,
        'norm_stds': np.sqrt(total_unnorm_var / total_frames)
    }
    if storage_path is not None:
        with open(storage_path, 'wb') as f:
            pickle.dump(stats, f)
    return stats
