import pickle
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np
import torch

from lhotse.array import Array, TemporalArray
from lhotse.audio import Recording
from lhotse.augmentation import AugmentFn
from lhotse.features.io import FeaturesWriter
from lhotse.serialization import load_yaml, save_to_yaml
from lhotse.utils import (
    Pathlike,
    Seconds,
    asdict_nonull,
    compute_num_frames,
    compute_num_frames_from_samples,
    fastcopy,
    ifnone,
    uuid4,
)


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
        assert is_dataclass(
            config
        ), "The feature configuration object must be a dataclass."
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

    @property
    def device(self) -> Union[str, torch.device]:
        return "cpu"

    @staticmethod
    def mix(
        features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float
    ) -> np.ndarray:
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
        raise ValueError(
            'The feature extractor\'s "mix" operation is undefined. '
            "It does not support feature-domain mix, consider computing the features "
            "after, rather than before mixing the cuts."
        )

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        """
        Compute the mean energy of a feature matrix. How the energy is computed depends on a
        particular type of features.
        It is averaged rather than summed, so that, like ``audio_energy`` for audio, it does not
        depend on the number of frames; otherwise SNRs in feature-domain mixing would depend on
        the length of the mixed signals.
        It is expected that when implemented, ``compute_energy`` will never return zero.

        :param features: A feature matrix.
        :return: A positive float value of the signal energy.
        """
        raise ValueError(
            'The feature extractor\'s "compute_energy" operation is undefined. '
            "It does not support feature-domain mix, consider computing the features "
            "after, rather than before mixing the cuts."
        )

    @staticmethod
    def scale(features: np.ndarray, energy_scaling_factor: float) -> np.ndarray:
        """
        Scale a single feature matrix by the provided energy factor.

        :param features: A feature matrix.
        :param energy_scaling_factor: The energy scaling factor to apply.
        :return: A scaled feature matrix.
        """
        raise ValueError(
            'The feature extractor\'s "scale" operation is undefined. '
            "It does not support feature-domain mix, consider computing the features "
            "after, rather than before mixing the cuts."
        )

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        """
        Performs batch extraction. It is not guaranteed to be faster
        than :meth:`FeatureExtractor.extract` -- it depends on whether
        the implementation of a particular feature extractor supports
        accelerated batch computation. If `lengths` is provided, it is
        assumed that the input is a batch of padded sequences, so we will
        not perform any further collation.

        .. note::
            Unless overridden by child classes, it defaults to sequentially
            calling :meth:`FeatureExtractor.extract` on the inputs.

        .. note::
            This method *should* support variable length inputs.
        """
        input_is_list = False
        input_is_torch = False

        if lengths is not None:
            feat_lens = [
                compute_num_frames_from_samples(l, self.frame_shift, sampling_rate)
                for l in lengths
            ]
            assert isinstance(
                samples, torch.Tensor
            ), "If `lengths` is provided, `samples` must be a batched and padded torch.Tensor."
        else:
            if isinstance(samples, list):
                input_is_list = True
                pass  # nothing to do with `samples`
            elif samples.ndim > 1:
                samples = list(samples)
            else:
                # The user passed an array/tensor of shape (num_samples,)
                samples = [samples.reshape(1, -1)]

            if any(isinstance(x, torch.Tensor) for x in samples):
                samples = [x.numpy() for x in samples]
                input_is_torch = True

        result = []
        for item in samples:
            res = self.extract(item, sampling_rate=sampling_rate)
            if lengths is not None:
                res = res[: feat_lens[len(result)]]
            result.append(res)

        if input_is_torch:
            result = [torch.from_numpy(x) for x in result]

        # If all items are of the same shape, concatenate
        if len(result) == 1:
            if input_is_list:
                return result
            else:
                return result[0]
        elif all(item.shape == result[0].shape for item in result[1:]):
            if input_is_torch:
                return torch.stack(result, dim=0)
            else:
                return np.stack(result, axis=0)
        else:
            return result

    def extract_from_samples_and_store(
        self,
        samples: np.ndarray,
        storage: FeaturesWriter,
        sampling_rate: int,
        offset: Seconds = 0,
        channel: Optional[Union[int, List[int]]] = None,
        augment_fn: Optional[AugmentFn] = None,
    ) -> "Features":
        """
        Extract the features from an array of audio samples in a full pipeline:

        * optional audio augmentation;
        * extract the features;
        * save them to disk in a specified directory;
        * return a ``Features`` object with a description of the extracted features.

        Note, unlike in ``extract_from_recording_and_store``, the returned ``Features`` object
        does not reference any particular ``Recording``. This method is useful when extracting
        features from cuts - especially ``MixedCut`` instances, which may be created from
        multiple recordings and channels.

        :param samples: a numpy ndarray with the audio samples.
        :param sampling_rate: integer sampling rate of ``samples``.
        :param storage: a ``FeaturesWriter`` object that will handle storing the feature matrices.
            When the optional ``lilcom`` dependency is installed and on-disk size matters,
            ``LilcomChunkyWriter`` is the preferred backend.
        :param offset: an offset in seconds for where to start reading the recording - when used for
            ``Cut`` feature extraction, must be equal to ``Cut.start``.
        :param channel: an optional channel number(s) to insert into ``Features`` manifest.
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
            array=Array(
                storage_type=storage.name,
                storage_path=str(storage.storage_path),
                storage_key=storage_key,
                shape=list(feats.shape),
            ),
            start=offset,
            duration=duration,
            type=self.name,
            frame_shift=self.frame_shift,
            sampling_rate=sampling_rate,
            channels=channel,
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
    ) -> "Features":
        """
        Extract the features from a ``Recording`` in a full pipeline:

        * load audio from disk;
        * optionally, perform audio augmentation;
        * extract the features;
        * save them to disk in a specified directory;
        * return a ``Features`` object with a description of the extracted features and the source data used.

        :param recording: a ``Recording`` that specifies what's the input audio.
        :param storage: a ``FeaturesWriter`` object that will handle storing the feature matrices.
            When the optional ``lilcom`` dependency is installed and on-disk size matters,
            ``LilcomChunkyWriter`` is the preferred backend.
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
            duration=recording.duration if duration is not None else recording.duration,
            type=self.name,
            frame_shift=self.frame_shift,
            sampling_rate=recording.sampling_rate,
            array=Array(
                storage_type=storage.name,
                storage_path=str(storage.storage_path),
                storage_key=storage_key,
                shape=list(feats.shape),
            ),
        )
        validate_features(manifest, feats_data=feats)
        return manifest

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureExtractor":
        feature_type = data.pop("feature_type")
        extractor_type = get_extractor_type(feature_type)
        # noinspection PyUnresolvedReferences
        config = extractor_type.config_type.from_dict(data)
        return extractor_type(config)

    def to_dict(self) -> Dict[str, Any]:
        d = self.config.to_dict()
        d["feature_type"] = self.name  # Insert the typename for config readability
        return d

    @classmethod
    def from_yaml(cls, path: Pathlike) -> "FeatureExtractor":
        return cls.from_dict(load_yaml(path))

    def to_yaml(self, path: Pathlike):
        data = self.to_dict()
        # Some feature extractors might have a "device" field:
        # to make sure they get nicely serialized to YAML, we will convert
        # the torch.device object to its string type.
        # Note: we don't store the device ID (e.g. we change "cuda:1" to "cuda")
        # so that the config remains valid even if we use it in a separate run
        # on a different device.
        if "device" in data and isinstance(data["device"], torch.device):
            data["device"] = data["device"].type
        save_to_yaml(data, path=path)


FEATURE_EXTRACTORS = {}


def get_extractor_type(name: str) -> Type:
    """
    Return the feature extractor type corresponding to the given name.

    :param name: specifies which feature extractor should be used.
    :return: A feature extractors type.
    """
    return FEATURE_EXTRACTORS[name]


def create_default_feature_extractor(name: str) -> "Optional[FeatureExtractor]":
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

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        params = asdict(self.config)
        params.update({"sample_frequency": sampling_rate, "snip_edges": False})
        params["frame_shift"] *= 1000.0
        params["frame_length"] *= 1000.0
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        # Torchaudio Kaldi feature extractors expect the channel dimension to be first.
        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        features = self._feature_fn(samples, **params).to(torch.float32)
        return features.numpy()

    def _feature_fn(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift


@dataclass(init=False, repr=False)
class Features(TemporalArray):
    """
    Represents features extracted for some particular time range in a given recording and channel.

    :class:`.Features` is a :class:`~lhotse.array.TemporalArray` whose first axis is time,
    i.e., its shape is ``(num_frames, num_features)``. On top of that, it describes how the
    features were obtained: their ``type`` (e.g. ``"kaldi-fbank"``), the ``sampling_rate``
    and ``duration`` of the source audio, and which ``recording_id`` and ``channels`` it came from.

    For backward compatibility, it also accepts and exposes the flat fields used before
    it became a :class:`~lhotse.array.TemporalArray`: ``num_frames``, ``num_features``,
    ``storage_type``, ``storage_path``, and ``storage_key``.
    :meth:`from_dict` reads manifests in either layout.
    """

    # Fields inherited from TemporalArray: array, temporal_dim, frame_shift, start.

    # The kind of features (e.g. "kaldi-fbank") and the sampling rate of the source audio.
    type: str
    sampling_rate: int

    # Duration of the audio the features were extracted from.
    # It may differ from ``num_frames * frame_shift`` by a fraction of a frame.
    # The default value shadows TemporalArray.duration property, which would compute it from shape.
    duration: Seconds = None

    # Information which recording and channels were used to extract the features.
    # When ``recording_id`` and ``channels`` are ``None``, it means that the
    # features were extracted from a cut (e.g. a ``MixedCut``), which might have consisted
    # of multiple recordings.
    recording_id: Optional[str] = None
    channels: Optional[Union[int, List[int]]] = None

    _start_tolerance = 1e-3

    def __init__(
        self,
        *,
        type: str,
        frame_shift: Seconds,
        sampling_rate: int,
        start: Seconds,
        array: Optional[Array] = None,
        temporal_dim: int = 0,
        duration: Optional[Seconds] = None,
        recording_id: Optional[str] = None,
        channels: Optional[Union[int, List[int]]] = None,
        # Legacy flat fields describing ``array``; when ``array`` is given, they override it.
        num_frames: Optional[int] = None,
        num_features: Optional[int] = None,
        storage_type: Optional[str] = None,
        storage_path: Optional[str] = None,
        storage_key: Optional[Union[str, bytes]] = None,
    ) -> None:
        legacy = {
            "storage_type": storage_type,
            "storage_path": storage_path,
            "storage_key": storage_key,
        }
        if array is None:
            array = Array(shape=[num_frames, num_features], **legacy)
        else:
            overrides = {k: v for k, v in legacy.items() if v is not None}
            if num_frames is not None or num_features is not None:
                overrides["shape"] = [
                    ifnone(num_frames, array.shape[0]),
                    ifnone(num_features, array.shape[1]),
                    *array.shape[2:],
                ]
            if overrides:
                array = fastcopy(array, **overrides)
        self.array = array
        self.temporal_dim = temporal_dim
        self.frame_shift = frame_shift
        self.start = start
        self.type = type
        self.sampling_rate = sampling_rate
        self.duration = ifnone(duration, self.num_frames * frame_shift)
        self.recording_id = recording_id
        self.channels = channels

    @property
    def num_features(self) -> int:
        return self.shape[1]

    @property
    def storage_type(self) -> str:
        return self.array.storage_type

    @storage_type.setter
    def storage_type(self, value: str) -> None:
        self.array = fastcopy(self.array, storage_type=value)

    @property
    def storage_path(self) -> str:
        return self.array.storage_path

    @storage_path.setter
    def storage_path(self, value: str) -> None:
        self.array = fastcopy(self.array, storage_path=value)

    @property
    def storage_key(self) -> Union[str, bytes]:
        return self.array.storage_key

    @storage_key.setter
    def storage_key(self, value: Union[str, bytes]) -> None:
        self.array = fastcopy(self.array, storage_key=value)

    def load(
        self,
        start: Optional[Seconds] = None,
        duration: Optional[Seconds] = None,
        channel_id: Union[int, List[int]] = 0,
    ) -> np.ndarray:
        """
        Load the feature matrix, or its ``[start, start + duration)`` sub-span.
        ``channel_id`` is accepted for backward compatibility and ignored.
        """
        return super().load(start=start, duration=duration)

    def _seconds_to_frames(self, seconds: Seconds) -> int:
        # Count frames the same way as the feature extractors do.
        return compute_num_frames(
            seconds, frame_shift=self.frame_shift, sampling_rate=self.sampling_rate
        )

    def move_to_memory(
        self,
        start: Seconds = 0,
        duration: Optional[Seconds] = None,
        lilcom: bool = False,
    ) -> "Features":
        out = super().move_to_memory(start=start, duration=duration, lilcom=lilcom)
        if out is not self and duration is not None:
            out.duration = duration
        return out

    def to_dict(self) -> dict:
        return asdict_nonull(self)

    def copy_feats(self, writer: FeaturesWriter) -> "Features":
        """
        Read the referenced feature array and save it using ``writer``.
        Returns a copy of the manifest with updated fields related to the feature storage.
        """
        feats = self.load()
        new_key = writer.write(self.storage_key, feats)
        return fastcopy(
            self,
            array=fastcopy(
                self.array,
                storage_type=writer.name,
                storage_path=writer.storage_path,
                storage_key=new_key,
            ),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Features":
        if "array" in data:
            return cls(**{**data, "array": Array.from_dict(data["array"])})
        # Legacy flat layout, written before Features became a TemporalArray.
        # The "storage_type" check is to ensure that the "data" dict actually contains
        # the data for a "Features" object, and not something else.
        # Some Lhotse utilities try to "guess" what is the right object type via trial-and-error,
        # and would have created a false alarm here.
        if "frame_shift" not in data and "storage_type" in data:
            warnings.warn(
                'The "frame_shift" field was not found in a feature manifest; '
                "we'll try to infer it for now, but you should recreate the manifests."
            )
            data["frame_shift"] = round(
                data["duration"] / data["num_frames"], ndigits=3
            )
        return cls(**data)

    def __setstate__(self, state: dict) -> None:
        # Objects pickled before Features became a TemporalArray hold the legacy flat fields.
        if "array" not in state:
            state = vars(Features(**state))
        self.__dict__.update(state)

    def __repr__(self):
        return (
            f"Features("
            f"type='{self.type}', "
            f"num_frames={self.num_frames}, "
            f"num_features={self.num_features}, "
            f"frame_shift={self.frame_shift}, "
            f"sampling_rate={self.sampling_rate}, "
            f"start={self.start}, "
            f"duration={self.duration}, "
            f"storage_type='{self.storage_type}', "
            f"storage_path='{self.storage_path}', "
            f"storage_key='{self.storage_key if isinstance(self.storage_key, str) else '<binary-data>'}', "
            f"recording_id='{self.recording_id}', "
            f"channels={self.channels}"
            f")"
        )


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
    feature_manifests: Iterable[Features], storage_path: Optional[Pathlike] = None
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
    stats = StatsAccumulator(feature_dim=first.num_features)
    for features in chain([first], feature_manifests):
        # Read the features
        arr = features.load().astype(np.float64)
        # Update
        stats.update(arr)
    mvn = stats.get()
    if storage_path is not None:
        with open(storage_path, "wb") as f:
            pickle.dump(mvn, f)
    return mvn


class StatsAccumulator:
    def __init__(self, feature_dim: int):
        self.total_sum = np.zeros((feature_dim,), dtype=np.float64)
        self.total_unnorm_var = np.zeros((feature_dim,), dtype=np.float64)
        self.total_frames = 0

    def update(self, arr: np.ndarray) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            arr = arr.astype(np.float64)
            # Update the sum for the means
            curr_sum = arr.sum(axis=0)
            updated_total_sum = self.total_sum + curr_sum
            # Update the number of frames
            curr_frames = arr.shape[0]
            updated_total_frames = self.total_frames + curr_frames
            # Update the unnormalized variance
            total_over_curr_frames = self.total_frames / curr_frames
            curr_unnorm_var = np.var(arr, axis=0) * curr_frames
            if self.total_frames > 0:
                self.total_unnorm_var = (
                    self.total_unnorm_var
                    + curr_unnorm_var
                    + total_over_curr_frames
                    / updated_total_frames
                    * (self.total_sum / total_over_curr_frames - curr_sum) ** 2
                )
            else:
                self.total_unnorm_var = curr_unnorm_var
            self.total_sum = updated_total_sum
            self.total_frames = updated_total_frames

    @property
    def norm_means(self) -> np.ndarray:
        return self.total_sum / self.total_frames

    @property
    def norm_stds(self) -> np.ndarray:
        return np.sqrt(self.total_unnorm_var / self.total_frames)

    def get(self) -> Dict[str, np.ndarray]:
        return {
            "norm_means": self.norm_means,
            "norm_stds": self.norm_stds,
        }
