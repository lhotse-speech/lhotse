from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Union, Optional, Sequence

import numpy as np
import warnings
from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import is_module_available, Seconds, compute_num_frames


@dataclass
class OpenSmileConfig:
    """
    OpenSmile configs are stored in separated txt files in its specific format.
    You can specify predefined config by setting ``feature_set`` and ``feature_level``
    class attributes with:
    (1) ``FeatureSet`` and ``FeatureLevel`` classes predefined in
    https://github.com/audeering/opensmile-python/blob/master/opensmile/core/define.py
    OR
    (2) strings refered to enum members,
    In opensmile-python You can also create your own config file and pass its path and
    corresponding feature level as documented here
    https://audeering.github.io/opensmile-python/usage.html#custom-config.
    For now custom configs are not supported in this extractor.
    """

    feature_set: Union[str, Any] = "ComParE_2016"  # default feature set or
    # string with set name
    feature_level: Union[str, Any] = "lld"  # default feature level or level name
    options: Optional[dict] = None  # dictionary with optional script parameters
    loglevel: int = 2  # log level (0-5), the higher the number the more log
    # messages are given
    logfile: Optional[str] = None  # if not ``None`` log messages will be
    # stored to this file
    sampling_rate: Optional[int] = None  # If ``None`` it will call ``process_func``
    # with the actual sampling rate of the signal.
    channels: Union[int, Sequence[int]] = 0
    mixdown: bool = False  # apply mono mix-down on selection
    resample: bool = False  # if ``True`` enforces given sampling rate by resampling
    num_workers: Optional[int] = 1  # number of parallel jobs or 1 for sequential
    # processing. If ``None`` will be set to the number of processors
    verbose: bool = False  # show debug messages

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OpenSmileConfig":
        return OpenSmileConfig(**data)

    @staticmethod
    def featuresets_names():
        """
        Returns list of strings with names of pretrained FeatureSets available in opensmile.
        """
        assert is_module_available(
            "opensmile"
        ), 'To use opensmile extractors, please "pip install opensmile" first.'
        import opensmile

        return list(opensmile.FeatureSet.__members__)


@register_extractor
class OpenSmileExtractor(FeatureExtractor):
    """Wrapper for extraction of features implemented in OpenSmile."""

    name = "opensmile-extractor"
    config_type = OpenSmileConfig

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config=config)
        assert is_module_available(
            "opensmile"
        ), 'To use opensmile extractors, please "pip install opensmile" first.'
        import opensmile

        if isinstance(self.config.feature_set, str):
            self.feature_set = opensmile.FeatureSet[self.config.feature_set]
        else:
            self.feature_set = self.config.feature_set
        self.feature_level = opensmile.FeatureLevel(self.config.feature_level)
        self.smileExtractor = opensmile.Smile(
            feature_set=self.feature_set,
            feature_level=self.feature_level,
            sampling_rate=self.config.sampling_rate,
            options=self.config.options,
            loglevel=self.config.loglevel,
            logfile=self.config.logfile,
            channels=self.config.channels,
            mixdown=self.config.mixdown,
            resample=self.config.resample,
            num_workers=self.config.num_workers,
            verbose=self.config.verbose,
        )

    @property
    def feature_names(self) -> List[str]:
        return self.smileExtractor.feature_names

    def is_lld_or_lld_de(self) -> bool:
        from opensmile import FeatureLevel

        return (
            self.feature_level is FeatureLevel.LowLevelDescriptors
            or self.feature_level is FeatureLevel.LowLevelDescriptors_Deltas
        )

    @property
    def frame_shift(self) -> Seconds:
        import opensmile

        if (
            self.is_lld_or_lld_de()
            and self.feature_set in opensmile.FeatureSet.__members__.values()
        ):
            # For all deafult opensmile configs frameshift is equal to 10 ms
            return 0.01
        else:
            raise NotImplementedError(
                f"frame_shift is not defined for Functionals feature level or for non default feature set. Defined featureset: {self.config.feature_set}"
            )

    def feature_dim(self, sampling_rate: int) -> int:
        return len(self.feature_names)

    def feature_names(self) -> List[str]:
        return self.smileExtractor.feature_names()

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        if (
            self.config.sampling_rate is not None
            and self.config.sampling_rate != sampling_rate
        ):
            raise ValueError(
                f"Given sampling rate ({sampling_rate}) mismatched with the value set in OpenSmileConfig ({self.config.sampling_rate})."
            )
        import opensmile

        feats = self.smileExtractor.process_signal(
            samples, sampling_rate=sampling_rate
        ).to_numpy()

        if self.is_lld_or_lld_de():
            feats = self._pad_frames(samples, feats, sampling_rate)

        return feats.copy()

    def _pad_frames(
        self, samples: np.ndarray, feats: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """Adds last diff frames to the end of feats matrix to fit lhotse.utils.compute_num_frames."""
        duration = np.shape(samples)[1] / sampling_rate
        diff = (
            compute_num_frames(duration, self.frame_shift, sampling_rate)
            - np.shape(feats)[0]
        )
        if abs(diff) >= 6:
            warnings.warn(f"Unusual difference in number of frames: {diff}")
        if diff > 0:
            feats = np.append(feats, feats[-diff:, :], axis=0)
        elif diff < 0:
            feats = feats[:-diff, :]
        return feats
