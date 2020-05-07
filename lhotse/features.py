"""
This file is just a rough sketch for now.
"""
from dataclasses import dataclass, field, asdict
from typing import Union, List, Iterable, Dict

import numpy as np
import torch
import torchaudio
import yaml

from lhotse.utils import Seconds, Milliseconds, Pathlike


@dataclass
class SpectrogramConfig:
    dither: float = 0.0
    window_type: str = "povey"
    frame_length: Milliseconds = 25.0
    frame_shift: Milliseconds = 10.0
    remove_dc_offset: bool = False
    round_to_power_of_two: bool = False
    energy_floor: float = 1.0
    min_duration: float = 0.0
    preemphasis_coefficient: float = 0.97
    raw_energy: bool = True
    snip_edges: bool = True


@dataclass
class MfccFbankCommonConfig:
    low_freq: float = 20.0
    high_freq: float = 0.0
    num_mel_bins: int = 23
    use_energy: bool = True
    vtln_low: float = 100.0
    vtln_high: float = -500.0
    vtln_warp: float = 1.0


@dataclass
class FbankSpecificConfig:
    use_log_fbank: bool = True


@dataclass
class MfccSpecificConfig:
    cepstral_lifter: float = 22.0


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

        return feature_fn(waveform=samples, **params)


@dataclass
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

    def load(self) -> np.ndarray:
        pass


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
            channel_id: int,
            start: Seconds,
            duration: Seconds,
    ) -> np.ndarray:
        # raise a KeyError when any of the requirements is not met
        pass

    def __iter__(self) -> Iterable[Features]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)
