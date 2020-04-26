from dataclasses import dataclass, asdict
from subprocess import run, PIPE
from typing import List, Optional, Dict, Union

import librosa
import numpy as np
import yaml

from lhotse.utils import Pathlike, INT16MAX, DummySet

Channels = Union[int, List[int]]


@dataclass
class AudioSource:
    """
    AudioSource represents audio data that can be retrieved from somewhere.
    Supported sources of audio are currently:
    - a file (possibly multi-channel)
    - a command/unix pipe (single-channel only)
    - a collection of any of the above (see AudioSourceCollection)
    """
    type: str
    channel_ids: List[int]
    source: str

    def load_audio(
            self,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None
    ) -> np.ndarray:
        assert self.type in ('file', 'command')

        if self.type == 'file':
            # TODO(pzelasko): make sure that librosa loads multi-channel audio
            #                 in the expected format (n_channels, n_samples)
            return librosa.load(
                self.source,
                sr=None,  # 'None' uses the native sampling rate
                offset=offset_seconds,
                duration=duration_seconds
            )[0]  # discard returned sampling rate

        # TODO(pzelasko): the following naively assumes we're dealing with raw PCM...
        #                 not sure if that's how we should do it
        #                 also, how should we support chunking for commands?
        raw_audio = run(self.source, shell=True, stdout=PIPE).stdout
        int16_audio = np.frombuffer(raw_audio, dtype=np.int16)
        return int16_audio / INT16MAX


@dataclass
class Recording:
    """
    Recording represents an AudioSource along with some metadata.
    """
    id: str
    sources: List[AudioSource]
    sampling_rate: int
    num_samples: int
    duration_seconds: float

    def __post_init__(self):
        self.sources = [AudioSource(**s) if isinstance(s, dict) else s for s in self.sources]

    @property
    def num_channels(self):
        return sum(len(source.channel_ids) for source in self.sources)

    def load_audio(
            self,
            channels: Optional[Channels] = None,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None
    ) -> np.ndarray:
        if channels is None:
            channels = DummySet()
        elif isinstance(channels, int):
            channels = frozenset([channels])
        else:
            channels = frozenset(channels)

        samples_per_source = []
        for source in self.sources:
            # Case: source not requested
            if not channels.intersection(source.channel_ids):
                continue
            samples = source.load_audio(offset_seconds=offset_seconds, duration_seconds=duration_seconds)

            # Case: two-channel audio file but only one channel requested
            #       it might not be optimal to load all channels, but IDK if there's anything we can do about it
            channels_to_remove = [
                idx for idx, cid in enumerate(source.channel_ids)
                if cid not in channels
            ]
            if channels_to_remove:
                samples = np.delete(samples, channels_to_remove, axis=0)
            samples_per_source.append(samples)

        # shapes: single-channel (n_samples); multi-channel (n_channels, n_samples)
        return np.vstack(samples_per_source) if len(samples_per_source) > 1 else samples_per_source[0]


@dataclass
class AudioSet:
    """
    AudioSet represents a dataset of recordings. It does not contain any annotation -
    just the information needed to retrieve a recording (possibly multi-channel, from files
    or from shell commands and pipes) and some metadata for each of them.

    It also supports (de)serialization to/from YAML and takes care of mapping between
    rich Python classes and YAML primitives during conversion.
    """
    recordings: Dict[str, Recording]

    @staticmethod
    def from_yaml(path: Pathlike) -> 'AudioSet':
        with open(path) as f:
            recordings = (Recording(**raw_rec) for raw_rec in yaml.safe_load(f))
        return AudioSet(recordings={r.id: r for r in recordings})

    def to_yaml(self, path: Pathlike):
        with open(path, 'w') as f:
            yaml.safe_dump([asdict(r) for r in self.recordings.values()], stream=f)

    def load_audio(
            self,
            recording_id: str,
            channels: Optional[Channels] = None,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None
    ) -> np.ndarray:
        return self.recordings[recording_id].load_audio(
            channels=channels,
            offset_seconds=offset_seconds,
            duration_seconds=duration_seconds
        )

    def num_channels(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_channels

    def sampling_rate(self, recording_id: str) -> int:
        return self.recordings[recording_id].sampling_rate

    def num_samples(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_samples

    def duration_seconds(self, recording_id: str) -> float:
        return self.recordings[recording_id].duration_seconds

    def __iter__(self):
        return iter(self.recordings.values())
