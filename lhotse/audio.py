import os
import warnings
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from subprocess import run, PIPE
from typing import List, Optional, Dict, Union, Iterable

# Workaround for SoundFile (librosa dep) raising exception when a native library, libsndfile1, is not installed.
# Read-the-docs does not allow to modify the Docker containers used to build documentation...
if not os.environ['READTHEDOCS']:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import librosa
import numpy as np

from lhotse.utils import Pathlike, SetContainingAnything, Seconds, load_yaml, save_to_yaml

Channels = Union[int, List[int]]


# TODO: document the dataclasses like this:
# https://stackoverflow.com/a/3051356/5285891


@dataclass
class AudioSource:
    """
    AudioSource represents audio data that can be retrieved from somewhere.
    Supported sources of audio are currently:
    - 'file' (formats supported by librosa, possibly multi-channel)
    - 'command' [unix pipe] (must be WAVE, possibly multi-channel)
    """
    type: str
    channel_ids: List[int]
    source: str

    def load_audio(
            self,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None,
            root_dir: Optional[Pathlike] = None,
    ) -> np.ndarray:
        """
        Load the AudioSource (both files and commands) with librosa,
        accounting for many audio formats and multi-channel inputs.
        Returns numpy array with shapes: (n_samples) for single-channel,
        (n_channels, n_samples) for multi-channel.
        """
        assert self.type in ('file', 'command')

        if self.type == 'command':
            if offset_seconds != 0.0 or duration_seconds is not None:
                # TODO(pzelasko): How should we support chunking for commands?
                #                 We risk being very inefficient when reading many chunks from the same file
                #                 without some caching scheme, because we'll be re-running commands.
                raise ValueError("Reading audio chunks from command AudioSource type is currently not supported.")
            # TODO: consider adding support for "root_dir" in "command" audio source type
            if root_dir is not None:
                warnings.warn('"root_dir" argument is currently not supported in "command" AudioSource type')
            source = BytesIO(run(self.source, shell=True, stdout=PIPE).stdout)
        else:
            source = self.source if root_dir is None else Path(root_dir) / self.source

        samples, sampling_rate = librosa.load(
            source,
            sr=None,  # 'None' uses the native sampling rate
            mono=False,  # Retain multi-channel if it's there
            offset=offset_seconds,
            duration=duration_seconds
        )

        # explicit sanity check for duration as librosa does not complain here
        if duration_seconds is not None:
            num_samples = samples.shape[0] if len(samples.shape) == 1 else samples.shape[1]
            available_duration = num_samples / sampling_rate
            if available_duration < duration_seconds - 1e-5:
                raise ValueError(
                    f'Requested more audio ({duration_seconds:.2f}s) than available ({available_duration:.2f}s)'
                )

        return samples

    @staticmethod
    def from_dict(data) -> 'AudioSource':
        return AudioSource(**data)


@dataclass
class Recording:
    """
    Recording represents an AudioSource along with some metadata.
    """
    id: str
    sources: List[AudioSource]
    sampling_rate: int
    num_samples: int
    duration_seconds: Seconds

    @property
    def num_channels(self):
        return sum(len(source.channel_ids) for source in self.sources)

    @property
    def channel_ids(self):
        return sorted(cid for source in self.sources for cid in source.channel_ids)

    def load_audio(
            self,
            channels: Optional[Channels] = None,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None,
            root_dir: Optional[Pathlike] = None,
    ) -> np.ndarray:
        if channels is None:
            channels = SetContainingAnything()
        elif isinstance(channels, int):
            channels = frozenset([channels])
        else:
            channels = frozenset(channels)

        samples_per_source = []
        for source in self.sources:
            # Case: source not requested
            if not channels.intersection(source.channel_ids):
                continue
            samples = source.load_audio(
                offset_seconds=offset_seconds,
                duration_seconds=duration_seconds,
                root_dir=root_dir
            )

            # Case: two-channel audio file but only one channel requested
            #       it might not be optimal to load all channels, but IDK if there's anything we can do about it
            channels_to_remove = [
                idx for idx, cid in enumerate(source.channel_ids)
                if cid not in channels
            ]
            if channels_to_remove:
                samples = np.delete(samples, channels_to_remove, axis=0)
            samples_per_source.append(samples)

        # shape: (n_channels, n_samples)
        return np.vstack(samples_per_source)

    @staticmethod
    def from_dict(data: dict) -> 'Recording':
        raw_sources = data.pop('sources')
        return Recording(sources=[AudioSource.from_dict(s) for s in raw_sources], **data)


@dataclass
class RecordingSet:
    """
    RecordingSet represents a dataset of recordings. It does not contain any annotation -
    just the information needed to retrieve a recording (possibly multi-channel, from files
    or from shell commands and pipes) and some metadata for each of them.

    It also supports (de)serialization to/from YAML and takes care of mapping between
    rich Python classes and YAML primitives during conversion.
    """
    recordings: Dict[str, Recording]

    @staticmethod
    def from_recordings(recordings: Iterable[Recording]) -> 'RecordingSet':
        return RecordingSet(recordings={r.id: r for r in recordings})

    @staticmethod
    def from_yaml(path: Pathlike) -> 'RecordingSet':
        raw_recordings = load_yaml(path)
        return RecordingSet.from_recordings(Recording.from_dict(raw_rec) for raw_rec in raw_recordings)

    def to_yaml(self, path: Pathlike):
        data = [asdict(r) for r in self]
        save_to_yaml(data, path)

    def load_audio(
            self,
            recording_id: str,
            channels: Optional[Channels] = None,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None,
            root_dir: Optional[Pathlike] = None,
    ) -> np.ndarray:
        return self.recordings[recording_id].load_audio(
            channels=channels,
            offset_seconds=offset_seconds,
            duration_seconds=duration_seconds,
            root_dir=root_dir
        )

    def num_channels(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_channels

    def sampling_rate(self, recording_id: str) -> int:
        return self.recordings[recording_id].sampling_rate

    def num_samples(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_samples

    def duration_seconds(self, recording_id: str) -> float:
        return self.recordings[recording_id].duration_seconds

    def __getitem__(self, item: str) -> Recording:
        return self.recordings[item]

    def __iter__(self) -> Iterable[Recording]:
        return iter(self.recordings.values())

    def __len__(self) -> int:
        return len(self.recordings)

    def __add__(self, other: 'RecordingSet') -> 'RecordingSet':
        return RecordingSet(recordings={**self.recordings, **other.recordings})
