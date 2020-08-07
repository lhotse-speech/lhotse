import os
import warnings
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from subprocess import PIPE, run
from typing import Callable, Dict, Iterable, List, Optional, Union

# Workaround for SoundFile (librosa dep) raising exception when a native library, libsndfile1, is not installed.
# Read-the-docs does not allow to modify the Docker containers used to build documentation...
if not os.environ.get('READTHEDOCS', False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import librosa
import numpy as np

from lhotse.utils import Decibels, Pathlike, Seconds, SetContainingAnything, load_yaml, save_to_yaml

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
                    f'Requested more audio ({duration_seconds}s) than available ({available_duration}s)'
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

    def filter(self, predicate: Callable[[Recording], bool]) -> 'RecordingSet':
        """
        Return a new RecordingSet with the Recordings that satisfy the `predicate`.

        :param predicate: a function that takes a recording as an argument and returns bool.
        :return: a filtered RecordingSet.
        """
        return RecordingSet.from_recordings(rec for rec in self if predicate(rec))

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

    def __getitem__(self, recording_id_or_index: Union[int, str]) -> Recording:
        if isinstance(recording_id_or_index, str):
            return self.recordings[recording_id_or_index]
        # ~100x faster than list(dict.values())[index] for 100k elements
        return next(val for idx, val in enumerate(self.recordings.values()) if idx == recording_id_or_index)

    def __iter__(self) -> Iterable[Recording]:
        return iter(self.recordings.values())

    def __len__(self) -> int:
        return len(self.recordings)

    def __add__(self, other: 'RecordingSet') -> 'RecordingSet':
        return RecordingSet(recordings={**self.recordings, **other.recordings})


class AudioMixer:
    """
    Utility class to mix multiple raw audio into a single one.
    It pads the signals with zero samples for differing lengths and offsets.
    """

    def __init__(self, base_audio: np.ndarray):
        """
        :param base_audio: The raw audio used to initialize the AudioMixer are a point of reference
            in terms of offset for all audios mixed into them.
        """
        # The mixing output will be available in self.mixed_audio
        self.mixed_audio = base_audio
        self.reference_energy = audio_energy(base_audio)

    def add_to_mix(
            self,
            audio: np.ndarray,
            snr: Optional[Decibels] = None,
            offset: Seconds = 0.0,
            sampling_rate: int = 16000,
    ):
        """
        Add audio (only support mono-channel) of a new track into the mix.
        :param audio: An array of audio samples to be mixed in.
        :param snr: Signal-to-noise ratio, assuming `audio` represents noise (positive SNR - lower `audio` energy,
        negative SNR - higher `audio` energy)
        :param offset: How many seconds to shift `audio` in time. For mixing, the signal will be padded before
        the start with low energy values.
        :param sampling_rate: Sampling rate of the audio.
        :return:
        """
        assert audio.shape[0] == 1  # TODO: support multi-channels
        assert offset >= 0.0, "Negative offset in mixing is not supported."

        num_samples_offset = round(offset * sampling_rate)
        current_num_samples = self.mixed_audio.shape[1]

        existing_audio = self.mixed_audio
        audio_to_add = audio

        # When there is an offset, we need to pad before the start of the audio we're adding.
        if offset > 0:
            audio_to_add = np.hstack([
                np.zeros((1, num_samples_offset)),
                audio_to_add
            ])

        incoming_num_samples = audio_to_add.shape[1]
        mix_num_samples = max(current_num_samples, incoming_num_samples)

        # When the existing samples are less than what we anticipate after the mix,
        # we need to pad after the end of the existing audio mixed so far.
        if current_num_samples < mix_num_samples:
            existing_audio = np.hstack([
                self.mixed_audio,
                np.zeros((1, mix_num_samples - current_num_samples))
            ])

        # When the audio we're mixing in are shorter that the anticipated mix length,
        # we need to pad after their end.
        # Note: we're doing that non-efficiently, as it we potentially re-allocate numpy arrays twice,
        # during this padding and the  offset padding before. If that's a bottleneck, we'll optimize.
        if incoming_num_samples < mix_num_samples:
            audio_to_add = np.hstack([
                audio_to_add,
                np.zeros((1, mix_num_samples - incoming_num_samples))
            ])

        # When SNR is requested, find what gain is needed to satisfy the SNR
        gain = 1.0
        if snr is not None:
            added_audio_energy = audio_energy(audio)
            target_energy = self.reference_energy * (10.0 ** (-snr / 10))
            gain = target_energy / added_audio_energy

        self.mixed_audio = existing_audio + gain * audio_to_add


def audio_energy(audio: np.ndarray) -> float:
    return float(np.average(audio ** 2))
