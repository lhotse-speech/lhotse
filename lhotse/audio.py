import warnings
from dataclasses import dataclass
from io import BytesIO
from math import sqrt
from pathlib import Path
from subprocess import PIPE, run
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from lhotse.augmentation import AudioTransform, Speed
from lhotse.utils import (Decibels, JsonMixin, Pathlike, Seconds, SetContainingAnything, YamlMixin, asdict_nonull,
                          compute_num_samples,
                          fastcopy,
                          index_by_id_and_check, perturb_num_samples, split_sequence)

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
    channels: List[int]
    source: str

    def load_audio(
            self,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None,
    ) -> np.ndarray:
        """
        Load the AudioSource (both files and commands) with librosa,
        accounting for many audio formats and multi-channel inputs.
        Returns numpy array with shapes: (n_samples) for single-channel,
        (n_channels, n_samples) for multi-channel.
        """
        assert self.type in ('file', 'command')

        source = self.source
        if self.type == 'command':
            if offset_seconds != 0.0 or duration_seconds is not None:
                # TODO(pzelasko): How should we support chunking for commands?
                #                 We risk being very inefficient when reading many chunks from the same file
                #                 without some caching scheme, because we'll be re-running commands.
                raise ValueError("Reading audio chunks from command AudioSource type is currently not supported.")
            source = BytesIO(run(self.source, shell=True, stdout=PIPE).stdout)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            samples, sampling_rate = read_audio(source, offset=offset_seconds, duration=duration_seconds)

        # explicit sanity check for duration as librosa does not complain here
        if duration_seconds is not None:
            num_samples = samples.shape[0] if len(samples.shape) == 1 else samples.shape[1]
            available_duration = num_samples / sampling_rate
            if available_duration < duration_seconds - 1e-3:  # set the allowance as 1ms to avoid float error
                raise ValueError(
                    f'Requested more audio ({duration_seconds}s) than available ({available_duration}s)'
                )

        return samples.astype(np.float32)

    def with_path_prefix(self, path: Pathlike) -> 'AudioSource':
        if self.type != 'file':
            return self
        return fastcopy(self, source=str(Path(path) / self.source))

    @staticmethod
    def from_dict(data) -> 'AudioSource':
        return AudioSource(**data)


def read_audio(path: Pathlike, offset: Seconds, duration: Seconds) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    with sf.SoundFile(path) as sf_desc:
        sampling_rate = sf_desc.samplerate
        if offset:
            # Seek to the start of the target read
            sf_desc.seek(compute_num_samples(offset, sampling_rate))
        if duration is not None:
            frame_duration = compute_num_samples(duration, sampling_rate)
        else:
            frame_duration = -1
        # Load the target number of frames, and transpose to match librosa form
        return sf_desc.read(frames=frame_duration, dtype=np.float32, always_2d=False).T, sampling_rate


@dataclass
class Recording:
    """
    Recording represents an AudioSource along with some metadata.
    """
    id: str
    sources: List[AudioSource]
    sampling_rate: int
    num_samples: int
    duration: Seconds
    transforms: Optional[List[Dict]] = None

    @staticmethod
    def from_sphere(
            sph_path: Pathlike,
            recording_id: Optional[str] = None,
            relative_path_depth: Optional[int] = None
    ) -> 'Recording':
        """
        Read a SPHERE file's header and create the corresponding ``Recording``.

        :param sph_path: Path to the sphere (.sph) file.
        :param recording_id: recording id, when not specified ream the filename's stem ("x.wav" -> "x").
        :param relative_path_depth: optional int specifying how many last parts of the file path
            should be retained in the ``AudioSource``. By default writes the path as is.
        :return: a new ``Recording`` instance pointing to the sphere file.
        """
        from sphfile import SPHFile
        sph_path = Path(sph_path)
        sphf = SPHFile(sph_path)
        return Recording(
            id=recording_id if recording_id is not None else sph_path.stem,
            sampling_rate=sphf.format['sample_rate'],
            num_samples=sphf.format['sample_count'],
            duration=sphf.format['sample_count'] / sphf.format['sample_rate'],
            sources=[
                AudioSource(
                    type='file',
                    channels=list(range(sphf.format['channel_count'])),
                    source=(
                        '/'.join(sph_path.parts[-relative_path_depth:])
                        if relative_path_depth is not None and relative_path_depth > 0
                        else str(sph_path)
                    )
                )
            ]
        )

    @staticmethod
    def from_wav(path: Pathlike, recording_id: Optional[str] = None) -> 'Recording':
        """
        Read a WAVE file's header and create the corresponding ``Recording``.

        :param path: Path to the WAVE (.wav) file.
        :param recording_id: recording id, when not specified ream the filename's stem ("x.wav" -> "x").
        :return: a new ``Recording`` instance pointing to the audio file.
        """
        warnings.warn(
            'Recording.from_wav() is deprecated and will be removed in Lhotse v0.5; '
            'please use Recording.from_file() instead.',
            category=DeprecationWarning
        )
        return Recording.from_file(path=path, recording_id=recording_id)

    @staticmethod
    def from_file(path: Pathlike, recording_id: Optional[str] = None):
        """
        Read an audio file's header and create the corresponding ``Recording``.
        Suitable to use when each physical file represents a separate recording session.

        If a recording session consists of multiple files (e.g. one per channel),
        it is advisable to create the ``Recording`` object manually, with each
        file represented as a separate ``AudioSource`` object.

        :param path: Path to an audio file supported by libsoundfile (pysoundfile).
        :param recording_id: recording id, when not specified ream the filename's stem ("x.wav" -> "x").
        :return: a new ``Recording`` instance pointing to the audio file.
        """
        import soundfile
        info = soundfile.info(path)
        return Recording(
            id=recording_id if recording_id is not None else Path(path).stem,
            sampling_rate=info.samplerate,
            num_samples=info.frames,
            duration=info.duration,
            sources=[
                AudioSource(
                    type='file',
                    channels=list(range(info.channels)),
                    source=str(path)
                )
            ]
        )

    @property
    def num_channels(self):
        return sum(len(source.channels) for source in self.sources)

    @property
    def channel_ids(self):
        return sorted(cid for source in self.sources for cid in source.channels)

    def load_audio(
            self,
            channels: Optional[Channels] = None,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None,
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
            if not channels.intersection(source.channels):
                continue
            offset, duration = self._determine_offset_and_duration(offset_seconds, duration_seconds)
            samples = source.load_audio(
                offset_seconds=offset,
                duration_seconds=duration,
            )

            # Case: two-channel audio file but only one channel requested
            #       it might not be optimal to load all channels, but IDK if there's anything we can do about it
            channels_to_remove = [
                idx for idx, cid in enumerate(source.channels)
                if cid not in channels
            ]
            if channels_to_remove:
                samples = np.delete(samples, channels_to_remove, axis=0)
            samples_per_source.append(samples)

        # shape: (n_channels, n_samples)
        audio = np.vstack(samples_per_source)

        # We'll apply the transforms now (if any).
        for params in self.transforms or []:
            transform = AudioTransform.from_dict(params)
            audio = transform(audio, self.sampling_rate)

        return audio

    def _determine_offset_and_duration(self, offset: Seconds, duration: Seconds) -> Tuple[Seconds, Seconds]:
        """
        This internal method helps estimate the original offset and duration for a recording
        before speed perturbation was applied.
        We need this estimate to know how much audio to actually load from disk during the
        call to ``load_audio()``.
        """
        if self.transforms is None or all(t['name'] != 'Speed' for t in self.transforms):
            return offset, duration
        start_sample = offset * self.sampling_rate
        num_samples = duration * self.sampling_rate if duration is not None else None
        for tfr in reversed(self.transforms):
            if tfr['name'] != 'Speed':
                continue
            factor = tfr['kwargs']['factor']
            start_sample = perturb_num_samples(start_sample, 1 / factor)
            num_samples = perturb_num_samples(num_samples, 1 / factor) if num_samples is not None else None
        return start_sample / self.sampling_rate, num_samples / self.sampling_rate if num_samples is not None else None

    def with_path_prefix(self, path: Pathlike) -> 'Recording':
        return fastcopy(self, sources=[s.with_path_prefix(path) for s in self.sources])

    def perturb_speed(self, factor: float, affix_id: bool = True) -> 'Recording':
        """
        Return a new ``Recording`` that will lazily perturb the speed while loading audio.
        The ``num_samples`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``Recording.id`` field
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``Recording``.
        """
        transforms = self.transforms if self.transforms is not None else []
        transforms.append(Speed(factor=factor).to_dict())
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f'{self.id}_sp{factor}' if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration,
            transforms=transforms
        )

    @staticmethod
    def from_dict(data: dict) -> 'Recording':
        raw_sources = data.pop('sources')
        return Recording(sources=[AudioSource.from_dict(s) for s in raw_sources], **data)


@dataclass
class RecordingSet(JsonMixin, YamlMixin, Sequence[Recording]):
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
        return RecordingSet(recordings=index_by_id_and_check(recordings))

    @staticmethod
    def from_dicts(data: Iterable[dict]) -> 'RecordingSet':
        return RecordingSet.from_recordings(Recording.from_dict(raw_rec) for raw_rec in data)

    def to_dicts(self) -> List[dict]:
        return [asdict_nonull(r) for r in self]

    def filter(self, predicate: Callable[[Recording], bool]) -> 'RecordingSet':
        """
        Return a new RecordingSet with the Recordings that satisfy the `predicate`.

        :param predicate: a function that takes a recording as an argument and returns bool.
        :return: a filtered RecordingSet.
        """
        return RecordingSet.from_recordings(rec for rec in self if predicate(rec))

    def split(self, num_splits: int, shuffle: bool = False) -> List['RecordingSet']:
        """
        Split the ``RecordingSet`` into ``num_splits`` pieces of equal size.

        :param num_splits: Requested number of splits.
        :param shuffle: Optionally shuffle the recordings order first.
        :return: A list of ``RecordingSet`` pieces.
        """
        return [
            RecordingSet.from_recordings(subset) for subset in
            split_sequence(self, num_splits=num_splits, shuffle=shuffle)
        ]

    def load_audio(
            self,
            recording_id: str,
            channels: Optional[Channels] = None,
            offset_seconds: float = 0.0,
            duration_seconds: Optional[float] = None,
    ) -> np.ndarray:
        return self.recordings[recording_id].load_audio(
            channels=channels,
            offset_seconds=offset_seconds,
            duration_seconds=duration_seconds
        )

    def with_path_prefix(self, path: Pathlike) -> 'RecordingSet':
        return RecordingSet.from_recordings(r.with_path_prefix(path) for r in self)

    def num_channels(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_channels

    def sampling_rate(self, recording_id: str) -> int:
        return self.recordings[recording_id].sampling_rate

    def num_samples(self, recording_id: str) -> int:
        return self.recordings[recording_id].num_samples

    def duration(self, recording_id: str) -> Seconds:
        return self.recordings[recording_id].duration

    def __repr__(self) -> str:
        return f'RecordingSet(len={len(self)})'

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
    Utility class to mix multiple waveforms into a single one.
    It should be instantiated separately for each mixing session (i.e. each ``MixedCut``
    will create a separate ``AudioMixer`` to mix its tracks).
    It is initialized with a numpy array of audio samples (typically float32 in [-1, 1] range)
    that represents the "reference" signal for the mix.
    Other signals can be mixed to it with different time offsets and SNRs using the
    ``add_to_mix`` method.
    The time offset is relative to the start of the reference signal
    (only positive values are supported).
    The SNR is relative to the energy of the signal used to initialize the ``AudioMixer``.
    """

    def __init__(self, base_audio: np.ndarray, sampling_rate: int):
        """
        :param base_audio: A numpy array with the audio samples for the base signal
            (all the other signals will be mixed to it).
        :param sampling_rate: Sampling rate of the audio.
        """
        self.tracks = [base_audio]
        self.sampling_rate = sampling_rate
        self.reference_energy = audio_energy(base_audio)
        self.dtype = self.tracks[0].dtype

    @property
    def unmixed_audio(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (num_tracks, num_samples), where each track is
        zero padded and scaled adequately to the offsets and SNR used in ``add_to_mix`` call.
        """
        return np.vstack(self.tracks)

    @property
    def mixed_audio(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (1, num_samples) - a mono mix of the tracks
        supplied with ``add_to_mix`` calls.
        """
        return np.sum(self.unmixed_audio, axis=0, keepdims=True)

    def add_to_mix(
            self,
            audio: np.ndarray,
            snr: Optional[Decibels] = None,
            offset: Seconds = 0.0,
    ):
        """
        Add audio (only support mono-channel) of a new track into the mix.
        :param audio: An array of audio samples to be mixed in.
        :param snr: Signal-to-noise ratio, assuming `audio` represents noise (positive SNR - lower `audio` energy,
        negative SNR - higher `audio` energy)
        :param offset: How many seconds to shift `audio` in time. For mixing, the signal will be padded before
        the start with low energy values.
        :return:
        """
        assert audio.shape[0] == 1  # TODO: support multi-channels
        assert offset >= 0.0, "Negative offset in mixing is not supported."

        reference_audio = self.tracks[0]
        num_samples_offset = round(offset * self.sampling_rate)
        current_num_samples = reference_audio.shape[1]

        audio_to_add = audio

        # When there is an offset, we need to pad before the start of the audio we're adding.
        if offset > 0:
            audio_to_add = np.hstack([
                np.zeros((1, num_samples_offset), self.dtype),
                audio_to_add
            ])

        incoming_num_samples = audio_to_add.shape[1]
        mix_num_samples = max(current_num_samples, incoming_num_samples)

        # When the existing samples are less than what we anticipate after the mix,
        # we need to pad after the end of the existing audio mixed so far.
        # Since we're keeping every track as a separate entry in the ``self.tracks`` list,
        # we need to pad each of them so that their shape matches when performing the final mix.
        if current_num_samples < mix_num_samples:
            for idx in range(len(self.tracks)):
                padded_audio = np.hstack([
                    self.tracks[idx],
                    np.zeros((1, mix_num_samples - current_num_samples), self.dtype)
                ])
                self.tracks[idx] = padded_audio

        # When the audio we're mixing in are shorter that the anticipated mix length,
        # we need to pad after their end.
        # Note: we're doing that non-efficiently, as it we potentially re-allocate numpy arrays twice,
        # during this padding and the  offset padding before. If that's a bottleneck, we'll optimize.
        if incoming_num_samples < mix_num_samples:
            audio_to_add = np.hstack([
                audio_to_add,
                np.zeros((1, mix_num_samples - incoming_num_samples), self.dtype)
            ])

        # When SNR is requested, find what gain is needed to satisfy the SNR
        gain = 1.0
        if snr is not None:
            added_audio_energy = audio_energy(audio)
            target_energy = self.reference_energy * (10.0 ** (-snr / 10))
            # When mixing time-domain singals, we are working with root-power (field) quantities,
            # whereas the energy ratio applies to power quantities. To compute the gain correctly,
            # we need to take a square root of the energy ratio.
            gain = sqrt(target_energy / added_audio_energy)

        # self.mixed_audio = reference_audio + gain * audio_to_add
        self.tracks.append(gain * audio_to_add)


def audio_energy(audio: np.ndarray) -> float:
    return float(np.average(audio ** 2))