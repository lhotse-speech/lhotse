from math import sqrt
from typing import List, Optional

import numpy as np
import torch

from lhotse.utils import Decibels, Seconds, compute_num_samples


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

    .. note:: Both single-channel and multi-channel signals are supported as reference
        and added signals. The only requirement is that the when mixing 2 multi-channel
        signals, they must have the same number of channels.

    .. note:: When the AudioMixer contains multi-channel tracks, 2 types of mixed signals
        can be generated:
        - `mixed_audio` mixes each channel independently, and returns a multi-channel signal.
          If there is a mono track, it is added to all the channels.
        - `mixed_mono_audio` mixes all channels together, and returns a single-channel signal.
    """

    def __init__(
        self,
        base_audio: np.ndarray,
        sampling_rate: int,
        reference_energy: Optional[float] = None,
        base_offset: Seconds = 0.0,
    ):
        """
        AudioMixer's constructor.

        :param base_audio: A numpy array with the audio samples for the base signal
            (all the other signals will be mixed to it).
        :param sampling_rate: Sampling rate of the audio.
        :param reference_energy: Optionally pass a reference energy value to compute SNRs against.
            This might be required when ``base_audio`` corresponds to zero-padding.
        :param base_offset: Optionally pass a time offset for the base signal.
        """
        self.tracks = [base_audio]
        self.offsets = [compute_num_samples(base_offset, sampling_rate)]
        self.sampling_rate = sampling_rate
        self.num_channels = base_audio.shape[0]
        self.dtype = self.tracks[0].dtype

        # Keep a pre-computed energy value of the audio that we initialize the Mixer with;
        # it is required to compute gain ratios that satisfy SNR during the mix.
        if reference_energy is None:
            self.reference_energy = audio_energy(base_audio)
        else:
            self.reference_energy = reference_energy

    def _pad_track(
        self, audio: np.ndarray, offset: int, total: Optional[int] = None
    ) -> np.ndarray:
        assert audio.ndim == 2, f"audio.ndim={audio.ndim}"
        if total is None:
            total = audio.shape[1] + offset
        assert (
            audio.shape[1] + offset <= total
        ), f"{audio.shape[1]} + {offset} <= {total}"
        return np.pad(
            audio, pad_width=((0, 0), (offset, total - audio.shape[1] - offset))
        )

    @property
    def num_samples_total(self) -> int:
        longest = 0
        for offset, audio in zip(self.offsets, self.tracks):
            longest = max(longest, offset + audio.shape[1])
        return longest

    @property
    def unmixed_audio(self) -> List[np.ndarray]:
        """
        Return a list of numpy arrays with the shape (C, num_samples), where each track is
        zero padded and scaled adequately to the offsets and SNR used in ``add_to_mix`` call.
        """
        total = self.num_samples_total
        return [
            self._pad_track(track, offset=offset, total=total)
            for offset, track in zip(self.offsets, self.tracks)
        ]

    @property
    def mixed_audio(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (num_channels, num_samples) - a mix of the tracks
        supplied with ``add_to_mix`` calls.
        """
        total = self.num_samples_total
        mixed = np.zeros((self.num_channels, total), dtype=self.dtype)
        for offset, track in zip(self.offsets, self.tracks):
            # Only two cases are possible here: either the track is mono, or it has the same
            # number of channels as the mixer. For the latter case, we don't need to do anything
            # special, as we can just add the track to the mix. For the former case, we need to
            # add the mono track to all channels by repeating it.
            if track.shape[0] == 1 and self.num_channels > 1:
                track = np.tile(track, (self.num_channels, 1))
            mixed[:, offset : offset + track.shape[1]] += track
        return mixed

    @property
    def mixed_mono_audio(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (1, num_samples) - a mix of the tracks
        supplied with ``add_to_mix`` calls.
        """
        total = self.num_samples_total
        mixed = np.zeros((1, total), dtype=self.dtype)
        for offset, track in zip(self.offsets, self.tracks):
            if track.shape[0] > 1:
                # Sum all channels of the track
                track = np.sum(track, axis=0, keepdims=True)
            mixed[:, offset : offset + track.shape[1]] += track
        return mixed

    def add_to_mix(
        self,
        audio: np.ndarray,
        snr: Optional[Decibels] = None,
        offset: Seconds = 0.0,
    ):
        """
        Add audio of a new track into the mix.
        :param audio: An array of audio samples to be mixed in.
        :param snr: Signal-to-noise ratio, assuming `audio` represents noise (positive SNR - lower `audio` energy,
        negative SNR - higher `audio` energy)
        :param offset: How many seconds to shift `audio` in time. For mixing, the signal will be padded before
        the start with low energy values.
        :return:
        """
        if audio.size == 0:
            return  # do nothing for empty arrays

        assert offset >= 0.0, "Negative offset in mixing is not supported."

        num_samples_offset = compute_num_samples(offset, self.sampling_rate)

        # When SNR is requested, find what gain is needed to satisfy the SNR
        gain = 1.0
        if snr is not None and self.reference_energy > 0:
            added_audio_energy = audio_energy(audio)
            if added_audio_energy > 0.0:
                target_energy = self.reference_energy * (10.0 ** (-snr / 10))
                # When mixing time-domain signals, we are working with root-power (field) quantities,
                # whereas the energy ratio applies to power quantities. To compute the gain correctly,
                # we need to take a square root of the energy ratio.
                gain = sqrt(target_energy / added_audio_energy)
        self.tracks.append(gain * audio)
        self.offsets.append(num_samples_offset)
        # We cannot mix 2 multi-channel audios with different number of channels.
        if (
            audio.shape[0] != self.num_channels
            and self.num_channels != 1
            and audio.shape[0] != 1
        ):
            raise ValueError(
                f"Cannot mix audios with {audio.shape[0]} and {self.num_channels} channels."
            )
        self.num_channels = max(self.num_channels, audio.shape[0])


def audio_energy(audio: np.ndarray) -> float:
    return float(np.average(audio**2))


class VideoMixer:
    """
    Simple video "mixing" class that actually does not mix anything but supports concatenation.
    """

    def __init__(
        self,
        base_video: torch.Tensor,
        fps: float,
        base_offset: Seconds = 0.0,
    ):
        from intervaltree import IntervalTree

        self.tracks = [base_video]
        self.offsets = [compute_num_samples(base_offset, fps)]
        self.fps = fps
        self.dtype = self.tracks[0].dtype
        self.tree = IntervalTree()
        self.tree.addi(self.offsets[0], self.offsets[0] + base_video.shape[0])

    def _pad_track(
        self, video: torch.Tensor, offset: int, total: Optional[int] = None
    ) -> torch.Tensor:
        if total is None:
            total = video.shape[0] + offset
        assert (
            video.shape[0] + offset <= total
        ), f"{video.shape[0]} + {offset} <= {total}"
        return torch.nn.functional.pad(
            video,
            (0, 0, 0, 0, 0, 0, offset, total - video.shape[0] - offset),
            mode="constant",
            value=0,
        )
        # return torch.from_numpy(
        #     np.pad(
        #         video.numpy(),
        #         pad_width=(
        #             (offset, total - video.shape[0] - offset),
        #             (0, 0),
        #             (0, 0),
        #             (0, 0),
        #         ),
        #     )
        # )

    @property
    def num_frames_total(self) -> int:
        longest = 0
        for offset, video in zip(self.offsets, self.tracks):
            longest = max(longest, offset + video.shape[0])
        return longest

    @property
    def unmixed_video(self) -> List[torch.Tensor]:
        """
        Return a list of numpy arrays with the shape (C, num_samples), where each track is
        zero padded and scaled adequately to the offsets and SNR used in ``add_to_mix`` call.
        """
        total = self.num_frames_total
        return [
            self._pad_track(track, offset=offset, total=total)
            for offset, track in zip(self.offsets, self.tracks)
        ]

    @property
    def mixed_video(self) -> torch.Tensor:
        """
        Return a numpy ndarray with the shape (num_channels, num_samples) - a mix of the tracks
        supplied with ``add_to_mix`` calls.
        """
        total = self.num_frames_total
        mixed = self.tracks[0].new_zeros((total,) + self.tracks[0].shape[1:])
        for offset, track in zip(self.offsets, self.tracks):
            mixed[offset : offset + track.shape[0]] = track
        return mixed

    def add_to_mix(
        self,
        video: torch.Tensor,
        offset: Seconds = 0.0,
    ):
        if video.size == 0:
            return  # do nothing for empty arrays

        assert offset >= 0.0, "Negative offset in mixing is not supported."
        frame_offset = compute_num_samples(offset, self.fps)

        from intervaltree import Interval

        interval = Interval(frame_offset, frame_offset + video.shape[0])
        assert not self.tree.overlaps(interval), (
            f"Cannot add an overlapping video. Got {interval} while we "
            f"have the following intervals: {self.tree.all_intervals()}"
        )

        self.tracks.append(video)
        self.offsets.append(frame_offset)
        self.tree.add(interval)
