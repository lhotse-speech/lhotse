"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
           2021 Johns Hopkins University  (Author: Piotr Żelasko)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

This whole module is authored and contributed by Jesus Villalba,
with minor changes by Piotr Żelasko to make it more consistent with Lhotse.

It contains a PyTorch implementation of feature extractors that is very close to Kaldi's
-- notably, it differs in that the preemphasis and DC offset removal are applied in the
time, rather than frequency domain. This should not significantly affect any results, as
confirmed by Jesus.

This implementation works well with autograd and batching, and can be used neural network
layers.

Update January 2022:
These modules now expose a new API function called "online_inference" that
may be used to compute the features when the audio is streaming.
The implementation is stateless, and passes the waveform remainders
back to the user to feed them to the modules once new data becomes available.
The implementation is compatible with JIT scripting via TorchScript.
"""
import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

try:
    from torch.fft import rfft as torch_rfft

    def _rfft(x: torch.Tensor) -> torch.Tensor:
        return torch_rfft(x, dim=-1)

    def _pow_spectrogram(x: torch.Tensor) -> torch.Tensor:
        return x.abs() ** 2

    def _spectrogram(x: torch.Tensor) -> torch.Tensor:
        return x.abs()

except ImportError:

    def _rfft(x: torch.Tensor) -> torch.Tensor:
        return torch.rfft(x, 1, normalized=False, onesided=True)

    def _pow_spectrogram(x: torch.Tensor) -> torch.Tensor:
        return x.pow(2).sum(-1)

    def _spectrogram(x: torch.Tensor) -> torch.Tensor:
        return x.pow(2).sum(-1).sqrt()


from lhotse.utils import EPSILON, Seconds


class Wav2Win(nn.Module):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and partition them into overlapping frames (of audio samples).
    Note: no feature extraction happens in here, the output is still a time-domain signal.

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2Win()
        >>> t(x).shape
        torch.Size([1, 100, 400])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, window_length)``.
    When ``return_log_energy==True``, returns a tuple where the second element
    is a log-energy tensor of shape ``(batch_size, num_frames)``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: Seconds = 0.025,
        frame_shift: Seconds = 0.01,
        pad_length: Optional[int] = None,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = EPSILON,
        raw_energy: bool = True,
        return_log_energy: bool = False,
    ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.remove_dc_offset = remove_dc_offset
        self.preemph_coeff = preemph_coeff
        self.window_type = window_type
        self.dither = dither
        # torchscript expects it to be a tensor
        self.snip_edges = snip_edges
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.return_log_energy = return_log_energy
        if snip_edges:
            warnings.warn(
                "Setting snip_edges=True is generally incompatible with Lhotse -- "
                "you might experience mismatched duration/num_frames errors."
            )

        N = int(math.floor(frame_length * sampling_rate))
        self._length = N
        self._shift = int(math.floor(frame_shift * sampling_rate))

        self._window = nn.Parameter(
            create_frame_window(N, window_type=window_type), requires_grad=False
        )
        self.pad_length = N if pad_length is None else pad_length
        assert (
            self.pad_length >= N
        ), f"pad_length (or fft_length) = {pad_length} cannot be smaller than N = {N}"

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = (
            "{}(sampling_rate={}, frame_length={}, frame_shift={}, pad_length={}, "
            "remove_dc_offset={}, preemph_coeff={}, window_type={} "
            "dither={}, snip_edges={}, energy_floor={}, raw_energy={}, return_log_energy={})"
        ).format(
            self.__class__.__name__,
            self.sampling_rate,
            self.frame_length,
            self.frame_shift,
            self.pad_length,
            self.remove_dc_offset,
            self.preemph_coeff,
            self.window_type,
            self.dither,
            self.snip_edges,
            self.energy_floor,
            self.raw_energy,
            self.return_log_energy,
        )
        return s

    def _forward_strided(
        self, x_strided: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # remove offset
        if self.remove_dc_offset:
            mu = torch.mean(x_strided, dim=2, keepdim=True)
            x_strided = x_strided - mu

        # Compute the log energy of each frame
        log_energy: Optional[torch.Tensor] = None
        if self.return_log_energy and self.raw_energy:
            log_energy = _get_log_energy(x_strided, self.energy_floor)  # size (m)

        # preemphasis
        if self.preemph_coeff != 0.0:
            x_offset = torch.nn.functional.pad(x_strided, (1, 0), mode="replicate")
            x_strided = x_strided - self.preemph_coeff * x_offset[:, :, :-1]

        # Apply window_function to each frame
        x_strided = x_strided * self._window

        # Pad columns with zero until we reach size (batch, num_frames, pad_length)
        if self.pad_length != self._length:
            pad = self.pad_length - self._length
            x_strided = torch.nn.functional.pad(
                # torchscript expects pad to be list of int
                x_strided.unsqueeze(1),
                [0, pad],
                mode="constant",
                value=0.0,
            ).squeeze(1)

        if self.return_log_energy and not self.raw_energy:
            # This energy is computed after preemphasis, window, etc.
            log_energy = _get_log_energy(x_strided, self.energy_floor)  # size (m)

        return x_strided, log_energy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Add dither
        if self.dither != 0.0:
            n = torch.randn(x.shape, device=x.device)
            x = x + self.dither * n

        x_strided = _get_strided_batch(x, self._length, self._shift, self.snip_edges)

        return self._forward_strided(x_strided)

    @torch.jit.export
    def online_inference(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        The same as the ``forward()`` method, except it accepts an extra argument with the
        remainder waveform from the previous call of ``online_inference()``, and returns
        a tuple of ``((frames, log_energy), remainder)``.
        """
        assert (
            not self.snip_edges
        ), "Unsupported operation: snip_edges == True is not supported for online inference."

        # Add dither
        if self.dither != 0.0:
            n = torch.randn(x.shape, device=x.device)
            x = x + self.dither * n

        x_strided, remainder = _get_strided_batch_streaming(
            x,
            window_length=self._length,
            window_shift=self._shift,
            prev_remainder=context,
        )

        x_strided, log_energy = self._forward_strided(x_strided)

        return (x_strided, log_energy), remainder


class Wav2FFT(nn.Module):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The output is a complex-valued tensor.

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2FFT()
        >>> t(x).shape
        torch.Size([1, 100, 257])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``
    with dtype ``torch.complex64``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: Seconds = 0.025,
        frame_shift: Seconds = 0.01,
        round_to_power_of_two: bool = True,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = EPSILON,
        raw_energy: bool = True,
        use_energy: bool = True,
    ) -> None:
        super().__init__()
        self.use_energy = use_energy
        N = int(math.floor(frame_length * sampling_rate))
        self.fft_length = next_power_of_2(N) if round_to_power_of_two else N
        self.wav2win = Wav2Win(
            sampling_rate,
            frame_length,
            frame_shift,
            pad_length=self.fft_length,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            return_log_energy=use_energy,
        )

    @property
    def sampling_rate(self) -> int:
        return self.wav2win.sampling_rate

    @property
    def frame_length(self) -> Seconds:
        return self.wav2win.frame_length

    @property
    def frame_shift(self) -> Seconds:
        return self.wav2win.frame_shift

    @property
    def remove_dc_offset(self) -> bool:
        return self.wav2win.remove_dc_offset

    @property
    def preemph_coeff(self) -> float:
        return self.wav2win.preemph_coeff

    @property
    def window_type(self) -> str:
        return self.wav2win.window_type

    @property
    def dither(self) -> float:
        return self.wav2win.dither

    def _forward_strided(
        self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Note: subclasses of this module can override ``_forward_strided()`` and get a working
        # implementation of ``forward()`` and ``online_inference()`` for free.
        X = _rfft(x_strided)

        # log_e is not None is needed by torchscript
        if self.use_energy and log_e is not None:
            X[:, :, 0] = log_e

        return X

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_strided, log_e = self.wav2win(x)
        return self._forward_strided(x_strided=x_strided, log_e=log_e)

    @torch.jit.export
    def online_inference(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x_strided, log_e), remainder = self.wav2win.online_inference(
            x, context=context
        )
        return self._forward_strided(x_strided=x_strided, log_e=log_e), remainder


class Wav2Spec(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The STFT is transformed either to a magnitude spectrum (``use_fft_mag=True``)
    or a power spectrum (``use_fft_mag=False``).

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2Spec()
        >>> t(x).shape
        torch.Size([1, 100, 257])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: Seconds = 0.025,
        frame_shift: Seconds = 0.01,
        round_to_power_of_two: bool = True,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = EPSILON,
        raw_energy: bool = True,
        use_energy: bool = True,
        use_fft_mag: bool = False,
    ) -> None:
        super().__init__(
            sampling_rate,
            frame_length,
            frame_shift,
            round_to_power_of_two=round_to_power_of_two,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            use_energy=use_energy,
        )
        self.use_fft_mag = use_fft_mag
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

    def _forward_strided(
        self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]
    ) -> torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)

        # log_e is not None is needed by torchscript
        if self.use_energy and log_e is not None:
            pow_spec[:, :, 0] = log_e

        return pow_spec


class Wav2LogSpec(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The STFT is transformed either to a log-magnitude spectrum (``use_fft_mag=True``)
    or a log-power spectrum (``use_fft_mag=False``).

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2LogSpec()
        >>> t(x).shape
        torch.Size([1, 100, 257])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: Seconds = 0.025,
        frame_shift: Seconds = 0.01,
        round_to_power_of_two: bool = True,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = EPSILON,
        raw_energy: bool = True,
        use_energy: bool = True,
        use_fft_mag: bool = False,
    ) -> None:
        super().__init__(
            sampling_rate,
            frame_length,
            frame_shift,
            round_to_power_of_two=round_to_power_of_two,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            use_energy=use_energy,
        )
        self.use_fft_mag = use_fft_mag
        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

    def _forward_strided(
        self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]
    ) -> torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)

        pow_spec = (pow_spec + 1e-15).log()

        # log_e is not None is needed by torchscript
        if self.use_energy and log_e is not None:
            pow_spec[:, :, 0] = log_e

        return pow_spec


class Wav2LogFilterBank(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their log-Mel filter bank energies (also known as "fbank").

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2LogFilterBank()
        >>> t(x).shape
        torch.Size([1, 100, 80])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_filters)``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: Seconds = 0.025,
        frame_shift: Seconds = 0.01,
        round_to_power_of_two: bool = True,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = EPSILON,
        raw_energy: bool = True,
        use_energy: bool = False,
        use_fft_mag: bool = False,
        low_freq: float = 20.0,
        high_freq: float = -400.0,
        num_filters: int = 80,
        norm_filters: bool = False,
        torchaudio_compatible_mel_scale: bool = True,
    ):

        super().__init__(
            sampling_rate,
            frame_length,
            frame_shift,
            round_to_power_of_two=round_to_power_of_two,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            use_energy=use_energy,
        )

        self.use_fft_mag = use_fft_mag
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self._eps = nn.Parameter(
            torch.tensor(torch.finfo(torch.float).eps), requires_grad=False
        )

        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

        if torchaudio_compatible_mel_scale:
            from torchaudio.compliance.kaldi import get_mel_banks

            # see torchaudio.compliance.kaldi.fbank, lines #581-587 for the original usage
            fb, _ = get_mel_banks(
                num_bins=num_filters,
                window_length_padded=self.fft_length,
                sample_freq=sampling_rate,
                low_freq=low_freq,
                high_freq=high_freq,
                # VTLN args are hardcoded to torchaudio default values;
                # they are not used anyway with wapr_factor == 1.0
                vtln_warp_factor=1.0,
                vtln_low=100.0,
                vtln_high=-500.0,
            )
            fb = torch.nn.functional.pad(fb, (0, 1), mode="constant", value=0).T
        else:
            fb = create_mel_scale(
                num_filters=num_filters,
                fft_length=self.fft_length,
                sampling_rate=sampling_rate,
                low_freq=low_freq,
                high_freq=high_freq,
                norm_filters=norm_filters,
            )
        self._fb = nn.Parameter(fb, requires_grad=False)

    def _forward_strided(
        self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]
    ) -> torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)

        pow_spec = torch.matmul(pow_spec, self._fb)
        pow_spec = torch.max(pow_spec, self._eps).log()

        # log_e is not None is needed by torchscript
        if self.use_energy and log_e is not None:
            pow_spec = torch.cat((log_e.unsqueeze(-1), pow_spec), dim=-1)

        return pow_spec


class Wav2MFCC(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Mel-Frequency Cepstral Coefficients (MFCC).

    Example::

        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2MFCC()
        >>> t(x).shape
        torch.Size([1, 100, 13])

    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_ceps)``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: Seconds = 0.025,
        frame_shift: Seconds = 0.01,
        round_to_power_of_two: bool = True,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = EPSILON,
        raw_energy: bool = True,
        use_energy: bool = False,
        use_fft_mag: bool = False,
        low_freq: float = 20.0,
        high_freq: float = -400.0,
        num_filters: int = 23,
        norm_filters: bool = False,
        num_ceps: int = 13,
        cepstral_lifter: int = 22,
        torchaudio_compatible_mel_scale: bool = True,
    ):

        super().__init__(
            sampling_rate,
            frame_length,
            frame_shift,
            round_to_power_of_two=round_to_power_of_two,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            use_energy=use_energy,
        )

        self.use_fft_mag = use_fft_mag
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self.num_ceps = num_ceps
        self.cepstral_lifter = cepstral_lifter
        self._eps = nn.Parameter(
            torch.tensor(torch.finfo(torch.float).eps), requires_grad=False
        )

        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

        if torchaudio_compatible_mel_scale:
            from torchaudio.compliance.kaldi import get_mel_banks

            # see torchaudio.compliance.kaldi.fbank, lines #581-587 for the original usage
            fb, _ = get_mel_banks(
                num_bins=num_filters,
                window_length_padded=self.fft_length,
                sample_freq=sampling_rate,
                low_freq=low_freq,
                high_freq=high_freq,
                # VTLN args are hardcoded to torchaudio default values;
                # they are not used anyway with wapr_factor == 1.0
                vtln_warp_factor=1.0,
                vtln_low=100.0,
                vtln_high=-500.0,
            )
            fb = torch.nn.functional.pad(fb, (0, 1), mode="constant", value=0).T
        else:
            fb = create_mel_scale(
                num_filters=num_filters,
                fft_length=self.fft_length,
                sampling_rate=sampling_rate,
                low_freq=low_freq,
                high_freq=high_freq,
                norm_filters=norm_filters,
            )
        self._fb = nn.Parameter(fb, requires_grad=False)

        self._dct = nn.Parameter(
            self.make_dct_matrix(self.num_ceps, self.num_filters), requires_grad=False
        )
        self._lifter = nn.Parameter(
            self.make_lifter(self.num_ceps, self.cepstral_lifter), requires_grad=False
        )

    @staticmethod
    def make_lifter(N, Q):
        """Makes the liftering function

        Args:
          N: Number of cepstral coefficients.
          Q: Liftering parameter
        Returns:
          Liftering vector.
        """
        if Q == 0:
            return 1
        return 1 + 0.5 * Q * torch.sin(
            math.pi * torch.arange(N, dtype=torch.get_default_dtype()) / Q
        )

    @staticmethod
    def make_dct_matrix(num_ceps, num_filters):
        n = torch.arange(float(num_filters)).unsqueeze(1)
        k = torch.arange(float(num_ceps))
        dct = torch.cos(
            math.pi / float(num_filters) * (n + 0.5) * k
        )  # size (n_mfcc, n_mels)
        dct[:, 0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(num_filters))
        return dct

    def _forward_strided(
        self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]
    ) -> torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)
        pow_spec = torch.matmul(pow_spec, self._fb)
        pow_spec = torch.max(pow_spec, self._eps).log()

        mfcc = torch.matmul(pow_spec, self._dct)
        if self.cepstral_lifter > 0:
            mfcc *= self._lifter

        # log_e is not None is needed by torchscript
        if self.use_energy and log_e is not None:
            mfcc[:, 0] = log_e

        return mfcc


def _get_strided_batch(
    waveform: torch.Tensor, window_length: int, window_shift: int, snip_edges: bool
) -> torch.Tensor:
    r"""Given a waveform (2D tensor of size ``(batch_size, num_samples)``,
    it returns a 2D tensor ``(batch_size, num_frames, window_length)``
    representing how the window is shifted along the waveform. Each row is a frame.
    Args:
        waveform (torch.Tensor): Tensor of size ``(batch_size, num_samples)``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.
    Returns:
        torch.Tensor: 3D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert waveform.dim() == 2
    batch_size = waveform.size(0)
    num_samples = waveform.size(-1)

    if snip_edges:
        if num_samples < window_length:
            return torch.empty((0, 0, 0))
        else:
            num_frames = 1 + (num_samples - window_length) // window_shift
    else:
        num_frames = (num_samples + (window_shift // 2)) // window_shift
        new_num_samples = (num_frames - 1) * window_shift + window_length
        npad = new_num_samples - num_samples
        npad_left = int((window_length - window_shift) // 2)
        npad_right = npad - npad_left
        # waveform = nn.functional.pad(waveform, (npad_left, npad_right), mode='reflect')
        pad_left = torch.flip(waveform[:, :npad_left], (1,))
        if npad_right >= 0:
            pad_right = torch.flip(waveform[:, -npad_right:], (1,))
        else:
            pad_right = torch.zeros(0, dtype=waveform.dtype)
        waveform = torch.cat((pad_left, waveform, pad_right), dim=1)

    strides = (
        waveform.stride(0),
        window_shift * waveform.stride(1),
        waveform.stride(1),
    )
    sizes = [batch_size, num_frames, window_length]
    return waveform.as_strided(sizes, strides)


def _get_strided_batch_streaming(
    waveform: torch.Tensor,
    window_shift: int,
    window_length: int,
    prev_remainder: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A variant of _get_strided_batch that creates short frames of a batch of audio signals
    in a way suitable for streaming. It accepts a waveform, window size parameters, and
    an optional buffer of previously unused samples. It returns a pair of waveform windows tensor,
    and unused part of the waveform to be passed as ``prev_remainder`` in the next call to this
    function.

    Example usage::

        >>> # get the first buffer of audio and make frames
        >>> waveform = get_incoming_audio_from_mic()
        >>> frames, remainder = _get_strided_batch_streaming(
        ...     waveform,
        ...     window_shift=160,
        ...     window_length=200,
        ... )
        >>>
        >>> process(frames)  # do sth with the frames
        >>>
        >>> # get the next buffer and use previous remainder to make frames
        >>> waveform = get_incoming_audio_from_mic()
        >>> frames, remainder = _get_strided_batch_streaming(
        ...     waveform,
        ...     window_shift=160,
        ...     window_length=200,
        ...     prev_remainder=prev_remainder,
        ... )

    .. caution:: This windowing mechanism only supports ``snip_edges=False``.

    :param waveform: A waveform tensor of shape ``(batch_size, num_samples)``.
    :param window_shift: The shift between frames measured in the number of samples.
    :param window_length: The number of samples in each window (frame).
    :param prev_remainder: An optional waveform tensor of shape ``(batch_size, num_samples)``.
        Can be ``None`` which indicates the start of a recording.
    :return: a pair of tensors with shapes ``(batch_size, num_frames, window_length)`` and
        ``(batch_size, remainder_len)``.
    """

    assert window_shift <= window_length
    assert waveform.dim() == 2
    batch_size = waveform.size(0)

    if prev_remainder is None:
        npad_left = int((window_length - window_shift) // 2)
        pad_left = torch.flip(waveform[:, :npad_left], (1,))
        waveform = torch.cat((pad_left, waveform), dim=1)
    else:
        assert prev_remainder.dim() == 2
        assert prev_remainder.size(0) == batch_size
        waveform = torch.cat((prev_remainder, waveform), dim=1)

    num_samples = waveform.size(-1)

    window_remainder = window_length - window_shift
    num_frames = (num_samples - window_remainder) // window_shift

    remainder = waveform[:, num_frames * window_shift :]

    strides = (
        waveform.stride(0),
        window_shift * waveform.stride(1),
        waveform.stride(1),
    )

    sizes = [batch_size, num_frames, window_length]

    return waveform.as_strided(sizes, strides), remainder


def _get_log_energy(x: torch.Tensor, energy_floor: float) -> torch.Tensor:
    """
    Returns the log energy of size (m) for a strided_input (m,*)
    """
    log_energy = (x.pow(2).sum(-1) + 1e-15).log()  # size (m)
    if energy_floor > 0.0:
        log_energy = torch.max(
            log_energy,
            torch.tensor(math.log(energy_floor), dtype=log_energy.dtype),
        )

    return log_energy


def create_mel_scale(
    num_filters: int,
    fft_length: int,
    sampling_rate: int,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    norm_filters: bool = True,
) -> torch.Tensor:
    if high_freq is None or high_freq == 0:
        high_freq = sampling_rate / 2
    if high_freq < 0:
        high_freq = sampling_rate / 2 + high_freq

    mel_low_freq = lin2mel(low_freq)
    mel_high_freq = lin2mel(high_freq)
    melfc = np.linspace(mel_low_freq, mel_high_freq, num_filters + 2)
    mels = lin2mel(np.linspace(0, sampling_rate, fft_length))

    B = np.zeros((int(fft_length / 2 + 1), num_filters), dtype=np.float32)
    for k in range(num_filters):
        left_mel = melfc[k]
        center_mel = melfc[k + 1]
        right_mel = melfc[k + 2]
        for j in range(int(fft_length / 2)):
            mel_j = mels[j]
            if left_mel < mel_j < right_mel:
                if mel_j <= center_mel:
                    B[j, k] = (mel_j - left_mel) / (center_mel - left_mel)
                else:
                    B[j, k] = (right_mel - mel_j) / (right_mel - center_mel)

    if norm_filters:
        B = B / np.sum(B, axis=0, keepdims=True)

    return torch.from_numpy(B)


def available_windows() -> List[str]:
    return [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]


HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"


def create_frame_window(window_size, window_type: str = "povey", blackman_coeff=0.42):
    r"""Returns a window function with the given type and size"""
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46)
    elif window_type == POVEY:
        return torch.hann_window(window_size, periodic=False).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, dtype=torch.get_default_dtype())
    elif window_type == BLACKMAN:
        a = 2 * math.pi / window_size
        window_function = torch.arange(window_size, dtype=torch.get_default_dtype())
        return (
            blackman_coeff
            - 0.5 * torch.cos(a * window_function)
            + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
        )
    else:
        raise Exception(f"Invalid window type: {window_type}")


def lin2mel(x):
    return 1127.0 * np.log(1 + x / 700)


def mel2lin(x):
    return 700 * (np.exp(x / 1127.0) - 1)


def next_power_of_2(x: int) -> int:
    """
    Returns the smallest power of 2 that is greater than x.

    Original source: TorchAudio (torchaudio/compliance/kaldi.py)
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()
