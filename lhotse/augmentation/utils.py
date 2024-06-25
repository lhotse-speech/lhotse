from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import torch

try:
    # Pytorch >= 1.7
    from torch.fft import irfft, rfft
except ImportError:
    from torch import irfft, rfft

# Implementation based on torch-audiomentations:
# https://github.com/asteroid-team/torch-audiomentations/blob/master/torch_audiomentations/utils/convolution.py

_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.
    Note: This function was originally copied from the https://github.com/pyro-ppl/pyro
    repository, where the license was Apache 2.0. Any modifications to the original code can be
    found at https://github.com/asteroid-team/torch-audiomentations/commits
    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1


def convolve1d(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Computes the 1-d convolution of signal by kernel using FFTs.
    Both signal and kernel must be 1-dimensional.
    :param torch.Tensor signal: A signal to convolve.
    :param torch.Tensor kernel: A convolution kernel.
    :param str mode: One of: 'full', 'valid', 'same'.
    :return: torch.Tensor Convolution of signal with kernel. Returns the full convolution, i.e.,
        the output tensor will have size m + n - 1, where m is the length of the
        signal and n is the length of the kernel.
    """
    assert (
        signal.ndim == 1 and kernel.ndim == 1
    ), "signal and kernel must be 1-dimensional"
    m = signal.size(-1)
    n = kernel.size(-1)

    # Compute convolution using fft.
    padded_size = m + n - 1
    # Round up for cheaper fft.
    fast_ftt_size = next_fast_len(padded_size)
    f_signal = rfft(signal, n=fast_ftt_size)
    f_kernel = rfft(kernel, n=fast_ftt_size)
    f_result = f_signal * f_kernel
    result = irfft(f_result, n=fast_ftt_size)

    return result[:padded_size]


# The following is based on: https://github.com/yluo42/FRA-RIR/blob/main/FRA-RIR.py
@dataclass
class FastRandomRIRGenerator:
    sr: int = 16000
    direct_range: List = field(default_factory=lambda: [-6, 50])
    max_T60: float = 0.8
    alpha: float = 0.25
    a: float = -2.0
    b: float = 2.0
    tau: float = 0.2
    room_seed: Optional[int] = None
    source_seed: Optional[int] = None

    def __post_init__(self):
        self.room_rng = (
            np.random.default_rng(self.room_seed)
            if self.room_seed is not None
            else np.random.default_rng()
        )
        self.source_rng = (
            np.random.default_rng(self.source_seed)
            if self.source_seed is not None
            else np.random.default_rng()
        )

    def to_dict(self):
        return asdict(self)

    def __call__(self, nsource: int = 1) -> np.ndarray:
        """
        :param nsource: number of sources (RIR filters) to simulate. Default: 1.
        :return: simulated RIR filter for all sources, shape: (nsource, nsample)
        """
        from lhotse.augmentation.torchaudio import (
            check_for_torchaudio,
            get_or_create_resampler,
        )

        check_for_torchaudio()

        from torchaudio.functional import highpass_biquad

        # the sample rate at which the original RIR filter is generated
        ratio = 64
        sample_sr = self.sr * ratio

        # two resampling operations
        resample1 = get_or_create_resampler(sample_sr, sample_sr // int(np.sqrt(ratio)))
        resample2 = get_or_create_resampler(sample_sr // int(np.sqrt(ratio)), self.sr)

        eps = np.finfo(np.float16).eps

        # sample T60 of the room
        T60 = torch.from_numpy(self.room_rng.uniform(0.1, self.max_T60, size=(1,)))[
            0
        ].data

        # sample room-related statistics for calculating the reflection coefficient R
        R = torch.from_numpy(self.room_rng.uniform(0.1, 1.2, size=(1,)))[0].data

        # sample distance between the sound sources and the receiver (d_0) if not given
        direct_dist = torch.from_numpy(
            self.source_rng.uniform(0.2, 12.0, size=(nsource,))
        )

        # number of virtual sound sources
        image = self.sr * 2

        # sound velocity
        velocity = 340.0

        # indices of direct-path signals based on the sampled d_0
        direct_idx = torch.ceil(direct_dist * sample_sr / velocity).long()

        # length of the RIR filter based on the sampled T60
        rir_length = int(np.ceil(sample_sr * T60))

        # calculate the reflection coefficient based on the Eyring's empirical equation
        reflect_coef = (1 - (1 - torch.exp(-0.16 * R / T60)).pow(2)).sqrt()

        # randomly sample the propagation distance for all the virtual sound sources
        dist_range = [
            torch.linspace(1.0, velocity * T60 / direct_dist[i] - 1, image)
            for i in range(nsource)
        ]
        # a simple quadratic function
        dist_prob = torch.linspace(self.alpha, 1.0, image).pow(2)
        dist_prob = dist_prob / dist_prob.sum()
        dist_select_idx = dist_prob.multinomial(
            num_samples=image * nsource, replacement=True
        ).view(nsource, image)
        # the distance is sampled as a ratio between d_0 and each virtual sound sources
        dist_ratio = torch.stack(
            [dist_range[i][dist_select_idx[i]] for i in range(nsource)], 0
        )
        dist = direct_dist.view(-1, 1) * dist_ratio

        # sample the number of reflections (can be nonintegers)
        # calculate the maximum number of reflections
        reflect_max = (
            torch.log10(velocity * T60) - torch.log10(direct_dist) - 3
        ) / torch.log10(reflect_coef + eps)
        # calculate the number of reflections based on the assumption that
        # virtual sound sources which have longer propagation distances may reflect more frequently
        reflect_ratio = (dist / (velocity * T60)).pow(2) * (
            reflect_max.view(nsource, -1) - 1
        ) + 1
        # add a random pertubation based on the assumption that
        # virtual sound sources which have similar propagation distances can have different routes and reflection patterns
        reflect_pertub = torch.from_numpy(
            self.source_rng.uniform(self.a, self.b, size=(nsource, image))
        ) * (dist_ratio.pow(self.tau))
        # all virtual sound sources should reflect for at least once
        reflect_ratio = torch.maximum(reflect_ratio + reflect_pertub, torch.ones(1))

        # calculate the rescaled dirac comb as RIR filter
        dist = torch.cat([direct_dist.reshape(-1, 1), dist], 1)
        reflect_ratio = torch.cat([torch.zeros(nsource, 1), reflect_ratio], 1)
        rir = torch.zeros(nsource, rir_length)
        delta_idx = torch.minimum(
            torch.ceil(dist * sample_sr / velocity), torch.ones(1) * rir_length - 1
        ).long()
        delta_decay = reflect_coef.pow(reflect_ratio) / dist
        for i in range(nsource):
            rir[i][delta_idx[i]] += delta_decay[i]

        # a binary mask for direct-path RIR
        direct_mask = torch.zeros(nsource, rir_length).float()
        for i in range(nsource):
            direct_mask[
                i,
                max(direct_idx[i] + sample_sr * self.direct_range[0] // 1000, 0) : min(
                    direct_idx[i] + sample_sr * self.direct_range[1] // 1000, rir_length
                ),
            ] = 1.0
        rir_direct = rir * direct_mask

        # downsample
        all_rir = torch.stack([rir, rir_direct], 1).view(nsource * 2, -1)
        rir_downsample = resample1(all_rir)

        # apply high-pass filter
        rir_hp = highpass_biquad(rir_downsample, sample_sr // int(np.sqrt(ratio)), 80.0)

        # downsample again
        rir = resample2(rir_hp).float().view(nsource, 2, -1)

        # RIR filter and direct-path RIR filter at target sample rate
        rir_filter = rir[:, 0]  # nsource, nsample

        # convert to numpy array
        return rir_filter.numpy()
