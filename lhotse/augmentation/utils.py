import torch

try:
    # Pytorch >= 1.7
    from torch.fft import rfft, irfft
except ImportError:
    from torch import rfft, irfft

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
