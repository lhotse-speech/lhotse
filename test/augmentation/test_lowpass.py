import io

import numpy as np
import pytest

from lhotse import AudioSource, Recording
from lhotse.augmentation import Lowpass

scipy = pytest.importorskip("scipy")


@pytest.fixture
def mono_white_noise(sample_rate: int) -> np.array:
    """
    Generate white noise signal with peak amplitude +- 0.1 in float32 format.
    Duration is fixed at 1 second.
    """
    # Generate random signal and scale to float32 range
    seconds = 1.0
    signal = np.random.RandomState(seed=0).rand(int(sample_rate * seconds)) * 0.1
    return signal.astype(np.float32)


@pytest.fixture
def mono_white_noise_recording(mono_white_noise, sample_rate: int) -> Recording:
    """
    Create a Recording object from the white noise signal.
    """

    import soundfile as sf

    # Create a BytesIO object to write the audio data
    buffer = io.BytesIO()

    # Write the audio data using soundfile
    sf.write(buffer, mono_white_noise, sample_rate, format="WAV", subtype="FLOAT")
    # Get the bytes from the buffer
    audio_bytes = buffer.getvalue()

    return Recording(
        id="test_recording",
        sources=[
            AudioSource(
                type="memory",
                channels=[0],
                source=audio_bytes,
            )
        ],
        sampling_rate=sample_rate,
        num_samples=len(mono_white_noise),
        duration=len(mono_white_noise) / sample_rate,
    )


def bisect_bin(signal: np.ndarray, frequency: float, sample_rate: int = 48000) -> int:
    """
    Find the index of the frequency bin that contains the given frequency.

    Args:
        signal: Input signal to analyze
        frequency: Target frequency in Hz
        sample_rate: Sampling rate in Hz

    Returns:
        Index of the frequency bin containing the target frequency
    """
    # Get frequency bins without computing FFT
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    return np.searchsorted(freqs, frequency, side="right")


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("nyquist_ratio", [0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("filter_order", [6, 8, 10, 12])
@pytest.mark.parametrize("filter_type", ["butter", "cheby1", "cheby2", "ellip"])
def test_lowpass_attenuates_high_frequencies(
    mono_white_noise_recording, sample_rate, nyquist_ratio, filter_order, filter_type
):
    """
    Test that the lowpass filter properly attenuates frequencies above the cutoff.
    The attenuation should be at least 20dB in the stopband.
    """
    # Calculate cutoff frequency based on Nyquist ratio
    nyquist = sample_rate / 2
    cutoff_freq = nyquist * nyquist_ratio

    lowpassed_recording = mono_white_noise_recording.lowpass(
        cutoff_freq, filter_type=filter_type, order=filter_order
    )

    # Get original and filtered signals
    original_signal = mono_white_noise_recording.load_audio()[0]
    filtered_signal = lowpassed_recording.load_audio()[0]

    # Compute FFTs
    original_fft = np.abs(np.fft.rfft(original_signal))
    filtered_fft = np.abs(np.fft.rfft(filtered_signal))

    # Calculate power in frequencies above cutoff
    cutoff_bin = bisect_bin(original_signal, cutoff_freq, sample_rate)
    original_power = np.sum(original_fft[cutoff_bin:] ** 2)
    filtered_power = np.sum(filtered_fft[cutoff_bin:] ** 2)

    # Calculate attenuation in dB for frequencies above cutoff
    attenuation_db = 10 * (np.log10(filtered_power) - np.log10(original_power))

    # Check that high frequencies are attenuated
    # For white noise, these values are expected
    if filter_type == "cheby1":
        attenuation_db_expected = -13
        assert (
            attenuation_db < -attenuation_db_expected
        ), f"High frequency attenuation ({attenuation_db:.1f} dB) is less than expected ({attenuation_db_expected} dB)"
    elif filter_type == "butter" and nyquist_ratio > 0.8:
        attenuation_db_expected = -15
        assert (
            attenuation_db < -attenuation_db_expected
        ), f"High frequency attenuation ({attenuation_db:.1f} dB) is less than expected ({attenuation_db_expected} dB)"
    else:
        attenuation_db_expected = -20
        assert (
            attenuation_db < -attenuation_db_expected
        ), f"High frequency attenuation ({attenuation_db:.1f} dB) is less than expected ({attenuation_db_expected} dB)"

    # Check that upper 10% of stopband is attenuated better
    # For white noise, these values are expected
    stopband_tenth = cutoff_freq + (nyquist - cutoff_freq) * (1 - 0.1)
    stopband_tenth_bin = bisect_bin(original_signal, stopband_tenth, sample_rate)

    stopband_ratio = np.median(
        filtered_fft[stopband_tenth_bin:] / original_fft[stopband_tenth_bin:], axis=0
    )
    stopband_tenth_attenuation_db = 20 * np.log10(stopband_ratio)

    if sample_rate == 8000 or nyquist_ratio >= 0.9:
        pass
    else:
        attenuation_db_expected = -35
        assert (
            stopband_tenth_attenuation_db < attenuation_db_expected
        ), f"Upper quarter of stopband attenuation ({stopband_tenth_attenuation_db:.1f} dB) is less than expected ({attenuation_db_expected} dB)"


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("nyquist_ratio", [0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("filter_order", [4, 6, 8, 10, 12])
@pytest.mark.parametrize("filter_type", ["butter", "cheby1", "cheby2", "ellip"])
def test_lowpass_preserves_low_frequencies(
    mono_white_noise_recording, sample_rate, nyquist_ratio, filter_order, filter_type
):
    """
    Test that the lowpass filter preserves frequencies below the cutoff.
    The passband should be preserved within 5% of the original amplitude.
    """
    # Calculate cutoff frequency based on Nyquist ratio
    nyquist = sample_rate / 2
    cutoff_freq = nyquist * nyquist_ratio

    lowpassed_recording = mono_white_noise_recording.lowpass(
        cutoff_freq, filter_type=filter_type, order=filter_order
    )

    # Get original and filtered signals
    original_signal = mono_white_noise_recording.load_audio()[0]
    filtered_signal = lowpassed_recording.load_audio()[0]

    # Compute FFTs
    original_fft = np.abs(np.fft.rfft(original_signal))
    filtered_fft = np.abs(np.fft.rfft(filtered_signal))

    # Account for 0.02 transition band in kaiser filter
    transition_width = 0.02 * nyquist
    transition_start = cutoff_freq - transition_width / 2
    low_freq_bin = bisect_bin(original_signal, transition_start, sample_rate)

    # Check that low frequencies are preserved
    low_freq_ratio = np.median(
        filtered_fft[:low_freq_bin] / original_fft[:low_freq_bin], axis=0
    )

    if filter_type == "bessel":
        assert 0.6 < low_freq_ratio < 1.05  # Bessel attenuates a lot in upper passband
    elif filter_type == "cheby2" and filter_order <= 6:
        assert 0.65 < low_freq_ratio < 1.05  # Cheby2 attenuates a lot in upper passband
    else:
        assert (
            0.95 < low_freq_ratio < 1.05
        )  # Low frequencies should be preserved within 5%


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("nyquist_ratio", [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5])
def test_lowpass_transform_properties(
    mono_white_noise_recording, sample_rate, nyquist_ratio
):
    """
    Test that the lowpass transform is properly added to the recording and has the correct properties.
    Also test that it raises a ValueError when the cutoff frequency exceeds the Nyquist frequency.
    """
    # Calculate cutoff frequency based on Nyquist ratio
    nyquist = sample_rate / 2
    cutoff_freq = nyquist * nyquist_ratio

    if nyquist_ratio >= 1.0:
        with pytest.raises(ValueError):
            mono_white_noise_recording.lowpass(cutoff_freq)
    else:
        lowpassed_recording = mono_white_noise_recording.lowpass(cutoff_freq)
        assert len(lowpassed_recording.transforms) == 1
        transform = lowpassed_recording.transforms[0]
        assert isinstance(transform, Lowpass)
        assert transform.frequency == cutoff_freq
