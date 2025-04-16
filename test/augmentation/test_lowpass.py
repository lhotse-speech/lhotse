import numpy as np
import pytest

from lhotse import AudioSource, Recording
from lhotse.augmentation import Lowpass


@pytest.fixture
def mono_white_noise(sample_rate: int) -> np.array:
    """
    Generate white noise signal with peak amplitude +- 0.1 in float32 format.
    Duration is fixed at 1 second.
    """
    # Generate random signal and scale to float32 range
    signal = np.random.RandomState(seed=0).rand(int(sample_rate * 1.0)) * 0.1
    return signal.astype(np.float32)


@pytest.fixture
def mono_white_noise_recording(mono_white_noise, sample_rate: int) -> Recording:
    """
    Create a Recording object from the white noise signal.
    """
    import io

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
def test_lowpass_attenuates_high_frequencies(
    mono_white_noise_recording, sample_rate, nyquist_ratio
):
    """
    Test that the lowpass filter properly attenuates frequencies above the cutoff.
    The attenuation should be at least 40dB in the stopband.
    """
    # Calculate cutoff frequency based on Nyquist ratio
    nyquist = sample_rate / 2
    cutoff_freq = nyquist * nyquist_ratio

    lowpassed_recording = mono_white_noise_recording.lowpass(cutoff_freq)

    # Get original and filtered signals
    original_signal = mono_white_noise_recording.load_audio()[0]
    filtered_signal = lowpassed_recording.load_audio()[0]

    # Compute FFTs
    original_fft = np.abs(np.fft.rfft(original_signal))
    filtered_fft = np.abs(np.fft.rfft(filtered_signal))

    # Calculate transition band width (2% of Nyquist frequency)
    transition_width = 0.02 * nyquist
    transition_end = cutoff_freq + transition_width / 2
    cutoff_bin = bisect_bin(original_signal, transition_end, sample_rate)

    # Check that high frequencies are attenuated
    high_freq_ratio = np.median(filtered_fft[cutoff_bin:], axis=0) / np.median(
        original_fft[cutoff_bin:], axis=0
    )
    assert high_freq_ratio < 10 ** (
        -40 / 20
    )  # High frequencies should be attenuated by at least 40 dB


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("nyquist_ratio", [0.25, 0.5, 0.75, 0.9])
def test_lowpass_preserves_low_frequencies(
    mono_white_noise_recording, sample_rate, nyquist_ratio
):
    """
    Test that the lowpass filter preserves frequencies below the cutoff.
    The passband should be preserved within 5% of the original amplitude.
    """
    # Calculate cutoff frequency based on Nyquist ratio
    nyquist = sample_rate / 2
    cutoff_freq = nyquist * nyquist_ratio

    lowpassed_recording = mono_white_noise_recording.lowpass(cutoff_freq)

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
    low_freq_ratio = np.median(filtered_fft[:low_freq_bin], axis=0) / np.median(
        original_fft[:low_freq_bin], axis=0
    )
    assert 0.95 < low_freq_ratio < 1.05  # Low frequencies should be preserved within 5%


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
