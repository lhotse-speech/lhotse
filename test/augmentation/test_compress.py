import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from lhotse import AudioSource, Recording
from lhotse.augmentation import Compress
from lhotse.augmentation.compress import (
    MP3_SUPPORTED_SAMPLING_RATES,
    OPUS_SUPPORTED_SAMPLING_RATES,
)
from lhotse.dataset.cut_transforms import Compress as CompressTransform


def mono_square_wave(sampling_rate: int = 48000) -> np.array:
    """
    Generate a 100Hz square wave.
    """
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    return 0.1 * signal.square(2 * np.pi * 100 * t)  # 100Hz square wave at -20dB


def create_recording_from_audio(
    audio: np.array, recording_id: str, sampling_rate: int = 48000
) -> Recording:
    """
    Create a Recording object from an audio signal.
    """
    import io

    import soundfile as sf

    # Create a BytesIO object to write the audio data
    buffer = io.BytesIO()

    # Write the audio data using soundfile
    sf.write(buffer, audio, sampling_rate, format="WAV", subtype="FLOAT")
    # Get the bytes from the buffer
    audio_bytes = buffer.getvalue()

    return Recording(
        id=recording_id,
        sources=[
            AudioSource(
                type="memory",
                channels=[0],
                source=audio_bytes,
            )
        ],
        sampling_rate=sampling_rate,
        num_samples=len(audio),
        duration=len(audio) / sampling_rate,
    )


@pytest.fixture(params=[8000, 12345, 16000, 22050, 48000])
def mono_square_wave_recording(request) -> Recording:
    sampling_rate = request.param
    return create_recording_from_audio(
        mono_square_wave(sampling_rate), "test_square", sampling_rate
    )


def test_compress_raises_on_invalid_codec(mono_square_wave_recording):
    # Test that compress raises ValueError on invalid codec
    with pytest.raises(ValueError):
        mono_square_wave_recording.compress("invalid_codec", 0.5)


def test_compress_raises_on_invalid_compression_level(mono_square_wave_recording):
    # Test that compress raises ValueError on invalid compression level
    with pytest.raises(ValueError):
        mono_square_wave_recording.compress("mp3", -0.1)  # Below 0.0
    with pytest.raises(ValueError):
        mono_square_wave_recording.compress("mp3", 1.1)  # Above 1.0


def unsupported_compression(codec, sampling_rate) -> bool:
    if codec == "gsm" and sampling_rate != 8000:
        return True
    if codec == "opus" and sampling_rate not in OPUS_SUPPORTED_SAMPLING_RATES:
        return True
    if codec == "mp3" and sampling_rate not in MP3_SUPPORTED_SAMPLING_RATES:
        return True
    return False


@pytest.mark.parametrize("codec", ["mp3", "opus", "vorbis"])
def test_compress_raises_on_invalid_combination_of_codec_and_sampling_rate(
    mono_square_wave_recording, codec
):
    if unsupported_compression(codec, mono_square_wave_recording.sampling_rate):
        with pytest.raises(sf.LibsndfileError):
            mono_square_wave_recording.compress(codec, 0.9).load_audio()


@pytest.mark.parametrize("codec", ["mp3", "opus", "vorbis", "gsm"])
@pytest.mark.parametrize("compression_level", [0.1, 0.5, 0.9])
def test_compress_alters_audio(mono_square_wave_recording, codec, compression_level):
    """
    Test that compression actually modifies the audio signal.
    """
    if unsupported_compression(codec, mono_square_wave_recording.sampling_rate):
        pytest.skip(
            f"Skipping compression test for unsupported combination of {codec} codec and sampling rate {mono_square_wave_recording.sampling_rate}Hz"
        )

    compressed_square = mono_square_wave_recording.compress(codec, compression_level)
    original_square_audio = mono_square_wave_recording.load_audio()[0]
    compressed_square_audio = compressed_square.load_audio()[0]

    assert not np.array_equal(original_square_audio, compressed_square_audio)


@pytest.mark.parametrize("codec", ["mp3", "opus", "vorbis", "gsm"])
@pytest.mark.parametrize("compression_level", [0.1, 0.5, 0.9])
def test_compress_preserves_rms(mono_square_wave_recording, codec, compression_level):
    """
    Test that compression preserves the RMS values within 5% in the frequency range of 100Hz to 12kHz.
    """
    if unsupported_compression(codec, mono_square_wave_recording.sampling_rate):
        pytest.skip(
            f"Skipping compression test for unsupported combination of {codec} codec and sampling rate {mono_square_wave_recording.sampling_rate}Hz"
        )

    compressed_square = mono_square_wave_recording.compress(codec, compression_level)
    original_square_audio = mono_square_wave_recording.load_audio()[0]
    compressed_square_audio = compressed_square.load_audio()[0]

    def calculate_band_rms(signal, sample_rate, low_freq=100, high_freq=12000):
        # Compute FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)

        # Create frequency mask for the band
        mask = (freqs >= low_freq) & (freqs <= high_freq)

        # Calculate RMS in the band
        band_rms = np.sqrt(np.mean(np.abs(fft[mask]) ** 2))

        return band_rms

    # Calculate RMS for both signals
    orig_rms = calculate_band_rms(
        original_square_audio, mono_square_wave_recording.sampling_rate
    )
    comp_rms = calculate_band_rms(
        compressed_square_audio, mono_square_wave_recording.sampling_rate
    )

    # Check that RMS is preserved within 5%
    rms_diff = np.abs(orig_rms - comp_rms) / orig_rms

    assert (
        rms_diff < 0.05
    ), f"Band RMS difference {rms_diff:.3f} exceeds 5% threshold for codec {codec} with compression level {compression_level}"


def test_compress_transforms_are_added(mono_square_wave_recording):
    # Test that compress transform is properly added to recording
    codec = "mp3"
    compression_level = 0.9

    compressed_recording = mono_square_wave_recording.compress(codec, compression_level)

    assert len(compressed_recording.transforms) == 1
    transform = compressed_recording.transforms[0]
    assert isinstance(transform, Compress)
    assert transform.codec == codec
    assert transform.compression_level == compression_level


@pytest.mark.parametrize("codec", ["mp3", "vorbis", "opus", "gsm"])
def test_compress_preserves_sampling_rate(mono_square_wave_recording, codec):
    # Test that compression preserves the sampling rate
    compression_level = 0.9

    compressed_recording = mono_square_wave_recording.compress(codec, compression_level)
    assert (
        compressed_recording.sampling_rate == mono_square_wave_recording.sampling_rate
    )


@pytest.mark.parametrize("codec", ["mp3", "vorbis", "opus", "gsm"])
def test_compress_preserves_duration(mono_square_wave_recording, codec):
    # Test that compression preserves the duration
    compression_level = 0.9

    compressed_recording = mono_square_wave_recording.compress(codec, compression_level)
    assert compressed_recording.duration == mono_square_wave_recording.duration


def test_compress_transform_raises_on_duplicate_codecs():
    with pytest.raises(AssertionError):
        CompressTransform(codecs=["mp3", "mp3"])


def test_compress_transform_raises_on_invalid_compression_level_interval():
    with pytest.raises(AssertionError):
        CompressTransform(codecs=["mp3"], compression_level=(0.1, 0.2, 0.3))


def test_compress_transform_raises_on_invalid_compression_level_interval_values():
    with pytest.raises(AssertionError):
        CompressTransform(codecs=["mp3"], compression_level=(0.5, 0.5))
    with pytest.raises(AssertionError):
        CompressTransform(codecs=["mp3"], compression_level=(0.6, 0.5))


def test_compress_transform_raises_on_invalid_probability():
    with pytest.raises(AssertionError):
        CompressTransform(codecs=["mp3"], p=-0.1)
    with pytest.raises(AssertionError):
        CompressTransform(codecs=["mp3"], p=1.1)


def test_compress_transform_raises_on_invalid_codec_weights_length():
    with pytest.raises(AssertionError):
        CompressTransform(codecs=["mp3", "opus"], codec_weights=[0.5])


def test_compress_transform_raises_on_negative_codec_weights():
    with pytest.raises(AssertionError):
        CompressTransform(codecs=["mp3", "opus"], codec_weights=[0.5, -0.5])
