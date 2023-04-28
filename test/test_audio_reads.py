from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import torch
import torchaudio

import lhotse
from lhotse import Recording
from lhotse.audio import read_opus_ffmpeg, read_opus_torchaudio, torchaudio_load


@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.wav",
        "test/fixtures/mono_c1.wav",
        "test/fixtures/stereo.wav",
        "test/fixtures/libri/libri-1088-134315-0000.wav",
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
        "test/fixtures/stereo.sph",
        "test/fixtures/stereo.mp3",
        "test/fixtures/common_voice_en_651325.mp3",
    ],
)
def test_info_and_read_audio_consistency(path):
    recording = Recording.from_file(path)
    audio = recording.load_audio()
    assert audio.shape[1] == recording.num_samples


@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.wav",
        "test/fixtures/mono_c1.wav",
        "test/fixtures/stereo.wav",
        "test/fixtures/libri/libri-1088-134315-0000.wav",
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
        "test/fixtures/stereo.mp3",
        "test/fixtures/common_voice_en_651325.mp3",
    ],
)
@pytest.mark.parametrize("offset", [0, 0.1])
@pytest.mark.parametrize("duration", [None, 0.1])
def test_torchaudio_load_with_offset_duration_works(path, offset, duration):
    audio, sr = torchaudio_load(path, offset=offset, duration=duration)
    # just test that it runs -- the assertions are inside the function


@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
    ],
)
def test_opus_torchaudio_vs_ffmpeg(path):
    audio_ta, sr_ta = read_opus_torchaudio(path)
    audio_ff, sr_ff = read_opus_ffmpeg(path)
    assert sr_ta == sr_ff
    assert audio_ta.shape == audio_ff.shape
    # Apparently FFMPEG and SOX (torchaudio) apply different decoders
    # and/or resampling algorithms for reading OPUS, so they yield
    # different results up to 3rd decimal place (for 16bit PCM,
    # this is affecting around 6 least significant bits)
    np.testing.assert_almost_equal(audio_ta, audio_ff, decimal=3)


def test_resample_opus():
    # Test that reading OPUS files after resampling
    # does not raise an exception.
    r = Recording.from_file("test/fixtures/mono_c0.opus")
    r.load_audio()
    r1 = r.resample(24000)
    r1.load_audio()


@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
    ],
)
@pytest.mark.parametrize(
    "force_opus_sampling_rate",
    [
        pytest.param(
            8000, marks=pytest.mark.xfail(reason="Mismatch in shape by one sample.")
        ),
        16000,
        pytest.param(
            22050, marks=pytest.mark.xfail(reason="Mismatch in shape by one sample.")
        ),
        24000,
        pytest.param(
            32000, marks=pytest.mark.xfail(reason="Mismatch in shape by one sample.")
        ),
        44100,
        48000,
    ],
)
def test_opus_torchaudio_vs_ffmpeg_with_resampling(path, force_opus_sampling_rate):
    audio_ta, sr_ta = read_opus_torchaudio(
        path, force_opus_sampling_rate=force_opus_sampling_rate
    )
    audio_ff, sr_ff = read_opus_ffmpeg(
        path, force_opus_sampling_rate=force_opus_sampling_rate
    )
    assert sr_ta == sr_ff
    # Note: for some resampling rates, there will be mismatch by one
    # sample. Recording.load_audio() will fix these cases.
    assert audio_ta.shape == audio_ff.shape
    # Note: when we resample, for a very small number of samples,
    # there are typically discrepancies up to second decimal place
    # between the two implementations.
    # I won't fight the audio codec world -- it is what it is.
    np.testing.assert_almost_equal(audio_ta, audio_ff, decimal=1)


def test_audio_caching_enabled_works():
    lhotse.set_caching_enabled(True)  # Enable caching.

    np.random.seed(89)  # Reproducibility.

    # Prepare two different waveforms.
    noise1 = np.random.rand(1, 32000).astype(np.float32)
    noise2 = np.random.rand(1, 32000).astype(np.float32)
    # Sanity check -- the noises are different
    assert np.abs(noise1 - noise2).sum() != 0

    # Save the first waveform in a file.
    with NamedTemporaryFile(suffix=".wav") as f:
        torchaudio.save(f.name, torch.from_numpy(noise1), sample_rate=16000)
        recording = Recording.from_file(f.name)

        # Read the audio -- should be equal to noise1.
        audio = recording.load_audio()
        np.testing.assert_allclose(audio, noise1, atol=3e-5)

        # Save noise2 to the same location.
        torchaudio.save(f.name, torch.from_numpy(noise2), sample_rate=16000)

        # Read the audio -- should *still* be equal to noise1,
        # because reading from this path was cached before.
        audio = recording.load_audio()
        np.testing.assert_allclose(audio, noise1, atol=3e-5)


def test_audio_caching_disabled_works():
    lhotse.set_caching_enabled(False)  # Disable caching.

    np.random.seed(89)  # Reproducibility.

    # Prepare two different waveforms.
    noise1 = np.random.rand(1, 32000).astype(np.float32)
    noise2 = np.random.rand(1, 32000).astype(np.float32)
    # Sanity check -- the noises are different
    assert np.abs(noise1 - noise2).sum() != 0

    # Save the first waveform in a file.
    with NamedTemporaryFile(suffix=".wav") as f:
        torchaudio.save(f.name, torch.from_numpy(noise1), sample_rate=16000)
        recording = Recording.from_file(f.name)

        # Read the audio -- should be equal to noise1.
        audio = recording.load_audio()
        np.testing.assert_allclose(audio, noise1, atol=3e-5)

        # Save noise2 to the same location.
        torchaudio.save(f.name, torch.from_numpy(noise2), sample_rate=16000)

        # Read the audio -- should be equal to noise2,
        # and the caching is ignored (doesn't happen).
        audio = recording.load_audio()
        np.testing.assert_allclose(audio, noise2, atol=3e-5)
