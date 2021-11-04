from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import torch
import torchaudio

import lhotse
from lhotse import Recording


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
        np.testing.assert_almost_equal(audio, noise1)

        # Save noise2 to the same location.
        torchaudio.save(f.name, torch.from_numpy(noise2), sample_rate=16000)

        # Read the audio -- should *still* be equal to noise1,
        # because reading from this path was cached before.
        audio = recording.load_audio()
        np.testing.assert_almost_equal(audio, noise1)


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
        np.testing.assert_almost_equal(audio, noise1)

        # Save noise2 to the same location.
        torchaudio.save(f.name, torch.from_numpy(noise2), sample_rate=16000)

        # Read the audio -- should be equal to noise2,
        # and the caching is ignored (doesn't happen).
        audio = recording.load_audio()
        np.testing.assert_almost_equal(audio, noise2)
