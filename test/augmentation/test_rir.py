from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from numpy.testing import assert_allclose

from lhotse import Recording
from lhotse.augmentation import ReverbWithImpulseResponse


def impulse_recording(gain):
    stream = BytesIO()
    sf.write(
        stream,
        np.array([gain, 0, 0], dtype=np.float32),
        16000,
        format="WAV",
        subtype="FLOAT",
    )
    return Recording.from_bytes(stream.getvalue(), recording_id="impulse")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "amplitude,gain", [(1, 1), (1, 1e-17), (1, 1e-25), (1e-25, 1), (1, 1e30)]
)
def test_reverb_normalization_preserves_energy(dtype, amplitude, gain):
    samples = (0.25 * np.sin(np.arange(256) * 0.07)).astype(dtype)[None, :]
    samples *= amplitude
    transform = ReverbWithImpulseResponse(rir=impulse_recording(gain))

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = transform(samples, sampling_rate=16000)

    # A positive single-tap RIR only changes gain. Normalization must undo it.
    assert actual.dtype == samples.dtype
    assert actual.shape == samples.shape
    assert_allclose(actual, samples, rtol=1e-5, atol=amplitude * 1e-7)


@pytest.mark.parametrize("normalize_output", [True, False])
def test_reverb_normalization_preserves_silence(normalize_output):
    samples = np.zeros((1, 256), dtype=np.float32)
    transform = ReverbWithImpulseResponse(
        rir=impulse_recording(1), normalize_output=normalize_output
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = transform(samples, sampling_rate=16000)
    assert_allclose(actual, samples, atol=0)


@pytest.mark.parametrize("normalize_output", [True, False])
def test_reverb_normalization_is_per_channel(normalize_output):
    samples = (0.25 * np.sin(np.arange(256) * 0.07)).astype(np.float32)
    samples = np.stack([samples, samples * 1e-25])
    transform = ReverbWithImpulseResponse(
        rir=impulse_recording(1), normalize_output=normalize_output
    )
    actual = transform(samples, sampling_rate=16000)
    factor = 1 if normalize_output else transform.RIR_SCALING_FACTOR
    assert actual.dtype == samples.dtype
    assert actual.shape == samples.shape
    for before, after in zip(samples, actual):
        assert_allclose(after, before * factor, rtol=1e-5, atol=before.max() * 1e-7)
