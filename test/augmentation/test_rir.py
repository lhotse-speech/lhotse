import sys
from io import BytesIO
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf
import torch
from numpy.testing import assert_allclose

from lhotse import Recording
from lhotse.augmentation import ReverbWithImpulseResponse
from lhotse.augmentation.utils import FastRandomRIRGenerator


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


@pytest.mark.parametrize("nsource", [1, 3])
def test_fast_rir_sums_colliding_arrivals(monkeypatch, nsource):
    # Keep only the impulse comb and force every image to the direct distance.
    # Every path for a source then arrives in the same sample.
    monkeypatch.setattr(
        "lhotse.augmentation.torchaudio.get_or_create_resampler",
        lambda *args: lambda x: x,
    )
    monkeypatch.setattr(
        "lhotse.augmentation.torchaudio.check_for_torchaudio", lambda: None
    )
    functional = ModuleType("torchaudio.functional")
    functional.highpass_biquad = lambda x, *args: x
    monkeypatch.setitem(sys.modules, "torchaudio.functional", functional)
    monkeypatch.setattr(
        torch.Tensor,
        "multinomial",
        lambda self, num_samples, replacement: torch.zeros(
            num_samples, dtype=torch.long
        ),
    )
    sr, t60 = 1000, 0.1
    generator = FastRandomRIRGenerator(
        sr=sr, max_T60=t60, a=0.0, b=0.0, room_seed=17, source_seed=23
    )
    actual = generator(nsource=nsource)

    # Independent scalar image-source sum, before filtering and resampling.
    rng = np.random.default_rng(17)
    rng.uniform(0.1, t60)  # The generator's T60 draw.
    room_ratio = rng.uniform(0.1, 1.2)
    distances = np.random.default_rng(23).uniform(0.2, 12.0, size=nsource)
    coefficient = np.sqrt(1 - (1 - np.exp(-0.16 * room_ratio / t60)) ** 2)
    expected = np.zeros_like(actual)
    for i, distance in enumerate(distances):
        max_reflections = (np.log10(340 * t60) - np.log10(distance) - 3) / np.log10(
            coefficient + np.finfo(np.float16).eps
        )
        reflections = max(
            1.0, (distance / (340 * t60)) ** 2 * (max_reflections - 1) + 1
        )
        arrival = int(np.ceil(distance * sr * 64 / 340))
        expected[i, arrival] = (1 + 2 * sr * coefficient**reflections) / distance
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
