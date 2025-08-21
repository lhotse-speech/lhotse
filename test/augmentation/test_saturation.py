import numpy as np
import pytest

from lhotse.augmentation import Saturation


@pytest.fixture
def mono_sine_wave() -> np.array:
    seconds = 1.0
    sample_rate = 16000
    t = np.linspace(0, seconds, int(sample_rate * seconds))
    signal = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    return signal


@pytest.mark.parametrize("hard", [True, False])
@pytest.mark.parametrize("gain_db", [0.0, 3.0, 6.0, 12.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_saturation_initialization_valid_params(hard, gain_db, normalize):
    saturation = Saturation(hard=hard, gain_db=gain_db, normalize=normalize)
    assert saturation.hard is hard
    assert saturation.gain_db == gain_db
    assert saturation.normalize is normalize


@pytest.mark.parametrize("gain_db", [0.0, 3.0, 6.0, 12.0])
def test_saturation_hard_clipping(mono_sine_wave, gain_db):
    saturation = Saturation(hard=True, gain_db=gain_db, normalize=True)

    saturated_signal = saturation(mono_sine_wave, 16000)
    assert saturated_signal.shape == mono_sine_wave.shape

    # For hard clipping, the output should be under -gain_db dBFS
    max_expected_amplitude = 10 ** (-gain_db / 20)
    assert np.max(np.abs(saturated_signal)) <= max_expected_amplitude + 1e-6


@pytest.mark.parametrize("gain_db", [-3.0, -6.0, -12.0])
def test_saturation_hard_clipping_negative_gain_db(mono_sine_wave, gain_db):
    saturation = Saturation(hard=True, gain_db=gain_db, normalize=True)
    saturated_signal = saturation(mono_sine_wave, 16000)
    assert np.allclose(saturated_signal, mono_sine_wave)


@pytest.mark.parametrize("gain_db", [3.0, 6.0, 12.0])
def test_saturation_soft_tanh(mono_sine_wave, gain_db):
    saturation = Saturation(hard=False, gain_db=gain_db, normalize=True)

    # Apply saturation
    saturated_signal = saturation(mono_sine_wave, 16000)

    # Check that output has correct shape
    assert saturated_signal.shape == mono_sine_wave.shape

    # For soft saturation with tanh, values should stay within reasonable bounds
    # tanh asymptotes to ±1, so values should be less than 1.0
    assert np.max(np.abs(saturated_signal)) < 1.0

    # With gain, soft saturation should create a smooth, compressed waveform
    # The signal should be visibly different from the original
    assert not np.allclose(saturated_signal, mono_sine_wave, rtol=1e-2)


@pytest.mark.parametrize("hard", [True, False])
@pytest.mark.parametrize("gain_db", [0.0, 3.0, 6.0, 12.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_saturation_silence(hard, gain_db, normalize):
    silence = np.zeros(16000, dtype=np.float32)
    saturation = Saturation(hard=hard, gain_db=gain_db, normalize=normalize)

    saturated_signal = saturation(silence, 16000)

    # Silence should remain silence
    assert np.allclose(saturated_signal, silence)


@pytest.mark.parametrize("hard", [True, False])
def test_saturation_small_signal(hard):
    # Create a quiet sine wave that won't saturate much with tanh
    t = np.linspace(0, 1, 16000)
    peak_db = -20
    peak = 10 ** (peak_db / 20)
    quiet_signal = peak * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    saturation = Saturation(hard=hard, gain_db=0.0, normalize=True)
    saturated_signal = saturation(quiet_signal, 16000)

    assert saturated_signal.shape == quiet_signal.shape

    if hard:
        assert np.allclose(saturated_signal, quiet_signal, rtol=1e-3)
    else:
        # For small inputs, tanh(x) ≈ x, so they should be close but not identical
        assert np.allclose(saturated_signal, quiet_signal, atol=0.05)


def test_saturation_reverse_timestamps():
    saturation = Saturation(hard=False, gain_db=0.0, normalize=True)

    offset = 1.5
    duration = 3.0
    sampling_rate = 16000

    reversed_offset, reversed_duration = saturation.reverse_timestamps(
        offset, duration, sampling_rate
    )

    assert reversed_offset == offset
    assert reversed_duration == duration


@pytest.mark.parametrize("hard", [True, False])
@pytest.mark.parametrize("gain_db", [0.0, 3.0, 6.0, 12.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_saturation_serialization(hard, gain_db, normalize):
    saturation = Saturation(hard=hard, gain_db=gain_db, normalize=normalize)

    saturation_dict = saturation.to_dict()
    assert saturation_dict["name"] == "Saturation"
    assert saturation_dict["kwargs"]["hard"] is hard
    assert saturation_dict["kwargs"]["gain_db"] == gain_db
    assert saturation_dict["kwargs"]["normalize"] is normalize

    saturation_restored = Saturation.from_dict(saturation_dict)
    assert isinstance(saturation_restored, Saturation)
    assert saturation_restored.hard is hard
    assert saturation_restored.gain_db == gain_db
    assert saturation_restored.normalize is normalize


@pytest.mark.parametrize("gain_db", [0.0, 3.0, 6.0, 12.0])
def test_saturation_various_gains(mono_sine_wave, gain_db):
    saturation = Saturation(hard=False, gain_db=gain_db, normalize=True)

    saturated_signal = saturation(mono_sine_wave, 16000)

    # Check that output shape is preserved
    assert saturated_signal.shape == mono_sine_wave.shape

    # Higher gains should produce more saturation
    if gain_db > 0:
        # With gain, signal should be different from original
        assert not np.allclose(saturated_signal, mono_sine_wave, rtol=1e-2)
        assert np.max(np.abs(saturated_signal)) <= np.max(np.abs(mono_sine_wave))
        assert np.all(np.abs(saturated_signal) <= np.abs(mono_sine_wave) + 1e-6)


def test_saturation_with_normalization_disabled(mono_sine_wave):
    # Create a quieter version of the sine wave
    quiet_sine = mono_sine_wave * 0.1  # -20dB relative to full scale

    saturation = Saturation(hard=False, gain_db=6.0, normalize=False)
    saturated_signal = saturation(quiet_sine, 16000)

    assert saturated_signal.shape == quiet_sine.shape

    # Peak value after saturation can't be higher than the input peak value
    output_max = np.max(np.abs(saturated_signal))
    input_max = np.max(np.abs(quiet_sine))
    assert output_max <= input_max + 1e-6
