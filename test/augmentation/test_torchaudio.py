import math

import pytest
import torch
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_array_almost_equal

torchaudio = pytest.importorskip("torchaudio", minversion="0.6")

from lhotse.augmentation import (
    SoxEffectTransform,
    pitch,
    reverb,
    speed,
    volume,
    Speed,
    Tempo,
    Volume,
    ReverbWithImpulseResponse,
)
from lhotse import AudioTransform, MonoCut, Recording, Resample, Seconds

SAMPLING_RATE = 16000


@pytest.fixture
def audio():
    return torch.sin(2 * math.pi * torch.linspace(0, 1, 16000)).unsqueeze(0).numpy()


@pytest.fixture(scope="module")
def recording():
    return Recording.from_file("test/fixtures/libri/libri-1088-134315-0000.wav")


# to avoid clipping during volume perturbation test
@pytest.fixture
def audio_volume():
    return (
        torch.sin(2 * math.pi * torch.linspace(0, 1, 16000)).unsqueeze(0).numpy() / 3.0
    )


@pytest.fixture(scope="module")
def rir():
    return Recording.from_file("test/fixtures/rir/sim_1ch.wav")


@pytest.fixture(scope="module")
def multi_channel_rir():
    return Recording.from_file("test/fixtures/rir/real_8ch.wav")


@pytest.mark.parametrize("effect", [reverb, pitch, speed, volume])
def test_example_augmentation(audio, effect):
    augment_fn = SoxEffectTransform(effects=effect(SAMPLING_RATE))
    augmented_audio = augment_fn(audio, sampling_rate=SAMPLING_RATE)
    assert augmented_audio.shape == audio.shape
    assert augmented_audio != audio


def test_speed_does_not_change_num_samples(audio):
    augment_fn = SoxEffectTransform(effects=speed(SAMPLING_RATE))
    # Since speed() is not deterministic and between 0.9x - 1.1x, multiple invocations
    # will yield either slower (more samples) or faster (less samples) signal.
    # The truncation/padding is performed inside of SoxEffectTransform so the user should not
    # see these changes.
    for _ in range(10):
        augmented_audio = augment_fn(audio, sampling_rate=SAMPLING_RATE)
        assert augmented_audio.shape == audio.shape
        assert augmented_audio != audio


def test_volume_does_not_change_num_samples(audio):
    augment_fn = SoxEffectTransform(effects=volume(SAMPLING_RATE))
    for _ in range(10):
        augmented_audio = augment_fn(audio, sampling_rate=SAMPLING_RATE)
        assert augmented_audio.shape == audio.shape
        assert augmented_audio != audio


@pytest.mark.parametrize("early_only", [True, False])
def test_reverb_does_not_change_num_samples(audio, rir, early_only):
    augment_fn = ReverbWithImpulseResponse(rir=rir, early_only=early_only)
    for _ in range(10):
        augmented_audio = augment_fn(audio, sampling_rate=SAMPLING_RATE)
        assert augmented_audio.shape == (1, 16000)


@pytest.mark.parametrize(
    "rir_channels, expected_num_channels", [(None, 8), ([0], 1), ([2], 1), ([0, 4], 2)]
)
def test_reverb_with_multi_channel_impulse_response(
    audio, multi_channel_rir, rir_channels, expected_num_channels
):
    augment_fn = ReverbWithImpulseResponse(
        rir=multi_channel_rir, rir_channels=rir_channels
    )
    for _ in range(10):
        augmented_audio = augment_fn(audio, sampling_rate=SAMPLING_RATE)
        assert augmented_audio.shape == (expected_num_channels, 16000)


@pytest.mark.parametrize("normalize_output", [True, False])
@pytest.mark.parametrize("early_only", [True, False])
def test_reverb_normalize_output(audio, rir, normalize_output, early_only):
    augment_fn = ReverbWithImpulseResponse(
        rir=rir, normalize_output=normalize_output, early_only=early_only
    )
    orig_energy = np.sum(np.abs(audio) ** 2) / audio.shape[1]
    for _ in range(10):
        augmented_audio = augment_fn(audio, sampling_rate=SAMPLING_RATE)
        rvb_energy = np.sum(np.abs(augmented_audio) ** 2) / augmented_audio.shape[1]
        if normalize_output:
            assert_array_almost_equal(rvb_energy, orig_energy)


def test_speed(audio):
    speed = Speed(factor=1.1)
    perturbed = speed(audio, SAMPLING_RATE)
    assert perturbed.shape == (1, 14545)


@pytest.mark.parametrize("scale", [0.125, 1.0, 2.0])
def test_volume(audio_volume, scale):
    volume = Volume(factor=scale)
    audio_perturbed = volume(audio_volume, SAMPLING_RATE)

    assert audio_perturbed.shape == audio_volume.shape
    assert_array_almost_equal(audio_perturbed, scale * audio_volume)


def test_deserialize_transform_speed(audio):
    speed = AudioTransform.from_dict({"name": "Speed", "kwargs": {"factor": 1.1}})
    perturbed_speed = speed(audio, SAMPLING_RATE)

    assert perturbed_speed.shape == (1, 14545)


def test_deserialize_transform_volume(audio):
    volume = AudioTransform.from_dict({"name": "Volume", "kwargs": {"factor": 0.5}})
    perturbed_volume = volume(audio, SAMPLING_RATE)

    assert perturbed_volume.shape == audio.shape
    assert_array_almost_equal(perturbed_volume, audio * 0.5)


def test_serialize_deserialize_transform_speed(audio):
    speed_orig = Speed(factor=1.1)
    data_speed = speed_orig.to_dict()
    speed = AudioTransform.from_dict(data_speed)
    perturbed_speed = speed(audio, SAMPLING_RATE)

    assert perturbed_speed.shape == (1, 14545)


def test_serialize_deserialize_transform_volume(audio):
    volume_orig = Volume(factor=0.5)
    data_volume = volume_orig.to_dict()
    volume = AudioTransform.from_dict(data_volume)
    perturbed_volume = volume(audio, SAMPLING_RATE)

    assert perturbed_volume.shape == audio.shape
    assert_array_almost_equal(perturbed_volume, audio * 0.5)


@pytest.mark.parametrize("recording_to_dict", [True, False])
def test_serialize_deserialize_transform_reverb(audio, rir, recording_to_dict):
    rir = rir.to_dict() if recording_to_dict else rir
    reverb_orig = ReverbWithImpulseResponse(rir=rir)
    data_reverb = reverb_orig.to_dict()
    reverb = ReverbWithImpulseResponse.from_dict(data_reverb)
    assert reverb_orig == reverb


@pytest.mark.parametrize("sampling_rate", [8000, 16000, 22050, 32000, 44100, 48000])
def test_resample(audio, sampling_rate):
    resample = Resample(source_sampling_rate=16000, target_sampling_rate=sampling_rate)
    resampled = resample(audio, SAMPLING_RATE)
    assert resampled.shape == (1, sampling_rate)


def test_tempo(audio):
    tempo = Tempo(factor=1.1)
    perturbed = tempo(audio, SAMPLING_RATE)
    assert perturbed.shape == (1, 14545)


@settings(deadline=None, print_blob=True, max_examples=200)
@given(
    # Target sampling rates
    target_sampling_rate=st.one_of(
        [st.just(v) for v in [8000, 22050, 32000, 44100, 48000]]
    ),
    # Speed perturbation values
    sp_factor=st.one_of([st.just(v) for v in [0.9, 0.95, 1.05, 1.1]]),
    # Volume perturbation values
    vp_factor=st.one_of([st.just(v) for v in [0.125, 0.5, 1.5, 2.0]]),
    # Apply reverb with impulse response?
    reverb=st.booleans(),
    # Resampling first?
    resample_first=st.booleans(),
    # Cut duration (full recording has 16.04s)
    cut_duration=st.floats(min_value=1.1, max_value=12.037575757575),
)
def test_augmentation_chain_randomized(
    recording: Recording,
    rir: Recording,
    target_sampling_rate: int,
    sp_factor: float,
    vp_factor: float,
    reverb: bool,
    resample_first: bool,
    cut_duration: Seconds,
):
    if resample_first:
        recording_aug = (
            recording.resample(target_sampling_rate)
            .perturb_speed(sp_factor)
            .perturb_volume(vp_factor)
        )
    else:
        recording_aug = (
            recording.perturb_speed(sp_factor)
            .resample(target_sampling_rate)
            .perturb_volume(vp_factor)
        )
    if reverb:
        recording_aug = recording_aug.reverb_rir(rir)

    audio_aug = recording_aug.load_audio()
    assert audio_aug.shape[1] == recording_aug.num_samples

    cut_aug = MonoCut(
        id="dummy",
        start=0.5125,
        duration=cut_duration,
        channel=0,
        recording=recording_aug,
    )
    assert cut_aug.load_audio().shape[1] == cut_aug.num_samples
