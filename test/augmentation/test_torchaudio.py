import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_array_almost_equal

torchaudio = pytest.importorskip('torchaudio', minversion='0.6')

from lhotse.augmentation import SoxEffectTransform, pitch, reverb, speed, volume, Speed, Tempo, Volume
from lhotse import AudioTransform, MonoCut, Recording, Resample, Seconds

SAMPLING_RATE = 16000


@pytest.fixture
def audio():
    return torch.sin(2 * math.pi * torch.linspace(0, 1, 16000)).unsqueeze(0).numpy()


# to avoid clipping during volume perturbation test
@pytest.fixture
def audio_volume():
    return torch.sin(2 * math.pi * torch.linspace(0, 1, 16000)).unsqueeze(0).numpy() / 3.


@pytest.mark.parametrize('effect', [reverb, pitch, speed, volume])
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


def test_speed(audio):
    speed = Speed(factor=1.1)
    perturbed = speed(audio, SAMPLING_RATE)
    assert perturbed.shape == (1, 14545)


@pytest.mark.parametrize('scale', [0.125, 1., 2.])
def test_volume(audio_volume, scale):
    volume = Volume(factor=scale)
    audio_perturbed = volume(audio_volume, SAMPLING_RATE)

    assert audio_perturbed.shape == audio_volume.shape
    assert_array_almost_equal(audio_perturbed, scale * audio_volume)


def test_deserialize_transform_speed(audio):
    speed = AudioTransform.from_dict({'name': 'Speed', 'kwargs': {'factor': 1.1}})
    perturbed_speed = speed(audio, SAMPLING_RATE)

    assert perturbed_speed.shape == (1, 14545)


def test_deserialize_transform_volume(audio):
    volume = AudioTransform.from_dict({'name': 'Volume', 'kwargs': {'factor': 0.5}})
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


@pytest.mark.parametrize('sampling_rate', [8000, 16000, 22050, 32000, 44100, 48000])
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
    st.one_of([st.just(v) for v in [8000, 22050, 32000, 44100, 48000]]),
    # Speed perturbation values
    st.one_of([st.just(v) for v in [0.9, 0.95, 1.05, 1.1]]),
    # Volume perturbation values
    st.one_of([st.just(v) for v in [0.125, 0.5, 1.5, 2.]]),
    # Resampling first?
    st.booleans(),
    # Cut duration (full recording has 16.04s)
    st.floats(min_value=1.1, max_value=12.037575757575)
)
def test_augmentation_chain_randomized(
        target_sampling_rate: int,
        sp_factor: float,
        vp_factor: float,
        resample_first: bool,
        cut_duration: Seconds
):
    recording = Recording.from_file('test/fixtures/libri/libri-1088-134315-0000.wav')

    if resample_first:
        recording_aug = recording.resample(target_sampling_rate).perturb_speed(sp_factor).perturb_volume(vp_factor)
    else:
        recording_aug = recording.perturb_speed(sp_factor).resample(target_sampling_rate).perturb_volume(vp_factor)

    audio_aug = recording_aug.load_audio()
    assert audio_aug.shape[1] == recording_aug.num_samples

    cut_aug = MonoCut(id='dummy', start=0.5125, duration=cut_duration, channel=0, recording=recording_aug)
    assert cut_aug.load_audio().shape[1] == cut_aug.num_samples
