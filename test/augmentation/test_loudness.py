import numpy as np
import pytest
import soundfile as sf

from lhotse import CutSet, Fbank, LilcomFilesWriter, Recording
from lhotse.augmentation import LoudnessNormalization
from lhotse.augmentation.loudness import normalize_loudness

pyln = pytest.importorskip("pyloudnorm")


@pytest.mark.parametrize("amplitude", [0.0, 1e-6])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("num_channels", [1, 2])
@pytest.mark.parametrize("duration", [0.1, 1.0])
def test_normalize_loudness_below_absolute_gate(
    amplitude, dtype, num_channels, duration
):
    sampling_rate = 16000
    time = np.arange(int(sampling_rate * duration)) / sampling_rate
    audio = np.tile(amplitude * np.sin(2 * np.pi * 1000 * time), (num_channels, 1))
    audio = audio.astype(dtype)
    original = audio.copy()
    target = -20.0

    normalized = normalize_loudness(audio, target, sampling_rate)

    assert np.isfinite(normalized).all()
    assert normalized.shape == audio.shape
    assert normalized.dtype == dtype
    np.testing.assert_array_equal(audio, original)
    # Below the meter's absolute gate, use -70 LUFS to keep the gain finite.
    expected = (original.astype(np.float64) * 10 ** (50 / 20)).astype(dtype)
    np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("num_channels", [1, 5])
def test_normalize_loudness_measurable_audio(dtype, num_channels):
    sampling_rate = 16000
    time = np.arange(sampling_rate) / sampling_rate
    audio = np.tile(0.1 * np.sin(2 * np.pi * 1000 * time), (num_channels, 1))
    audio = audio.astype(dtype)
    # A silent channel should remain silent when another channel is measurable.
    if num_channels > 1:
        audio[0] = 0
    original = audio.copy()
    target = -20.0

    normalized = LoudnessNormalization(target)(audio, sampling_rate)

    assert normalized.dtype == dtype
    np.testing.assert_array_equal(audio, original)
    meter = pyln.Meter(sampling_rate)
    assert meter.integrated_loudness(normalized.T) == pytest.approx(target, abs=1e-5)
    expected = pyln.normalize.loudness(
        audio.T, meter.integrated_loudness(audio.T), target
    ).T.astype(dtype)
    np.testing.assert_array_equal(normalized, expected)


@pytest.mark.parametrize("amplitude", [0.0, 1e-6])
def test_cut_set_normalize_loudness_stores_finite_features(tmp_path, amplitude):
    pytest.importorskip("lilcom")
    sampling_rate = 16000
    time = np.arange(sampling_rate) / sampling_rate
    audio = (amplitude * np.sin(2 * np.pi * 1000 * time)).astype(np.float32)
    path = tmp_path / "quiet.wav"
    sf.write(path, audio, sampling_rate, subtype="FLOAT")
    cuts = CutSet.from_cuts([Recording.from_file(path).to_cut()])
    extractor = Fbank()

    with_features = cuts.normalize_loudness(-20).compute_and_store_features(
        extractor=extractor,
        storage_path=tmp_path / "features",
        storage_type=LilcomFilesWriter,
        num_jobs=1,
        progress_bar=False,
    )

    assert len(with_features) == 1
    cut = with_features[0]
    assert cut.duration == cuts[0].duration
    features = cut.load_features()
    assert np.isfinite(features).all()
    reference_audio = (audio.astype(np.float64) * 10 ** (50 / 20)).astype(np.float32)
    expected = extractor.extract(reference_audio[None], sampling_rate)
    # Lilcom's default quantization step is 2**-5.
    np.testing.assert_allclose(features, expected, rtol=0, atol=2**-5)
