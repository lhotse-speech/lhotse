from tempfile import NamedTemporaryFile
import numpy as np
import pytest
from lhotse import Recording, OpenSmileExtractor, OpenSmileConfig
from lhotse.utils import is_module_available, compute_num_frames, compute_num_samples

opensmile = pytest.importorskip(
    "opensmile", reason="Opensmile is an optional dependency."
)


@pytest.fixture()
def recording():
    return Recording.from_file("test/fixtures/libri/libri-1088-134315-0000.wav")


def test_extract_feats_default_config(recording):
    feature_extractor = OpenSmileExtractor()
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert np.shape(y) == (1604, 65)


def test_extract_feats_opensmile_config(recording):
    config = OpenSmileConfig(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        sampling_rate=recording.sampling_rate,
    )
    feature_extractor = OpenSmileExtractor(config=config)
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert np.shape(y) == (1604, 65)


@pytest.mark.parametrize(
    "feature_set,feature_level,expected_shape",
    [
        ("ComParE_2016", "lld", (1604, 65)),
        ("ComParE_2016", "lld_de", (1604, 65)),
        ("ComParE_2016", "func", (1, 6373)),
        # ('GeMAPSv01a','lld',(1600,18)), #deprecated
        # ('GeMAPSv01a','func',(1,62)),  #deprecated
        ("GeMAPSv01b", "lld", (1604, 18)),
        ("GeMAPSv01b", "func", (1, 62)),
        # ('eGeMAPSv01a','lld',(1600,23)), #deprecated
        # ('eGeMAPSv01a','func',(1,88)), #deprecated
        # ('eGeMAPSv01b','lld',(1600,23)), #deprecated
        # ('eGeMAPSv01b','func',(1,88)), #deprecated
        ("eGeMAPSv02", "lld", (1604, 25)),
        ("eGeMAPSv02", "func", (1, 88)),
        ("emobase", "lld", (1604, 26)),
        ("emobase", "lld_de", (1604, 26)),
        ("emobase", "func", (1, 988)),
    ],
)
def test_extract_feats_config_defined_by_string(
    recording, feature_set, feature_level, expected_shape
):
    config = OpenSmileConfig(
        feature_set=feature_set,
        feature_level=feature_level,
        sampling_rate=recording.sampling_rate,
    )
    feature_extractor = OpenSmileExtractor(config=config)
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert np.shape(y) == expected_shape


@pytest.mark.parametrize(
    "feature_set,feature_level",
    [
        ("ComParE_2016", "lld"),
        ("ComParE_2016", "lld_de"),
        # ('ComParE_2016','func'), -> NotImplementedError
        ("GeMAPSv01b", "lld"),
        # ('GeMAPSv01b','func'), -> NotImplementedError
        ("eGeMAPSv02", "lld"),
        # ('eGeMAPSv02','func'), -> NotImplementedError
        ("emobase", "lld"),
        ("emobase", "lld_de"),
        # ('emobase','func'), -> NotImplementedError
    ],
)
def test_num_frames(
    feature_set,
    feature_level,
):
    sr = 8000
    duration = 12.059
    config = OpenSmileConfig(
        feature_set=feature_set,
        feature_level=feature_level,
        sampling_rate=sr,
        resample=True,
    )
    feature_extractor = OpenSmileExtractor(config=config)

    num_frames = compute_num_frames(duration, feature_extractor.frame_shift, sr)
    num_samples = compute_num_samples(duration, sr)

    signal = np.random.rand(1, num_samples)
    y = feature_extractor.extract(signal, sr)
    assert np.shape(y)[0] == num_frames


def test_sampling_rate_mismatch(recording):
    config = OpenSmileConfig(sampling_rate=8000)
    feature_extractor = OpenSmileExtractor(config=config)
    with pytest.raises(ValueError):
        y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
