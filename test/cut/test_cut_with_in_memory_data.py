from math import isclose
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from lhotse import CutSet, MonoCut, NumpyFilesWriter, Recording
from lhotse.array import Array
from lhotse.cut import MixedCut, PaddingCut
from lhotse.testing.dummies import dummy_cut
from lhotse.utils import compute_num_frames
from lhotse.utils import nullcontext as does_not_raise


def test_features_move_to_memory():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    feats = cut.features
    assert feats is not None

    arr = feats.load()

    feats_mem = feats.move_to_memory()

    arr_mem = feats_mem.load()

    np.testing.assert_equal(arr, arr_mem)


def test_cut_with_features_move_to_memory():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]

    arr = cut.load_features()
    assert arr is not None

    cut_mem = cut.move_to_memory()
    arr_mem = cut_mem.load_features()

    np.testing.assert_almost_equal(arr, arr_mem, decimal=1)


def test_cut_move_to_memory_load_features_false():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    assert cut.has_features

    cut_mem = cut.move_to_memory(load_features=False)

    assert cut.features == cut_mem.features  # nothing was copied


def test_cut_move_to_memory_load_audio_false():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    assert cut.has_recording

    cut_mem = cut.move_to_memory(load_audio=False)

    assert cut.recording == cut_mem.recording  # nothing was copied


def test_cut_move_to_memory_load_custom_false():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    cut.custom_array = Array("irrelevant", "irrelevant", "irrelevant", [10])

    cut_mem = cut.move_to_memory(load_custom=False)

    assert cut.custom_array == cut_mem.custom_array  # nothing was copied


def test_cut_with_audio_move_to_memory():
    path = "test/fixtures/mono_c0.wav"
    cut = dummy_cut(0, duration=0.5).drop_recording()
    cut.recording = Recording.from_file(path)

    memory_cut = cut.move_to_memory()

    ref = cut.load_audio()
    hyp = memory_cut.load_audio()
    np.testing.assert_equal(hyp, ref)


def test_cut_with_audio_move_to_memory_large_offset():
    path = "test/fixtures/mono_c0.wav"
    cut = dummy_cut(0, duration=0.1).drop_recording()
    cut.recording = Recording.from_file(path)
    cut.start = 0.4
    assert isclose(cut.end, 0.5)

    memory_cut = cut.move_to_memory()

    np.testing.assert_equal(memory_cut.load_audio(), cut.load_audio())


def test_cut_with_array_move_to_memory():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as w:
        arr = np.array([0, 1, 2, 3])
        cut.custom_array = w.store_array(key="dummy-key", value=arr)

        cut_mem = cut.move_to_memory()
        arr_mem = cut_mem.load_custom_array()

        assert arr.dtype == arr_mem.dtype
        np.testing.assert_equal(arr, arr_mem)


def test_cut_with_temporal_array_move_to_memory():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as w:
        arr = np.array(
            np.arange(
                compute_num_frames(cut.duration, frame_shift=0.01, sampling_rate=16000)
            )
        )
        cut.custom_array = w.store_array(
            key="dummy-key", value=arr, frame_shift=0.01, temporal_dim=0, start=0
        )

        cut_mem = cut.move_to_memory()
        arr_mem = cut_mem.load_custom_array()

        assert arr.dtype == arr_mem.dtype
        np.testing.assert_equal(arr, arr_mem)

        arr_trunc = cut.truncate(duration=0.5).load_custom_array()
        arr_mem_trunc = cut_mem.truncate(duration=0.5).load_custom_array()

        assert arr_trunc.dtype == arr_mem_trunc.dtype
        np.testing.assert_equal(arr_trunc, arr_mem_trunc)


def test_cut_with_temporal_array_move_to_memory_large_offset():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    cut.start = 10.0
    cut.duration = 1.5

    with TemporaryDirectory() as d, NumpyFilesWriter(d) as w:
        arr = np.array(
            np.arange(
                compute_num_frames(cut.duration, frame_shift=0.01, sampling_rate=16000)
            )
        )
        cut.custom_array = w.store_array(
            key="dummy-key",
            value=arr,
            frame_shift=0.01,
            temporal_dim=0,
            start=cut.start,
        )

        cut_mem = cut.move_to_memory()
        arr_mem = cut_mem.load_custom_array()

        assert arr.dtype == arr_mem.dtype
        np.testing.assert_equal(arr, arr_mem)

        arr_trunc = cut.truncate(duration=0.5).load_custom_array()
        arr_mem_trunc = cut_mem.truncate(duration=0.5).load_custom_array()

        assert arr_trunc.dtype == arr_mem_trunc.dtype
        np.testing.assert_equal(arr_trunc, arr_mem_trunc)


def test_cut_move_to_memory_audio_serialization():
    path = "test/fixtures/mono_c0.wav"
    cut = dummy_cut(0, duration=0.5).drop_recording()
    cut.recording = Recording.from_file(path)

    cut_with_audio = cut.move_to_memory()

    data = cut_with_audio.to_dict()
    cut_deserialized = MonoCut.from_dict(data)

    np.testing.assert_equal(cut_deserialized.load_audio(), cut_with_audio.load_audio())


def test_padding_cut_move_to_memory():
    cut = PaddingCut(
        "dummy", duration=10.0, sampling_rate=16000, feat_value=-23, num_samples=160000
    )
    cut_mem = cut.move_to_memory()
    assert isinstance(cut_mem, PaddingCut)


def test_mixed_cut_move_to_memory():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    cut = cut.pad(duration=cut.duration + 2.0).append(cut)

    cut_mem = cut.move_to_memory(audio_format="wav")
    assert isinstance(cut_mem, MixedCut)

    audio = cut.load_audio()
    audio_mem = cut_mem.load_audio()
    np.testing.assert_almost_equal(audio, audio_mem, decimal=1)

    feats = cut.load_features()
    feats_mem = cut_mem.load_features()
    np.testing.assert_almost_equal(feats, feats_mem, decimal=1)


def test_mixed_cut_to_mono():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    cut = cut.pad(duration=cut.duration + 2.0).append(cut)

    cut_mem = cut.to_mono("wav")
    assert isinstance(cut_mem, MonoCut)
    assert not cut_mem.has_features

    audio = cut.load_audio()
    audio_mem = cut_mem.load_audio()
    np.testing.assert_almost_equal(audio, audio_mem, decimal=1)


def test_mixed_cut_to_mono_with_custom():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    cut.custom_str = "custom_str"
    cut = cut.pad(duration=cut.duration + 2.0).append(cut)

    cut_mem = cut.to_mono("wav")
    assert isinstance(cut_mem, MonoCut)
    assert not cut_mem.has_features
    assert cut_mem.custom is not None
    assert "custom_str" in cut_mem.custom
    assert cut_mem.custom_str == "custom_str"

    audio = cut.load_audio()
    audio_mem = cut_mem.load_audio()
    np.testing.assert_almost_equal(audio, audio_mem, decimal=1)


def test_drop_in_memory_data():
    cut = dummy_cut(0, with_data=True)

    # Assertions about test data (not the actual test)
    assert cut.is_in_memory
    expected_keys = {
        "recording",
        "features",
        "custom_recording",
        "custom_features",
        "custom_indexes",
        "custom_embedding",
    }
    observed_keys = set()
    for k, v in cut.iter_data():
        observed_keys.add(k)
        if k == "features":
            assert not v.is_in_memory
        else:
            assert v.is_in_memory
    assert expected_keys == observed_keys

    # The actual test
    cut_nomem = cut.drop_in_memory_data()
    assert not cut_nomem.is_in_memory
    observed_keys = set()
    for k, v in cut_nomem.iter_data():
        observed_keys.add(k)
        assert not v.is_in_memory
        if k == "recording":
            with pytest.raises(Exception):
                cut_nomem.load_audio()
        elif k == "features":
            with does_not_raise():
                cut_nomem.load_features()
        else:
            with pytest.raises(Exception):
                cut_nomem.load_custom(k)
    assert expected_keys == observed_keys


def test_drop_in_memory_data_mixed():
    cut = dummy_cut(0, with_data=True)
    cut = cut.pad(duration=cut.duration + 2.0)

    # Assertions about test data (not the actual test)
    assert cut.is_in_memory
    expected_keys = {
        "recording",
        "features",
        "custom_recording",
        "custom_features",
        "custom_indexes",
        "custom_embedding",
    }
    observed_keys = set()
    for k, v in cut.iter_data():
        observed_keys.add(k)
        if k == "features":
            assert not v.is_in_memory
        else:
            assert v.is_in_memory
    assert expected_keys == observed_keys

    # The actual test
    cut_nomem = cut.drop_in_memory_data()
    assert not cut_nomem.is_in_memory
    observed_keys = set()
    for k, v in cut_nomem.iter_data():
        observed_keys.add(k)
        assert not v.is_in_memory
        if k == "recording":
            with pytest.raises(Exception):
                cut_nomem.load_audio()
        elif k == "features":
            with does_not_raise():
                cut_nomem.load_features()
        else:
            with pytest.raises(Exception):
                cut_nomem.load_custom(k)
    assert expected_keys == observed_keys
