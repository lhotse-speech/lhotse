from math import isclose
from tempfile import NamedTemporaryFile

import numpy as np

from lhotse import CutSet, NumpyHdf5Writer, Recording, MonoCut
from lhotse.array import Array
from lhotse.testing.dummies import dummy_cut
from lhotse.utils import compute_num_frames


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

    np.testing.assert_almost_equal(arr, arr_mem, decimal=2)


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

    np.testing.assert_equal(memory_cut.load_audio(), cut.load_audio())


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
    with NamedTemporaryFile(suffix=".h5") as f, NumpyHdf5Writer(f.name) as w:
        arr = np.array([0, 1, 2, 3])
        cut.custom_array = w.store_array(key="dummy-key", value=arr)

        cut_mem = cut.move_to_memory()
        arr_mem = cut_mem.load_custom_array()

        assert arr.dtype == arr_mem.dtype
        np.testing.assert_equal(arr, arr_mem)


def test_cut_with_temporal_array_move_to_memory():
    path = "test/fixtures/libri/cuts.json"
    cut = CutSet.from_file(path)[0]
    with NamedTemporaryFile(suffix=".h5") as f, NumpyHdf5Writer(f.name) as w:
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

    with NamedTemporaryFile(suffix=".h5") as f, NumpyHdf5Writer(f.name) as w:
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

    assert cut.custom is None  # original cut is unmodified

    data = cut_with_audio.to_dict()
    cut_deserialized = MonoCut.from_dict(data)

    np.testing.assert_equal(cut_deserialized.load_audio(), cut_with_audio.load_audio())
