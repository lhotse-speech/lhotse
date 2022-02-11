import numpy as np

from lhotse import Recording, MonoCut
from lhotse.testing.dummies import dummy_cut


def test_cut_move_to_memory():
    path = "test/fixtures/mono_c0.wav"
    cut = dummy_cut(0, duration=0.5).drop_recording()
    cut.recording = Recording.from_file(path)

    memory_cut = cut.move_to_memory()

    np.testing.assert_equal(
        memory_cut.load_audio(), cut.load_audio()
    )


def test_cut_move_to_memory_audio_serialization():
    path = "test/fixtures/mono_c0.wav"
    cut = dummy_cut(0, duration=0.5).drop_recording()
    cut.recording = Recording.from_file(path)

    cut_with_audio = cut.move_to_memory()

    assert cut.custom is None  # original cut is unmodified

    data = cut_with_audio.to_dict()
    cut_deserialized = MonoCut.from_dict(data)

    np.testing.assert_equal(
        cut_deserialized.load_audio(), cut_with_audio.load_audio()
    )
