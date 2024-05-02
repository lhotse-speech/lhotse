import importlib
from pathlib import Path

import numpy as np
import pytest


def is_torchaudio_available():
    return importlib.util.find_spec("torchaudio") is not None


notorchaudio = pytest.mark.skipif(
    is_torchaudio_available(),
    reason="These are basic tests that check Lhotse is importable "
    "when Torchaudio is not installed and should not be run when Torchaudio is present.",
)


@notorchaudio
def test_lhotse_imports():
    import lhotse


@notorchaudio
def test_lhotse_cutset_works():
    import lhotse

    cuts = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json")
    for _ in cuts:
        pass


@notorchaudio
def test_lhotse_cutset_exportable_to_shar(tmp_path: Path):
    import lhotse

    cuts = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json")
    cuts.to_shar(tmp_path, fields={"recording": "wav"})
    cuts_shar = lhotse.CutSet.from_shar(in_dir=tmp_path)
    for lhs, rhs in zip(cuts, cuts_shar):
        assert lhs.id == rhs.id
        lhsa = lhs.load_audio()
        rhsa = rhs.load_audio()
        np.testing.assert_array_equal(lhsa, rhsa)


@notorchaudio
def test_lhotse_load_audio():
    import lhotse

    cuts = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json")
    cut = cuts[0]
    audio = cut.load_audio()
    assert isinstance(audio, np.ndarray)


@notorchaudio
@pytest.mark.parametrize("sr", [8000, 16000, 22500, 24000, 44100])
def test_lhotse_resample(sr):
    import lhotse

    cuts = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json")
    cut = cuts[0]
    cut = cut.resample(sr)
    audio = cut.load_audio()
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (1, cut.num_samples)


@notorchaudio
def test_lhotse_audio_in_memory():
    import lhotse

    cuts = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json")
    cut = cuts[0]
    cut = cut.move_to_memory()
    audio = cut.load_audio()
    assert isinstance(audio, np.ndarray)


@notorchaudio
def test_lhotse_audio_in_memory_from_wav(tmp_path):
    import soundfile as sf

    import lhotse

    path = str(tmp_path / "test.wav")
    audio = np.random.randint(0, 2**16, size=(16000,)) / 2**16
    sf.write(path, audio, samplerate=16000)

    cut = lhotse.Recording.from_file(path).to_cut()
    cut = cut.truncate(
        duration=cut.duration / 2
    )  # force move_to_memory to go through transcoding
    cut = cut.move_to_memory()
    audio = cut.load_audio()
    assert isinstance(audio, np.ndarray)


@notorchaudio
@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_lhotse_save_audios(tmp_path, fmt):
    import lhotse

    cuts = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json")
    cuts.save_audios(tmp_path, format=fmt)


@notorchaudio
def test_create_dummy_recording():
    from lhotse.testing.dummies import dummy_recording

    recording = dummy_recording(0, with_data=True)
    audio = recording.load_audio()
    assert audio.shape == (1, 16000)


@notorchaudio
def test_create_dummy_multichannel_recording():
    from lhotse.testing.dummies import dummy_multi_channel_recording

    recording = dummy_multi_channel_recording(
        0, channel_ids=[0, 1], with_data=True, source_per_channel=True
    )
    audio = recording.load_audio()
    assert audio.shape == (2, 16000)
