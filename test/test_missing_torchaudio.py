import importlib

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
def test_lhotse_load_audio():
    import lhotse

    cuts = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json")
    cut = cuts[0]
    audio = cut.load_audio()
    assert isinstance(audio, np.ndarray)


@notorchaudio
def test_lhotse_audio_in_memory():
    import lhotse

    cuts = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json")
    cut = cuts[0]
    cut = cut.move_to_memory()
    audio = cut.load_audio()
    assert isinstance(audio, np.ndarray)
