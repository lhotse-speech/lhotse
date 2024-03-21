import pytest

from lhotse.cut import CutSet
from lhotse.dataset import AudioTaggingDataset


@pytest.fixture
def dummy_cut_set():
    cuts = CutSet.from_json("test/fixtures/libri/cuts.json")

    def _add_audio_event(c):
        c.supervisions[0].audio_event = "Speech; Whisper"
        return c

    cuts = cuts.map(_add_audio_event)
    return cuts


def test_audio_tagging_dataset(dummy_cut_set):
    dataset = AudioTaggingDataset()
    out = dataset[dummy_cut_set]
    supervisions = out["supervisions"]
    assert "audio_event" in supervisions
    print("Pass the test")
