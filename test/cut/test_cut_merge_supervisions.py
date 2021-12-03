from lhotse import CutSet
from lhotse.cut import PaddingCut
from lhotse.testing.dummies import DummyManifest, dummy_cut, dummy_supervision


def test_mono_cut_merge_supervisions():
    cut = dummy_cut(
        0,
        duration=10,
        supervisions=[
            dummy_supervision(0, start=1, duration=2),
            dummy_supervision(1, start=5, duration=3),
        ],
    )
    assert len(cut.supervisions) == 2

    mcut = cut.merge_supervisions()

    # original not modified
    assert len(cut.supervisions) == 2
    assert len(mcut.supervisions) == 1

    s = mcut.supervisions[0]
    assert s.id == "cat#dummy-segment-0000#dummy-segment-0001"
    assert s.recording_id == "dummy-recording-0000"  # not changed
    assert s.recording_id == cut.supervisions[0].recording_id
    assert s.start == 1
    assert s.end == 8
    assert s.duration == 7
    assert s.channel == 0
    assert s.text == "irrelevant irrelevant"
    assert s.language == "cat#irrelevant#irrelevant"
    assert s.speaker == "cat#irrelevant#irrelevant"
    assert s.gender == "cat#irrelevant#irrelevant"
    assert s.custom is not None
    assert s.custom["custom_field"] == "cat#irrelevant#irrelevant"


def test_mono_cut_merge_supervisions_identity():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0)])
    mcut = cut.merge_supervisions()
    assert cut == mcut


def test_padding_cut_merge_supervisions():
    cut = PaddingCut('x', 1, 16000, 0)
    mcut = cut.merge_supervisions()
    assert cut == mcut


def test_mixed_cut_merge_supervisions():
    cut0 = dummy_cut(0, supervisions=[dummy_supervision(0)])
    cut1 = dummy_cut(1, supervisions=[dummy_supervision(1)])
    # overlapping supervisions -- note that we don't do anything smart for them.
    mixed = cut0.mix(cut1, offset_other_by=0.5)
    assert len(mixed.supervisions) == 2

    mcut = mixed.merge_supervisions()

    # original not modified
    assert len(mixed.supervisions) == 2
    assert len(mcut.supervisions) == 1

    s = mcut.supervisions[0]
    assert s.id == "cat#dummy-segment-0000#dummy-segment-0001"
    assert s.recording_id == "cat#dummy-recording-0000#dummy-recording-0001"
    assert s.start == 0
    assert s.end == 1.5
    assert s.duration == 1.5
    assert s.channel == -1
    assert s.text == "irrelevant irrelevant"
    assert s.language == "cat#irrelevant#irrelevant"
    assert s.speaker == "cat#irrelevant#irrelevant"
    assert s.gender == "cat#irrelevant#irrelevant"
    assert s.custom is not None
    assert s.custom["custom_field"] == "cat#irrelevant#irrelevant"


def test_mixed_cut_merge_supervisions_identity():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0)])
    cut = cut.append(cut.drop_supervisions())
    mcut = cut.merge_supervisions()
    assert cut == mcut


def test_cut_set_merge_supervisions():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2)
    mcuts = cuts.merge_supervisions()
    assert cuts == mcuts
