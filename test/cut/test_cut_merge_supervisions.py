import pytest

from lhotse import CutSet
from lhotse.cut import PaddingCut
from lhotse.testing.dummies import (
    DummyManifest,
    dummy_cut,
    dummy_multi_cut,
    dummy_supervision,
)


@pytest.mark.parametrize("merge_policy", ["delimiter", "keep_first"])
def test_mono_cut_merge_supervisions(merge_policy):
    cut = dummy_cut(
        0,
        duration=10,
        supervisions=[
            dummy_supervision(0, start=1, duration=2),
            dummy_supervision(1, start=5, duration=3),
        ],
    )
    assert len(cut.supervisions) == 2

    mcut = cut.merge_supervisions(merge_policy=merge_policy)

    # original not modified
    assert len(cut.supervisions) == 2
    assert len(mcut.supervisions) == 1

    s = mcut.supervisions[0]
    assert s.recording_id == "dummy-recording-0000"  # not changed
    assert s.recording_id == cut.supervisions[0].recording_id
    assert s.start == 1
    assert s.end == 8
    assert s.duration == 7
    assert s.channel == 0
    assert s.text == "irrelevant irrelevant"
    assert s.custom is not None
    if merge_policy == "delimiter":
        assert s.id == "cat#dummy-segment-0000#dummy-segment-0001"
        assert s.language == "cat#irrelevant#irrelevant"
        assert s.speaker == "cat#irrelevant#irrelevant"
        assert s.gender == "cat#irrelevant#irrelevant"
        assert s.custom["custom_field"] == "cat#irrelevant#irrelevant"
    else:
        assert s.id == "dummy-segment-0000"
        assert s.language == "irrelevant"
        assert s.speaker == "irrelevant"
        assert s.gender == "irrelevant"
        assert s.custom["custom_field"] == "irrelevant"


def test_mono_cut_merge_supervisions_identity():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0)])
    mcut = cut.merge_supervisions()
    assert cut == mcut


def test_mono_cut_merge_supervisions_no_supervisions():
    cut = dummy_cut(0, supervisions=[])
    mcut = cut.merge_supervisions()
    assert cut == mcut


def test_mono_cut_merge_supervisions_empty_fields():
    cut = dummy_cut(
        0,
        duration=2,
        supervisions=[dummy_supervision(0), dummy_supervision(1, start=1)],
    )
    # set the fields to None to check if the merged spvn also has None
    cut.supervisions[0].speaker = None
    cut.supervisions[1].speaker = None
    mcut = cut.merge_supervisions()
    assert mcut.supervisions[0].speaker is None


def test_mono_cut_merge_supervisions_custom_merge_fn():
    cut = dummy_cut(
        0,
        duration=2,
        supervisions=[dummy_supervision(0), dummy_supervision(1, start=1)],
    )
    # Note: in tests, by default there exists one custom field called "custom_field"
    #       we add custom field "a" and define a different merging behavior for it.
    cut.supervisions[0].custom["a"] = 20
    cut.supervisions[1].custom["a"] = -13
    mcut = cut.merge_supervisions(
        custom_merge_fn=lambda k, vs: sum(vs) if k == "a" else None
    )
    assert isinstance(mcut.supervisions[0].custom, dict)
    assert mcut.supervisions[0].custom["a"] == 7
    # "dummy_supervision" object has a "custom_field" set by default
    assert mcut.supervisions[0].custom["custom_field"] is None


def test_padding_cut_merge_supervisions():
    cut = PaddingCut("x", 1, 16000, 0)
    mcut = cut.merge_supervisions()
    assert cut == mcut


@pytest.mark.parametrize("merge_policy", ["delimiter", "keep_first"])
def test_mixed_cut_merge_supervisions(merge_policy):
    cut0 = dummy_cut(0, supervisions=[dummy_supervision(0)])
    cut1 = dummy_cut(1, supervisions=[dummy_supervision(1)])
    # overlapping supervisions -- note that we don't do anything smart for them.
    mixed = cut0.mix(cut1, offset_other_by=0.5)
    assert len(mixed.supervisions) == 2

    mcut = mixed.merge_supervisions(merge_policy=merge_policy)

    # original not modified
    assert len(mixed.supervisions) == 2
    assert len(mcut.supervisions) == 1

    s = mcut.supervisions[0]
    assert s.custom is not None
    if merge_policy == "delimiter":
        assert s.id == "cat#dummy-segment-0000#dummy-segment-0001"
        assert s.recording_id == "cat#dummy-recording-0000#dummy-recording-0001"
        assert s.language == "cat#irrelevant#irrelevant"
        assert s.speaker == "cat#irrelevant#irrelevant"
        assert s.gender == "cat#irrelevant#irrelevant"
        assert s.custom["custom_field"] == "cat#irrelevant#irrelevant"
    else:
        assert s.id == "dummy-segment-0000"
        assert s.recording_id == "dummy-recording-0000"
        assert s.language == "irrelevant"
        assert s.speaker == "irrelevant"
        assert s.gender == "irrelevant"
        assert s.custom["custom_field"] == "irrelevant"
    assert s.start == 0
    assert s.end == 1.5
    assert s.duration == 1.5
    assert s.channel == -1
    assert s.text == "irrelevant irrelevant"


def test_mixed_cut_merge_supervisions_identity():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0)])
    cut = cut.append(cut.drop_supervisions())
    mcut = cut.merge_supervisions()
    assert cut == mcut


@pytest.mark.parametrize("merge_policy", ["delimiter", "keep_first"])
def test_multi_cut_merge_supervisions_simple(merge_policy):
    cut = dummy_multi_cut(
        0,
        duration=10,
        supervisions=[
            dummy_supervision(0, start=1, duration=2),
            dummy_supervision(1, start=5, duration=3),
        ],
    )
    assert len(cut.supervisions) == 2

    mcut = cut.merge_supervisions(merge_policy=merge_policy)

    # original not modified
    assert len(cut.supervisions) == 2
    assert len(mcut.supervisions) == 1

    s = mcut.supervisions[0]
    assert s.recording_id == "dummy-recording-0000"  # not changed
    assert s.recording_id == cut.supervisions[0].recording_id
    assert s.start == 1
    assert s.end == 8
    assert s.duration == 7
    assert s.channel == [0]
    assert s.text == "irrelevant irrelevant"
    assert s.custom is not None
    if merge_policy == "delimiter":
        assert s.id == "cat#dummy-segment-0000#dummy-segment-0001"
        assert s.language == "cat#irrelevant#irrelevant"
        assert s.speaker == "cat#irrelevant#irrelevant"
        assert s.gender == "cat#irrelevant#irrelevant"
        assert s.custom["custom_field"] == "cat#irrelevant#irrelevant"
    else:
        assert s.id == "dummy-segment-0000"
        assert s.language == "irrelevant"
        assert s.speaker == "irrelevant"
        assert s.gender == "irrelevant"
        assert s.custom["custom_field"] == "irrelevant"


@pytest.mark.parametrize("merge_channels", [True, False])
def test_multi_cut_merge_supervisions_per_channel(merge_channels):
    cut = dummy_multi_cut(
        0,
        duration=10,
        supervisions=[
            dummy_supervision(0, start=1, duration=2, channel=[0, 1]),
            dummy_supervision(1, start=5, duration=3, channel=[0, 1]),
            dummy_supervision(2, start=2, duration=3, channel=[5, 6]),
        ],
    )
    assert len(cut.supervisions) == 3

    mcut = cut.merge_supervisions(merge_channels=merge_channels)

    # original not modified
    assert len(cut.supervisions) == 3
    assert len(mcut.supervisions) == (1 if merge_channels else 2)

    s = mcut.supervisions[0]
    if merge_channels:
        assert s.id == "cat#dummy-segment-0000#dummy-segment-0002#dummy-segment-0001"
        assert s.recording_id == "dummy-recording-0000"  # not changed
        assert s.recording_id == cut.supervisions[0].recording_id
        assert s.start == 1
        assert s.end == 8
        assert s.duration == 7
        assert s.channel == [0, 1, 5, 6]
        assert s.text == "irrelevant irrelevant irrelevant"
        assert s.language == "cat#irrelevant#irrelevant#irrelevant"
        assert s.speaker == "cat#irrelevant#irrelevant#irrelevant"
        assert s.gender == "cat#irrelevant#irrelevant#irrelevant"
        assert s.custom is not None
        assert s.custom["custom_field"] == "cat#irrelevant#irrelevant#irrelevant"
    else:
        assert s.id == "cat#dummy-segment-0000#dummy-segment-0001"
        assert s.recording_id == "dummy-recording-0000"  # not changed
        assert s.recording_id == cut.supervisions[0].recording_id
        assert s.start == 1
        assert s.end == 8
        assert s.duration == 7
        assert s.channel == [0, 1]
        assert s.text == "irrelevant irrelevant"
        assert s.language == "cat#irrelevant#irrelevant"
        assert s.speaker == "cat#irrelevant#irrelevant"
        assert s.gender == "cat#irrelevant#irrelevant"
        assert s.custom is not None
        assert s.custom["custom_field"] == "cat#irrelevant#irrelevant"


def test_cut_set_merge_supervisions():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2)
    mcuts = cuts.merge_supervisions()
    assert cuts == mcuts
