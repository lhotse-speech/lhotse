import random
from math import isclose

import pytest

from lhotse import RecordingSet
from lhotse.cut import CutSet, MixedCut, MixTrack, MonoCut, PaddingCut
from lhotse.features import Features
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.testing.dummies import DummyManifest, as_lazy, dummy_cut, dummy_recording
from lhotse.testing.random import deterministic_rng


@pytest.fixture
def overlapping_supervisions_cut():
    return MonoCut(
        id="cut-1",
        start=0.0,
        duration=0.5,
        channel=0,
        features=Features(
            recording_id="recording-1",
            channels=0,
            start=0,
            duration=0.5,
            type="fbank",
            num_frames=50,
            num_features=80,
            frame_shift=0.01,
            sampling_rate=16000,
            storage_type="lilcom",
            storage_path="test/fixtures/dummy_feats/storage/",
            storage_key="e66b6386-aee5-4a5a-8369-fdde1d2b97c7.llc",
        ),
        supervisions=[
            SupervisionSegment(
                id="s1", recording_id="recording-1", start=0.0, duration=0.2
            ),
            SupervisionSegment(
                id="s2", recording_id="recording-1", start=0.1, duration=0.2
            ),
            SupervisionSegment(
                id="s3", recording_id="recording-1", start=0.2, duration=0.2
            ),
            SupervisionSegment(
                id="s4", recording_id="recording-1", start=0.3, duration=0.2
            ),
        ],
    )


@pytest.mark.parametrize(
    [
        "offset",
        "duration",
        "keep_excessive_supervisions",
        "expected_end",
        "expected_supervision_ids",
    ],
    [
        (0.0, None, True, 0.5, ["s1", "s2", "s3", "s4"]),
        (0.0, None, False, 0.5, ["s1", "s2", "s3", "s4"]),
        (0.0, 0.5, True, 0.5, ["s1", "s2", "s3", "s4"]),
        (0.0, 0.5, False, 0.5, ["s1", "s2", "s3", "s4"]),
        (0.1, None, True, 0.5, ["s1", "s2", "s3", "s4"]),
        (0.1, None, False, 0.5, ["s2", "s3", "s4"]),
        (0.0, 0.4, True, 0.4, ["s1", "s2", "s3", "s4"]),
        (0.0, 0.4, False, 0.4, ["s1", "s2", "s3"]),
        (0.1, 0.3, True, 0.4, ["s1", "s2", "s3", "s4"]),
        (0.1, 0.3, False, 0.4, ["s2", "s3"]),
        (0.1, 0.1, True, 0.2, ["s1", "s2"]),
        (0.1, 0.1, False, 0.2, []),
        (0.2, None, True, 0.5, ["s2", "s3", "s4"]),
        (0.2, None, False, 0.5, ["s3", "s4"]),
        (0.2, 0.2, True, 0.4, ["s2", "s3", "s4"]),
        (0.2, 0.2, False, 0.4, ["s3"]),
        (0.0, 0.1, True, 0.1, ["s1"]),
        (0.0, 0.1, False, 0.1, []),
        (0.1, 0.1, False, 0.2, []),
        (0.2, 0.1, False, 0.3, []),
        (0.3, 0.1, False, 0.4, []),
        (0.4, 0.1, False, 0.5, []),
        (0.27, 0.04, False, 0.31, []),
    ],
)
def test_truncate_cut(
    offset,
    duration,
    keep_excessive_supervisions,
    expected_end,
    expected_supervision_ids,
    overlapping_supervisions_cut,
):
    truncated_cut = overlapping_supervisions_cut.truncate(
        offset=offset,
        duration=duration,
        keep_excessive_supervisions=keep_excessive_supervisions,
    )
    remaining_supervision_ids = [s.id for s in truncated_cut.supervisions]
    assert remaining_supervision_ids == expected_supervision_ids
    assert truncated_cut.duration == duration or duration is None
    assert isclose(truncated_cut.end, expected_end)


def test_truncate_above_duration_has_no_effect(overlapping_supervisions_cut):
    truncated_cut = overlapping_supervisions_cut.truncate(
        duration=1.0, preserve_id=True
    )
    assert truncated_cut == overlapping_supervisions_cut


def test_cut_truncate_offset_with_nonzero_start():
    cut = dummy_cut(0, start=1.0, duration=4.0)
    left = cut.truncate(duration=2.0)
    assert left.start == pytest.approx(1.0)
    assert left.end == pytest.approx(3.0)

    right = cut.truncate(offset=2.0)
    assert right.start == pytest.approx(3.0)
    assert right.end == pytest.approx(5.0)


def test_cut_split():
    cut = dummy_cut(0, start=1.0, duration=4.0)
    left, right = cut.split(2.0)
    assert left.start == pytest.approx(1.0)
    assert left.end == pytest.approx(3.0)
    assert right.start == pytest.approx(3.0)
    assert right.end == pytest.approx(5.0)


@pytest.fixture
def simple_mixed_cut():
    return MixedCut(
        id="simple-mixed-cut",
        tracks=[
            MixTrack(cut=dummy_cut(0, duration=10.0)),
            MixTrack(cut=dummy_cut(1, duration=10.0), offset=5.0),
        ],
    )


@pytest.fixture
def gapped_mixed_cut():
    return MixedCut(
        id="gapped-mixed-cut",
        tracks=[
            MixTrack(cut=dummy_cut(0, duration=10.0)),
            MixTrack(cut=dummy_cut(1, duration=10.0).pad(12.0).pad(15.0), offset=15.0),
        ],
    )


def test_truncate_mixed_cut_without_args(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate()
    assert truncated_cut.duration == 15.0


def test_truncate_mixed_cut_with_small_offset(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(offset=1.0)

    assert len(truncated_cut.tracks) == 2

    assert truncated_cut.tracks[0].offset == 0.0
    assert truncated_cut.tracks[0].cut.start == 1.0
    assert truncated_cut.tracks[0].cut.duration == 9.0
    assert truncated_cut.tracks[0].cut.end == 10.0

    assert truncated_cut.tracks[1].offset == 4.0
    assert truncated_cut.tracks[1].cut.start == 0.0
    assert truncated_cut.tracks[1].cut.duration == 10.0
    assert truncated_cut.tracks[1].cut.end == 10.0

    assert truncated_cut.duration == 14.0


def test_truncate_mixed_cut_with_offset_exceeding_first_track(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(offset=11.0)
    assert isinstance(truncated_cut, MonoCut)
    assert truncated_cut.start == 6.0
    assert truncated_cut.duration == 4.0
    assert truncated_cut.end == 10.0


def test_truncate_mixed_cut_decreased_duration(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(duration=14.0)

    assert len(truncated_cut.tracks) == 2

    assert truncated_cut.tracks[0].offset == 0.0
    assert truncated_cut.tracks[0].cut.start == 0.0
    assert truncated_cut.tracks[0].cut.duration == 10.0
    assert truncated_cut.tracks[0].cut.end == 10.0

    assert truncated_cut.tracks[1].offset == 5.0
    assert truncated_cut.tracks[1].cut.start == 0.0
    assert truncated_cut.tracks[1].cut.duration == 9.0
    assert truncated_cut.tracks[1].cut.end == 9.0

    assert truncated_cut.duration == 14.0


def test_truncate_mixed_cut_decreased_duration_removing_last_cut(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(duration=4.0)
    assert isinstance(truncated_cut, MonoCut)
    assert truncated_cut.start == 0.0
    assert truncated_cut.duration == 4.0
    assert truncated_cut.end == 4.0


def test_truncate_mixed_cut_with_small_offset_and_duration(simple_mixed_cut):
    truncated_cut = simple_mixed_cut.truncate(offset=1.0, duration=13.0)

    assert len(truncated_cut.tracks) == 2

    assert truncated_cut.tracks[0].offset == 0.0
    assert truncated_cut.tracks[0].cut.start == 1.0
    assert truncated_cut.tracks[0].cut.duration == 9.0
    assert truncated_cut.tracks[0].cut.end == 10.0

    assert truncated_cut.tracks[1].offset == 4.0
    assert truncated_cut.tracks[1].cut.start == 0.0
    assert truncated_cut.tracks[1].cut.duration == 9.0
    assert truncated_cut.tracks[1].cut.end == 9.0

    assert truncated_cut.duration == 13.0


@pytest.mark.parametrize("offset", [11.0, 26.0])
def test_truncate_mixed_cut_gap_or_padding(gapped_mixed_cut, offset):
    print(gapped_mixed_cut.duration)
    truncated_cut = gapped_mixed_cut.truncate(offset=offset, duration=3.0)
    assert isinstance(truncated_cut, PaddingCut)
    assert truncated_cut.start == 0.0
    assert truncated_cut.duration == 3.0
    assert truncated_cut.end == 3.0
    audio = truncated_cut.load_audio()
    assert audio is not None


def test_truncate_cut_set_lazy_result(cut_set):
    with as_lazy(cut_set, ".jsonl") as lazy_cuts:
        truncated_cut_set = lazy_cuts.truncate(max_duration=5, offset_type="start")
        assert truncated_cut_set.is_lazy
        assert all(c.duration == pytest.approx(5.0) for c in truncated_cut_set)


def test_truncate_cut_set_offset_start(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type="start")
    cut1, cut2 = truncated_cut_set
    assert isclose(cut1.start, 0.0)
    assert isclose(cut1.end, 5.0)
    assert isclose(cut1.duration, 5.0)
    assert isclose(cut2.start, 180.0)
    assert isclose(cut2.end, 185.0)
    assert isclose(cut2.duration, 5.0)


def test_truncate_cut_set_offset_end(cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type="end")
    cut1, cut2 = truncated_cut_set
    assert isclose(cut1.start, 5.0)
    assert isclose(cut1.end, 10.0)
    assert isclose(cut1.duration, 5.0)
    assert isclose(cut2.start, 185.0)
    assert isclose(cut2.end, 190.0)
    assert isclose(cut2.duration, 5.0)


def test_truncate_cut_set_offset_random(deterministic_rng, cut_set):
    truncated_cut_set = cut_set.truncate(max_duration=5, offset_type="random")
    cut1, cut2 = truncated_cut_set
    assert 0.0 <= cut1.start <= 5.0
    assert 5.0 <= cut1.end <= 10.0
    assert isclose(cut1.duration, 5.0)
    assert 180.0 <= cut2.start <= 185.0
    assert 185.0 <= cut2.end <= 190.0
    assert isclose(cut2.duration, 5.0)
    # Check that start and end is not the same in every cut
    assert len(set(cut.start for cut in truncated_cut_set)) > 1
    assert len(set(cut.end for cut in truncated_cut_set)) > 1


@pytest.mark.parametrize("use_rng", [False, True])
def test_truncate_cut_set_offset_random_rng(deterministic_rng, use_rng):
    cuts1 = DummyManifest(CutSet, begin_id=0, end_id=30)
    cuts2 = DummyManifest(CutSet, begin_id=0, end_id=30)

    rng = None
    state = None
    if use_rng:
        rng = random.Random(10)
        state = rng.getstate()

    trunc1 = cuts1.truncate(
        max_duration=0.3, offset_type="random", rng=rng, preserve_id=True
    )

    if use_rng:
        rng.setstate(state)

    trunc2 = cuts2.truncate(
        max_duration=0.3, offset_type="random", rng=rng, preserve_id=True
    )

    # IDs were not changed or changed identically.
    assert list(trunc1.ids) == list(trunc2.ids)

    # Offsets are identical with the use of RNG + state set/reset.
    offsets1 = [c.start for c in trunc1]
    offsets2 = [c.start for c in trunc2]
    if use_rng:
        assert offsets1 == offsets2
    else:
        assert offsets1 != offsets2


@pytest.mark.parametrize("num_jobs", [1, 2])
def test_cut_set_windows_even_split_keep_supervisions(cut_set, num_jobs):
    windows_cut_set = cut_set.cut_into_windows(
        duration=5.0, num_jobs=num_jobs
    ).to_eager()
    assert len(windows_cut_set) == 4
    assert all(cut.duration == 5.0 for cut in windows_cut_set)

    cut1, cut2, cut3, cut4 = windows_cut_set

    assert len(cut1.supervisions) == 1
    assert cut1.supervisions[0].start == 0.5
    assert cut1.supervisions[0].duration == 6.0

    assert len(cut2.supervisions) == 2
    assert cut2.supervisions[0].start == -4.5
    assert cut2.supervisions[0].duration == 6.0
    assert cut2.supervisions[1].start == 2.0
    assert cut2.supervisions[1].duration == 2.0

    assert len(cut3.supervisions) == 1
    assert cut3.supervisions[0].start == 3.0
    assert cut3.supervisions[0].duration == 2.5

    assert len(cut4.supervisions) == 1
    assert cut4.supervisions[0].start == -2.0
    assert cut4.supervisions[0].duration == 2.5


def test_known_issue_with_overlap():
    r = dummy_recording(0)
    rec = RecordingSet.from_recordings([r])

    # Make two segments. The first segment is 1s long. The segment segment
    # is 0.3 seconds long and lies entirely within the first. Both have the
    # same recording_id as the single entry in rec.
    sup = SupervisionSet.from_segments(
        [
            SupervisionSegment(
                id="utt1",
                recording_id=r.id,
                start=0.0,
                duration=1.0,
                channel=0,
                text="Hello",
            ),
            SupervisionSegment(
                id="utt2",
                recording_id=r.id,
                start=0.2,
                duration=0.5,
                channel=0,
                text="World",
            ),
        ]
    )

    cuts = CutSet.from_manifests(recordings=rec, supervisions=sup)
    assert len(cuts) == 1

    cuts_trim = cuts.trim_to_supervisions(keep_overlapping=False).to_eager()
    assert len(cuts_trim) == 2

    cut = cuts_trim[0]
    assert cut.start == 0
    assert cut.duration == 1
    assert len(cut.supervisions) == 1
    sup = cut.supervisions[0]
    assert sup.start == 0
    assert sup.duration == 1
    assert sup.text == "Hello"

    cut = cuts_trim[1]
    assert cut.start == 0.2
    assert cut.duration == 0.5
    assert len(cut.supervisions) == 1
    sup = cut.supervisions[0]
    assert sup.start == 0
    assert sup.duration == 0.5
    assert sup.text == "World"
