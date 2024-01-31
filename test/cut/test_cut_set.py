import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest

from lhotse import (
    AudioSource,
    Fbank,
    FbankConfig,
    Features,
    FeatureSet,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    load_manifest,
)
from lhotse.cut import CutSet, MixedCut, MixTrack, MonoCut, MultiCut
from lhotse.cut.describe import CutSetStatistics
from lhotse.serialization import load_jsonl
from lhotse.testing.dummies import (
    DummyManifest,
    as_lazy,
    dummy_cut,
    dummy_recording,
    dummy_supervision,
    remove_spaces_from_segment_text,
)
from lhotse.utils import is_module_available


@pytest.fixture
def mini_librispeeh2_cut_set():
    recordings = RecordingSet.from_file(
        "test/fixtures/mini_librispeech2/lhotse/recordings.jsonl.gz"
    )
    supervisions = SupervisionSet.from_file(
        "test/fixtures/mini_librispeech2/lhotse/supervisions.jsonl.gz"
    )
    return CutSet.from_manifests(recordings=recordings, supervisions=supervisions)


@pytest.fixture
def cut_set_with_mixed_cut(cut1, cut2):
    mixed_cut = MixedCut(
        id="mixed-cut-id",
        tracks=[MixTrack(cut=cut1), MixTrack(cut=cut2, offset=1.0, snr=10)],
    )
    return CutSet([cut1, cut2, mixed_cut])


@pytest.mark.parametrize(
    ["ascending", "expected"],
    [
        (False, [11.0, 10.0, 10.0]),
        (True, [10.0, 10.0, 11.0]),
    ],
)
def test_cut_set_sort_by_duration(cut_set_with_mixed_cut, ascending, expected):
    cs = cut_set_with_mixed_cut.sort_by_duration(ascending=ascending)
    assert [c.duration for c in cs] == expected


@pytest.mark.parametrize(
    ["ascending", "expected"],
    [
        (True, ["lbi-3536-23268-0000", "lbi-6241-61943-0000", "lbi-8842-304647-0000"]),
        (False, ["lbi-8842-304647-0000", "lbi-6241-61943-0000", "lbi-3536-23268-0000"]),
    ],
)
def test_cut_set_sort_by_recording_id(mini_librispeeh2_cut_set, ascending, expected):
    cs = mini_librispeeh2_cut_set.sort_by_recording_id(ascending)
    assert [c.recording.id for c in cs] == expected


def test_cut_set_iteration(cut_set_with_mixed_cut):
    cuts = list(cut_set_with_mixed_cut)
    assert len(cut_set_with_mixed_cut) == 3
    assert len(cuts) == 3


def test_cut_set_holds_both_simple_and_mixed_cuts(cut_set_with_mixed_cut):
    simple_cuts = cut_set_with_mixed_cut.simple_cuts
    assert all(isinstance(c, MonoCut) for c in simple_cuts)
    assert len(simple_cuts) == 2
    mixed_cuts = cut_set_with_mixed_cut.mixed_cuts
    assert all(isinstance(c, MixedCut) for c in mixed_cuts)
    assert len(mixed_cuts) == 1


def test_filter_cut_set(cut_set, cut1):
    filtered = cut_set.filter(lambda cut: cut.id == "cut-1")
    assert len(filtered) == 1
    assert list(filtered)[0] == cut1


def test_trim_to_unsupervised_segments():
    cut_set = CutSet.from_cuts(
        [
            # Yields 3 unsupervised cuts - before first supervision,
            # between sup2 and sup3, and after sup3.
            MonoCut(
                "cut1",
                start=0,
                duration=30,
                channel=0,
                supervisions=[
                    SupervisionSegment("sup1", "rec1", start=1.5, duration=8.5),
                    SupervisionSegment("sup2", "rec1", start=10, duration=5),
                    SupervisionSegment("sup3", "rec1", start=20, duration=8),
                ],
                recording=dummy_recording(1, duration=30),
            ),
            # Does not yield any "unsupervised" cut.
            MonoCut(
                "cut2",
                start=0,
                duration=30,
                channel=0,
                supervisions=[
                    SupervisionSegment("sup4", "rec1", start=0, duration=30),
                ],
                recording=dummy_recording(2, duration=30),
            ),
        ]
    )
    unsupervised_cuts = cut_set.trim_to_unsupervised_segments()

    assert len(unsupervised_cuts) == 3

    assert unsupervised_cuts[0].start == 0
    assert unsupervised_cuts[0].duration == 1.5
    assert unsupervised_cuts[0].supervisions == []

    assert unsupervised_cuts[1].start == 15
    assert unsupervised_cuts[1].duration == 5
    assert unsupervised_cuts[1].supervisions == []

    assert unsupervised_cuts[2].start == 28
    assert unsupervised_cuts[2].duration == 2
    assert unsupervised_cuts[2].supervisions == []


@pytest.mark.parametrize("keep_overlapping", [True, False])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_trim_to_supervisions_simple_cuts(keep_overlapping, num_jobs):
    cut_set = CutSet.from_cuts(
        [
            MonoCut(
                "cut1",
                start=0,
                duration=30,
                channel=0,
                supervisions=[
                    SupervisionSegment("sup1", "rec1", start=1.5, duration=10.5),
                    SupervisionSegment("sup2", "rec1", start=10, duration=5),
                    SupervisionSegment("sup3", "rec1", start=20, duration=8),
                ],
                recording=dummy_recording(1, duration=30),
            ),
            MonoCut(
                "cut2",
                start=0,
                duration=30,
                channel=0,
                supervisions=[
                    SupervisionSegment("sup4", "rec1", start=0, duration=30),
                ],
                recording=dummy_recording(2, duration=30),
            ),
        ]
    )
    cuts = cut_set.trim_to_supervisions(
        keep_overlapping=keep_overlapping, num_jobs=num_jobs
    ).to_eager()
    assert len(cuts) == 4

    # Note: expected results diverge here depending on the value of keep_overlapping flag
    cut = cuts[0]
    assert cut.start == 1.5
    assert cut.duration == 10.5
    if keep_overlapping:
        assert len(cut.supervisions) == 2
        sup = cut.supervisions[0]
        assert sup.id == "sup1"
        assert sup.start == 0
        assert sup.duration == 10.5
        sup = cut.supervisions[1]
        assert sup.id == "sup2"
        assert sup.start == 8.5
        assert sup.duration == 5
    else:
        assert len(cut.supervisions) == 1
        sup = cut.supervisions[0]
        assert sup.id == "sup1"
        assert sup.start == 0
        assert sup.duration == 10.5

    # Note: expected results diverge here depending on the value of keep_overlapping flag
    cut = cuts[1]
    assert cut.start == 10
    assert cut.duration == 5
    if keep_overlapping:
        assert len(cut.supervisions) == 2
        sup = cut.supervisions[0]
        assert sup.id == "sup1"
        assert sup.start == -8.5
        assert sup.duration == 10.5
        sup = cut.supervisions[1]
        assert sup.id == "sup2"
        assert sup.start == 0
        assert sup.duration == 5
    else:
        assert len(cut.supervisions) == 1
        sup = cut.supervisions[0]
        assert sup.id == "sup2"
        assert sup.start == 0
        assert sup.duration == 5

    # Note: both test cases have same results
    cut = cuts[2]
    assert len(cut.supervisions) == 1
    assert cut.start == 20
    assert cut.duration == 8
    assert cut.supervisions[0].id == "sup3"

    # Note: both test cases have same results
    cut = cuts[3]
    assert len(cut.supervisions) == 1
    assert cut.start == 0
    assert cut.duration == 30
    assert cut.supervisions[0].id == "sup4"


@pytest.fixture()
def mixed_overlapping_cut_set():
    """
    Input mixed cut::
        |---------------mixedcut--------------------|
        |--------rec1 0-30s--------|
                     |-------rec2 15-45s--------|
         |---sup1--|         |-----sup3-----|
                 |sup2|
    """
    cut_set = CutSet.from_cuts(
        [
            MonoCut(
                "cut1",
                start=0,
                duration=30,
                channel=0,
                recording=Recording(
                    id="rec1",
                    sources=[],
                    sampling_rate=16000,
                    num_samples=160000,
                    duration=60.0,
                ),
                supervisions=[
                    SupervisionSegment("sup1", "rec1", start=1.5, duration=10.5),
                    SupervisionSegment("sup2", "rec1", start=10, duration=6),
                ],
            ).mix(
                MonoCut(
                    "cut2",
                    start=15,
                    duration=30,
                    channel=0,
                    recording=Recording(
                        id="rec2",
                        sources=[],
                        sampling_rate=16000,
                        num_samples=160000,
                        duration=60.0,
                    ),
                    supervisions=[
                        SupervisionSegment("sup3", "rec2", start=8, duration=18),
                    ],
                ),
                offset_other_by=15.0,
            )
        ]
    )
    assert isinstance(cut_set[0], MixedCut)
    return cut_set


def test_trim_to_supervisions_mixed_cuts_keep_overlapping_false(
    mixed_overlapping_cut_set,
):
    cuts = mixed_overlapping_cut_set.trim_to_supervisions(
        keep_overlapping=False
    ).to_eager()
    assert len(cuts) == 3
    # After "trimming", in some instances the MixedCut "decayed" into simple, unmixed cuts, as they did not overlap;
    # In other instances, it's still a MixedCut
    assert all(len(cut.supervisions) == 1 for cut in cuts)

    cut = cuts[0]
    assert isinstance(cut, MonoCut)
    assert cut.start == 1.5
    assert cut.duration == 10.5
    sup = cut.supervisions[0]
    assert sup.id == "sup1"
    assert sup.start == 0
    assert sup.duration == 10.5

    cut = cuts[1]
    assert isinstance(cut, MixedCut)
    assert cut.start == 0
    assert cut.duration == 6
    sup = cut.supervisions[0]
    assert sup.id == "sup2"
    assert sup.start == 0
    assert sup.duration == 6

    cut = cuts[2]
    assert isinstance(cut, MixedCut)
    assert cut.start == 0
    assert cut.duration == 18
    sup = cut.supervisions[0]
    assert sup.id == "sup3"
    assert sup.start == 0
    assert sup.duration == 18


def test_trim_to_supervisions_mixed_cuts_keep_overlapping_true(
    mixed_overlapping_cut_set,
):
    cuts = mixed_overlapping_cut_set.trim_to_supervisions(
        keep_overlapping=True
    ).to_eager()
    assert len(cuts) == 3
    # After "trimming", in some instances the MixedCut "decayed" into simple, unmixed cuts, as they did not overlap;
    # In other instances, it's still a MixedCut

    cut = cuts[0]
    assert isinstance(cut, MonoCut)
    assert cut.start == 1.5
    assert cut.duration == 10.5
    assert len(cut.supervisions) == 2
    sup = cut.supervisions[0]
    assert sup.id == "sup1"
    assert sup.start == 0
    assert sup.duration == 10.5
    sup = cut.supervisions[1]
    assert sup.id == "sup2"
    assert sup.start == 8.5
    assert sup.duration == 6

    cut = cuts[1]
    assert isinstance(cut, MixedCut)
    assert cut.start == 0
    assert cut.duration == 6
    assert len(cut.supervisions) == 2
    sup = cut.supervisions[0]
    assert sup.id == "sup1"
    assert sup.start == -8.5
    assert sup.duration == 10.5
    sup = cut.supervisions[1]
    assert sup.id == "sup2"
    assert sup.start == 0
    assert sup.duration == 6

    cut = cuts[2]
    assert isinstance(cut, MixedCut)
    assert cut.start == 0
    assert cut.duration == 18
    assert len(cut.supervisions) == 1
    sup = cut.supervisions[0]
    assert sup.id == "sup3"
    assert sup.start == 0
    assert sup.duration == 18


@pytest.mark.parametrize("full", [True, False])
def test_cut_set_describe_runs(cut_set, full, capfd):
    cut_set.describe(full=full)
    out, err = capfd.readouterr()
    assert out != ""
    assert err == ""


@pytest.mark.parametrize("full", [True, False])
def test_cut_set_stats_combine(cut_set, full, capfd):

    # Describe a "large" cut set containing two parts
    cs = cut_set.repeat(2)
    cs.describe(full=full)
    out, err = capfd.readouterr()

    # Describe a combination of stats from two parts of that cut set
    stats1 = CutSetStatistics(full=full).accumulate(cut_set)
    stats2 = CutSetStatistics(full=full).accumulate(cut_set)
    stats = stats1.combine(stats2)
    stats.describe()
    out2, err2 = capfd.readouterr()

    assert out == out2
    assert err == err2


def test_cut_map_supervisions(cut_set):
    for cut in cut_set.map_supervisions(remove_spaces_from_segment_text):
        for s in cut.supervisions:
            if s.text is not None:
                assert " " not in s.text


def test_supervision_transform_text(cut_set):
    for cut in cut_set.transform_text(lambda text: "dummy"):
        for s in cut.supervisions:
            if s.text is not None:
                assert s.text == "dummy"


@pytest.fixture
def cut_with_relative_paths():
    return MonoCut(
        "cut",
        0,
        10,
        0,
        features=Features(
            type="fbank",
            num_frames=1000,
            num_features=40,
            sampling_rate=8000,
            storage_type="lilcom_files",
            storage_path="storage_dir",
            storage_key="feats.llc",
            start=0,
            duration=10,
            frame_shift=0.01,
        ),
        recording=Recording(
            "rec", [AudioSource("file", [0], "audio.wav")], 8000, 80000, 10.0
        ),
    )


def test_cut_set_prefix(cut_with_relative_paths):
    cut_set = CutSet.from_cuts([cut_with_relative_paths])
    for c in cut_set.with_recording_path_prefix("/data"):
        assert c.recording.sources[0].source == "/data/audio.wav"
    for c in cut_set.with_features_path_prefix("/data"):
        assert c.features.storage_path == "/data/storage_dir"


def test_mixed_cut_set_prefix(cut_with_relative_paths):
    cut_set = CutSet.from_cuts([cut_with_relative_paths.mix(cut_with_relative_paths)])
    for c in cut_set.with_recording_path_prefix("/data"):
        for t in c.tracks:
            assert t.cut.recording.sources[0].source == "/data/audio.wav"
    for c in cut_set.with_features_path_prefix("/data"):
        for t in c.tracks:
            assert t.cut.features.storage_path == "/data/storage_dir"


def test_combine_same_recording_channels():
    recording = Recording(
        "rec",
        sampling_rate=8000,
        num_samples=30 * 8000,
        duration=30,
        sources=[
            AudioSource("file", channels=[0], source="irrelevant1.wav"),
            AudioSource("file", channels=[1], source="irrelevant2.wav"),
        ],
    )
    cut_set = CutSet.from_cuts(
        [
            MonoCut("cut1", start=0, duration=30, channel=0, recording=recording),
            MonoCut("cut2", start=0, duration=30, channel=1, recording=recording),
        ]
    )

    multi = cut_set.combine_same_recording_channels()
    assert len(multi) == 1

    cut = multi[0]
    assert isinstance(cut, MultiCut)
    assert cut.num_channels == 2


def test_cut_set_filter_supervisions(cut_set):
    def get_supervision_ids(cutset):
        ids = []
        for cut in cutset:
            ids.extend([supervision.id for supervision in cut.supervisions])
        return ids

    all_ids = get_supervision_ids(cut_set)
    train_ids = all_ids[:-1]
    test_ids = all_ids[-1:]

    # filter based on supervision ids
    train_set = cut_set.filter_supervisions(lambda s: s.id in train_ids)
    test_set = cut_set.filter_supervisions(lambda s: s.id in test_ids)

    assert get_supervision_ids(train_set) == train_ids
    assert get_supervision_ids(test_set) == test_ids


def test_compute_cmvn_stats():
    cut_set = CutSet.from_json("test/fixtures/libri/cuts.json")
    with NamedTemporaryFile() as f:
        stats = cut_set.compute_global_feature_stats(storage_path=f.name)
        f.flush()
        read_stats = pickle.load(f)
    assert stats["norm_means"].shape == (cut_set[0].num_features,)
    assert stats["norm_stds"].shape == (cut_set[0].num_features,)
    assert (stats["norm_means"] == read_stats["norm_means"]).all()
    assert (stats["norm_stds"] == read_stats["norm_stds"]).all()


@pytest.mark.parametrize("max_cuts", [None, 1])
def test_compute_cmvn_stats_on_the_fly(max_cuts):
    cut_set = CutSet.from_json("test/fixtures/libri/cuts.json")
    fbank = Fbank()
    with TemporaryDirectory() as d:
        cut_set = cut_set.compute_and_store_features(fbank, d)
        # precomputed
        precomputed_stats = cut_set.compute_global_feature_stats(max_cuts=max_cuts)
        # on the fly
        on_the_fly_stats = cut_set.compute_global_feature_stats(
            max_cuts=max_cuts, extractor=fbank
        )
    for key in ("norm_means", "norm_stds"):
        np.testing.assert_almost_equal(
            precomputed_stats[key], on_the_fly_stats[key], decimal=3
        )


@pytest.mark.parametrize("nj", [1, 2])
def test_compute_and_store_features_lazy(nj):
    eager_cuts = CutSet.from_json("test/fixtures/libri/cuts.json").repeat(10)
    with as_lazy(eager_cuts) as cut_set:
        fbank = Fbank()
        with TemporaryDirectory() as d:
            with_feats = cut_set.compute_and_store_features(fbank, d, num_jobs=nj)
            assert len(with_feats) == len(cut_set)
            assert set(with_feats.ids) == set(cut_set.ids)
            assert all(c.has_features for c in with_feats)


def test_modify_ids(cut_set_with_mixed_cut):
    cut_set = cut_set_with_mixed_cut.modify_ids(lambda cut_id: f"{cut_id}_suffix")
    for ref_cut, mod_cut in zip(cut_set_with_mixed_cut, cut_set):
        assert mod_cut.id == f"{ref_cut.id}_suffix"


def test_map_cut_set(cut_set_with_mixed_cut):
    cut_set = cut_set_with_mixed_cut.map(lambda cut: cut.pad(duration=1000.0))
    for cut in cut_set:
        assert cut.duration == 1000.0


@pytest.mark.skip(reason="For now, we are avoiding checking this explicitly.")
def test_map_cut_set_rejects_noncut(cut_set_with_mixed_cut):
    with pytest.raises(AssertionError):
        cut_set = cut_set_with_mixed_cut.map(lambda cut: "not-a-cut")


@pytest.mark.parametrize("num_jobs", [1, 2])
def test_store_audio(num_jobs):
    cut_set = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    cut_set = cut_set.sort_by_duration()
    with TemporaryDirectory() as tmpdir:
        for enc, bits in (
            ("PCM_S", 16),
            ("PCM_F", 32),
            (None, 16),
            ("PCM_S", None),
            (None, None),
        ):
            stored_cut_set = cut_set.save_audios(
                tmpdir, encoding=enc, bits_per_sample=bits, num_jobs=num_jobs
            )

            stored_cut_set = stored_cut_set.sort_by_duration()
            for cut1, cut2 in zip(cut_set, stored_cut_set):
                samples1 = cut1.load_audio()
                samples2 = cut2.load_audio()
                assert np.array_equal(samples1, samples2)
            assert len(stored_cut_set) == len(cut_set)

    with TemporaryDirectory() as tmpdir:
        for bits in (16, 24, None):
            stored_cut_set = cut_set.save_audios(
                tmpdir, format="flac", bits_per_sample=bits, num_jobs=num_jobs
            )
            stored_cut_set = stored_cut_set.sort_by_duration()
            for cut1, cut2 in zip(cut_set, stored_cut_set):
                samples1 = cut1.load_audio()
                samples2 = cut2.load_audio()
                assert np.array_equal(samples1, samples2)
            assert len(stored_cut_set) == len(cut_set)


def test_cut_set_subset_cut_ids_preserves_order():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=1000)
    cut_ids = ["dummy-mono-cut-0010", "dummy-mono-cut-0171", "dummy-mono-cut-0009"]
    subcuts = cuts.subset(cut_ids=cut_ids)
    cut1, cut2, cut3 = subcuts
    assert cut1.id == "dummy-mono-cut-0010"
    assert cut2.id == "dummy-mono-cut-0171"
    assert cut3.id == "dummy-mono-cut-0009"


def test_cut_set_subset_cut_ids_preserves_order_with_lazy_manifest():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=1000)
    cut_ids = ["dummy-mono-cut-0010", "dummy-mono-cut-0171", "dummy-mono-cut-0009"]
    with NamedTemporaryFile(suffix=".jsonl.gz") as f:
        cuts.to_file(f.name)
        cuts = cuts.from_jsonl_lazy(f.name)
        subcuts = cuts.subset(cut_ids=cut_ids)
        cut1, cut2, cut3 = subcuts
        assert cut1.id == "dummy-mono-cut-0010"
        assert cut2.id == "dummy-mono-cut-0171"
        assert cut3.id == "dummy-mono-cut-0009"


def test_cut_set_decompose():
    c = dummy_cut(
        0,
        start=5.0,
        duration=10.0,
        supervisions=[dummy_supervision(0, start=0.0), dummy_supervision(1, start=6.5)],
    )
    assert c.start == 5.0
    assert c.end == 15.0
    cuts = CutSet.from_cuts([c])

    recs, sups, feats = cuts.decompose()

    assert isinstance(recs, RecordingSet)
    assert len(recs) == 1
    assert recs[0].id == "dummy-recording-0000"

    assert isinstance(sups, SupervisionSet)
    assert len(sups) == 2
    assert sups[0].id == "dummy-segment-0000"
    assert sups[0].start == 5.0
    assert sups[0].end == 6.0
    assert sups[1].id == "dummy-segment-0001"
    assert sups[1].start == 11.5
    assert sups[1].end == 12.5

    assert isinstance(feats, FeatureSet)
    assert len(feats) == 1


def test_cut_set_decompose_doesnt_duplicate_recording():
    c = dummy_cut(0)
    c2 = dummy_cut(0)
    c2.id = "dummy-cut-0001"  # override cut ID, retain identical recording ID as `c`
    cuts = CutSet.from_cuts([c, c2])

    recs, sups, feats = cuts.decompose()

    assert isinstance(recs, RecordingSet)
    # deduplicated recording
    assert len(recs) == 1
    assert recs[0].id == "dummy-recording-0000"

    assert sups is None

    assert isinstance(feats, FeatureSet)
    # not deduplicated features
    assert len(feats) == 2


def test_cut_set_decompose_output_dir():
    c = dummy_cut(
        0,
        start=5.0,
        duration=10.0,
        supervisions=[dummy_supervision(0, start=0.0), dummy_supervision(1, start=6.5)],
    )
    assert c.start == 5.0
    assert c.end == 15.0
    cuts = CutSet.from_cuts([c])

    with TemporaryDirectory() as td:
        td = Path(td)
        recs, sups, feats = cuts.decompose(output_dir=td)
        assert list(recs) == list(load_manifest(td / "recordings.jsonl.gz"))
        assert list(sups) == list(load_manifest(td / "supervisions.jsonl.gz"))
        assert list(feats) == list(load_manifest(td / "features.jsonl.gz"))


def test_cut_set_decompose_output_dir_doesnt_duplicate_recording():
    c = dummy_cut(0)
    c2 = dummy_cut(0)
    c2.id = "dummy-cut-0001"  # override cut ID, retain identical recording ID as `c`
    cuts = CutSet.from_cuts([c, c2])

    with TemporaryDirectory() as td:
        td = Path(td)
        cuts.decompose(output_dir=td)

        recs = load_manifest(td / "recordings.jsonl.gz")
        assert isinstance(recs, RecordingSet)
        # deduplicated recording
        assert len(recs) == 1
        assert recs[0].id == "dummy-recording-0000"


def test_cut_set_from_files():
    cs1 = DummyManifest(CutSet, begin_id=0, end_id=10)
    cs2 = DummyManifest(CutSet, begin_id=10, end_id=20)
    with NamedTemporaryFile(suffix=".jsonl.gz") as f1, NamedTemporaryFile(
        suffix=".jsonl.gz"
    ) as f2:
        cs1.to_file(f1.name)
        f1.flush()
        cs2.to_file(f2.name)
        f2.flush()

        cs = CutSet.from_files([f1.name, f2.name], shuffle_iters=True, seed=0)
        # __getitem__ with int index iterates lazy manifets
        assert cs[0].id == "dummy-mono-cut-0000"
        # On second iteration, we see a different order
        assert cs[0].id == "dummy-mono-cut-0010"


def test_cut_set_duplicate_ids_allowed():
    cut = dummy_cut(0)
    cuts = CutSet.from_cuts([cut, cut])
    assert len(cuts) == 2
    assert cuts[0].id == cuts[1].id
