import pickle
from tempfile import NamedTemporaryFile, TemporaryDirectory
import numpy as np

import pytest

from lhotse import Features, Recording, SupervisionSegment
from lhotse.audio import AudioSource
from lhotse.cut import MonoCut, CutSet, MixTrack, MixedCut
from lhotse.testing.dummies import remove_spaces_from_segment_text
from lhotse.utils import is_module_available


@pytest.fixture
def cut_set_with_mixed_cut(cut1, cut2):
    mixed_cut = MixedCut(id='mixed-cut-id', tracks=[
        MixTrack(cut=cut1),
        MixTrack(cut=cut2, offset=1.0, snr=10)
    ])
    return CutSet({cut.id: cut for cut in [cut1, cut2, mixed_cut]})


@pytest.mark.parametrize(
    ['ascending', 'expected'],
    [
        (False, [11.0, 10.0, 10.0]),
        (True, [10.0, 10.0, 11.0]),
    ]
)
def test_cut_set_sort_by_duration(cut_set_with_mixed_cut, ascending, expected):
    cs = cut_set_with_mixed_cut.sort_by_duration(ascending=ascending)
    assert [c.duration for c in cs] == expected


def test_cut_set_iteration(cut_set_with_mixed_cut):
    cuts = list(cut_set_with_mixed_cut)
    assert len(cut_set_with_mixed_cut) == 3
    assert len(cuts) == 3


def test_cut_set_holds_both_simple_and_mixed_cuts(cut_set_with_mixed_cut):
    simple_cuts = cut_set_with_mixed_cut.simple_cuts.values()
    assert all(isinstance(c, MonoCut) for c in simple_cuts)
    assert len(simple_cuts) == 2
    mixed_cuts = cut_set_with_mixed_cut.mixed_cuts.values()
    assert all(isinstance(c, MixedCut) for c in mixed_cuts)
    assert len(mixed_cuts) == 1


def test_filter_cut_set(cut_set, cut1):
    filtered = cut_set.filter(lambda cut: cut.id == 'cut-1')
    assert len(filtered) == 1
    assert list(filtered)[0] == cut1


def test_trim_to_unsupervised_segments():
    cut_set = CutSet.from_cuts([
        # Yields 3 unsupervised cuts - before first supervision,
        # between sup2 and sup3, and after sup3.
        MonoCut('cut1', start=0, duration=30, channel=0, supervisions=[
            SupervisionSegment('sup1', 'rec1', start=1.5, duration=8.5),
            SupervisionSegment('sup2', 'rec1', start=10, duration=5),
            SupervisionSegment('sup3', 'rec1', start=20, duration=8),
        ]),
        # Does not yield any "unsupervised" cut.
        MonoCut('cut2', start=0, duration=30, channel=0, supervisions=[
            SupervisionSegment('sup4', 'rec1', start=0, duration=30),
        ]),
    ])
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


def test_trim_to_supervisions_simple_cuts():
    cut_set = CutSet.from_cuts([
        MonoCut('cut1', start=0, duration=30, channel=0, supervisions=[
            SupervisionSegment('sup1', 'rec1', start=1.5, duration=8.5),
            SupervisionSegment('sup2', 'rec1', start=10, duration=5),
            SupervisionSegment('sup3', 'rec1', start=20, duration=8),
        ]),
        MonoCut('cut2', start=0, duration=30, channel=0, supervisions=[
            SupervisionSegment('sup4', 'rec1', start=0, duration=30),
        ]),
    ])
    cuts = cut_set.trim_to_supervisions()
    assert len(cuts) == 4
    assert all(len(cut.supervisions) == 1 for cut in cuts)
    assert all(cut.supervisions[0].start == 0 for cut in cuts)
    cut = cuts[0]
    assert cut.start == 1.5
    assert cut.duration == 8.5
    assert cut.supervisions[0].id == 'sup1'
    cut = cuts[1]
    assert cut.start == 10
    assert cut.duration == 5
    assert cut.supervisions[0].id == 'sup2'
    cut = cuts[2]
    assert cut.start == 20
    assert cut.duration == 8
    assert cut.supervisions[0].id == 'sup3'
    cut = cuts[3]
    assert cut.start == 0
    assert cut.duration == 30
    assert cut.supervisions[0].id == 'sup4'


def test_trim_to_supervisions_mixed_cuts():
    cut_set = CutSet.from_cuts([
        MonoCut('cut1', start=0, duration=30, channel=0,
                recording=Recording(
                    id='rec1', sources=[], sampling_rate=16000, num_samples=160000, duration=10.0
                ),
                supervisions=[
                    SupervisionSegment('sup1', 'rec1', start=1.5, duration=8.5),
                    SupervisionSegment('sup2', 'rec1', start=10, duration=5),
                    SupervisionSegment('sup3', 'rec1', start=20, duration=8),
                ]
                ).append(
            MonoCut('cut2', start=0, duration=30, channel=0,
                    recording=Recording(
                        id='rec1', sources=[], sampling_rate=16000, num_samples=160000, duration=10.0
                    ),
                    supervisions=[
                        SupervisionSegment('sup4', 'rec1', start=0, duration=30),
                    ]
                    )
        )
    ])
    assert isinstance(cut_set[0], MixedCut)
    cuts = cut_set.trim_to_supervisions()
    assert len(cuts) == 4
    # After "trimming", the MixedCut "decayed" into simple, unmixed cuts, as they did not overlap
    assert all(isinstance(cut, MonoCut) for cut in cuts)
    assert all(len(cut.supervisions) == 1 for cut in cuts)
    assert all(cut.supervisions[0].start == 0 for cut in cuts)
    cut = cuts[0]
    # Check that the cuts preserved their start/duration/supervisions after trimming
    assert cut.start == 1.5
    assert cut.duration == 8.5
    assert cut.supervisions[0].id == 'sup1'
    cut = cuts[1]
    assert cut.start == 10
    assert cut.duration == 5
    assert cut.supervisions[0].id == 'sup2'
    cut = cuts[2]
    assert cut.start == 20
    assert cut.duration == 8
    assert cut.supervisions[0].id == 'sup3'
    cut = cuts[3]
    assert cut.start == 0
    assert cut.duration == 30
    assert cut.supervisions[0].id == 'sup4'


@pytest.mark.skipif(not is_module_available('pandas'), reason='Requires pandas to be installed.')
def test_cut_set_describe_runs(cut_set):
    cut_set.describe()


def test_cut_map_supervisions(cut_set):
    for cut in cut_set.map_supervisions(remove_spaces_from_segment_text):
        for s in cut.supervisions:
            if s.text is not None:
                assert ' ' not in s.text


def test_supervision_transform_text(cut_set):
    for cut in cut_set.transform_text(lambda text: 'dummy'):
        for s in cut.supervisions:
            if s.text is not None:
                assert s.text == 'dummy'


@pytest.fixture
def cut_with_relative_paths():
    return MonoCut('cut', 0, 10, 0,
                   features=Features(type='fbank', num_frames=1000, num_features=40, sampling_rate=8000,
                                     storage_type='lilcom_files', storage_path='storage_dir', storage_key='feats.llc',
                                     start=0,
                                     duration=10, frame_shift=0.01),
                   recording=Recording('rec', [AudioSource('file', [0], 'audio.wav')], 8000, 80000, 10.0)
                   )


def test_cut_set_prefix(cut_with_relative_paths):
    cut_set = CutSet.from_cuts([cut_with_relative_paths])
    for c in cut_set.with_recording_path_prefix('/data'):
        assert c.recording.sources[0].source == '/data/audio.wav'
    for c in cut_set.with_features_path_prefix('/data'):
        assert c.features.storage_path == '/data/storage_dir'


def test_mixed_cut_set_prefix(cut_with_relative_paths):
    cut_set = CutSet.from_cuts([cut_with_relative_paths.mix(cut_with_relative_paths)])
    for c in cut_set.with_recording_path_prefix('/data'):
        for t in c.tracks:
            assert t.cut.recording.sources[0].source == '/data/audio.wav'
    for c in cut_set.with_features_path_prefix('/data'):
        for t in c.tracks:
            assert t.cut.features.storage_path == '/data/storage_dir'


def test_mix_same_recording_channels():
    recording = Recording('rec', sampling_rate=8000, num_samples=30 * 8000, duration=30, sources=[
        AudioSource('file', channels=[0], source='irrelevant1.wav'),
        AudioSource('file', channels=[1], source='irrelevant2.wav')
    ])
    cut_set = CutSet.from_cuts([
        MonoCut('cut1', start=0, duration=30, channel=0, recording=recording),
        MonoCut('cut2', start=0, duration=30, channel=1, recording=recording)
    ])

    mixed = cut_set.mix_same_recording_channels()
    assert len(mixed) == 1

    cut = mixed[0]
    assert isinstance(cut, MixedCut)
    assert len(cut.tracks) == 2
    assert cut.tracks[0].cut == cut_set[0]
    assert cut.tracks[1].cut == cut_set[1]


def test_cut_set_filter_supervisions(cut_set):
    def get_supervision_ids(cutset):
        ids = []
        for cut in cutset:
            ids.extend([supervision.id for supervision in cut.supervisions])
        return ids

    all_ids = get_supervision_ids(cut_set)
    train_ids = all_ids[:-1]
    test_ids = all_ids[-1:]

    # filter based on sueprvision ids
    train_set = cut_set.filter_supervisions(lambda s: s.id in train_ids)
    test_set = cut_set.filter_supervisions(lambda s: s.id in test_ids)

    assert get_supervision_ids(train_set) == train_ids
    assert get_supervision_ids(test_set) == test_ids


def test_compute_cmvn_stats():
    cut_set = CutSet.from_json('test/fixtures/libri/cuts.json')
    with NamedTemporaryFile() as f:
        stats = cut_set.compute_global_feature_stats(storage_path=f.name)
        f.flush()
        read_stats = pickle.load(f)
    assert stats['norm_means'].shape == (cut_set[0].num_features,)
    assert stats['norm_stds'].shape == (cut_set[0].num_features,)
    assert (stats['norm_means'] == read_stats['norm_means']).all()
    assert (stats['norm_stds'] == read_stats['norm_stds']).all()


def test_modify_ids(cut_set_with_mixed_cut):
    cut_set = cut_set_with_mixed_cut.modify_ids(lambda cut_id: f'{cut_id}_suffix')
    for ref_cut, mod_cut in zip(cut_set_with_mixed_cut, cut_set):
        assert mod_cut.id == f'{ref_cut.id}_suffix'


def test_map_cut_set(cut_set_with_mixed_cut):
    cut_set = cut_set_with_mixed_cut.map(lambda cut: cut.pad(duration=1000.0))
    for cut in cut_set:
        assert cut.duration == 1000.0


def test_map_cut_set_rejects_noncut(cut_set_with_mixed_cut):
    with pytest.raises(AssertionError):
        cut_set = cut_set_with_mixed_cut.map(lambda cut: 'not-a-cut')


def test_store_audio(cut_set):
    cut_set = CutSet.from_json('test/fixtures/libri/cuts.json')
    with TemporaryDirectory() as tmpdir:
        stored_cut_set = cut_set.compute_and_store_recordings(tmpdir)
        for cut1, cut2 in zip(cut_set, stored_cut_set):
            samples1 = cut1.load_audio()
            samples2 = cut2.load_audio()
            assert np.array_equal(samples1, samples2)
        assert len(stored_cut_set) == len(cut_set)
