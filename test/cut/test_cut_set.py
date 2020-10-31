from tempfile import NamedTemporaryFile

import pytest

from lhotse import SupervisionSegment, Features, Recording
from lhotse.audio import AudioSource
from lhotse.cut import Cut, CutSet, MixedCut, MixTrack
from lhotse.test_utils import remove_spaces_from_segment_text


@pytest.fixture
def cut_set_with_mixed_cut(cut1, cut2):
    mixed_cut = MixedCut(id='mixed-cut-id', tracks=[
        MixTrack(cut=cut1),
        MixTrack(cut=cut2, offset=1.0, snr=10)
    ])
    return CutSet({cut.id: cut for cut in [cut1, cut2, mixed_cut]})


def test_cut_set_iteration(cut_set_with_mixed_cut):
    cuts = list(cut_set_with_mixed_cut)
    assert len(cut_set_with_mixed_cut) == 3
    assert len(cuts) == 3


def test_cut_set_holds_both_simple_and_mixed_cuts(cut_set_with_mixed_cut):
    simple_cuts = cut_set_with_mixed_cut.simple_cuts.values()
    assert all(isinstance(c, Cut) for c in simple_cuts)
    assert len(simple_cuts) == 2
    mixed_cuts = cut_set_with_mixed_cut.mixed_cuts.values()
    assert all(isinstance(c, MixedCut) for c in mixed_cuts)
    assert len(mixed_cuts) == 1


@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('yaml', False),
        ('yaml', True),
        ('json', False),
        ('json', True),
    ]
)
def test_simple_cut_set_serialization(cut_set, format, compressed):
    with NamedTemporaryFile(suffix='.gz' if compressed else '') as f:
        if format == 'yaml':
            cut_set.to_yaml(f.name)
            restored = CutSet.from_yaml(f.name)
        if format == 'json':
            cut_set.to_json(f.name)
            restored = CutSet.from_json(f.name)
    assert cut_set == restored


@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('yaml', False),
        ('yaml', True),
        ('json', False),
        ('json', True),
    ]
)
def test_mixed_cut_set_serialization(cut_set_with_mixed_cut, format, compressed):
    with NamedTemporaryFile(suffix='.gz' if compressed else '') as f:
        if format == 'yaml':
            cut_set_with_mixed_cut.to_yaml(f.name)
            restored = CutSet.from_yaml(f.name)
        if format == 'json':
            cut_set_with_mixed_cut.to_json(f.name)
            restored = CutSet.from_json(f.name)
    assert cut_set_with_mixed_cut == restored


def test_filter_cut_set(cut_set, cut1):
    filtered = cut_set.filter(lambda cut: cut.id == 'cut-1')
    assert len(filtered) == 1
    assert list(filtered)[0] == cut1


def test_trim_to_unsupervised_segments():
    cut_set = CutSet.from_cuts([
        # Yields 3 unsupervised cuts - before first supervision,
        # between sup2 and sup3, and after sup3.
        Cut('cut1', start=0, duration=30, channel=0, supervisions=[
            SupervisionSegment('sup1', 'rec1', start=1.5, duration=8.5),
            SupervisionSegment('sup2', 'rec1', start=10, duration=5),
            SupervisionSegment('sup3', 'rec1', start=20, duration=8),
        ]),
        # Does not yield any "unsupervised" cut.
        Cut('cut2', start=0, duration=30, channel=0, supervisions=[
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
    return Cut('cut', 0, 10, 0,
               features=Features(type='fbank', num_frames=1000, num_features=40, sampling_rate=8000,
                                 storage_type='lilcom_files', storage_path='storage_dir', storage_key='feats.llc',
                                 start=0,
                                 duration=10),
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
        Cut('cut1', start=0, duration=30, channel=0, recording=recording),
        Cut('cut2', start=0, duration=30, channel=1, recording=recording)
    ])

    mixed = cut_set.mix_same_recording_channels()
    assert len(mixed) == 1

    cut = mixed[0]
    assert isinstance(cut, MixedCut)
    assert len(cut.tracks) == 2
    assert cut.tracks[0].cut == cut_set[0]
    assert cut.tracks[1].cut == cut_set[1]
