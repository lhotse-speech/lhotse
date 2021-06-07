import pytest
from tempfile import NamedTemporaryFile
import numpy as np

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.cut import Cut, CutSet
from lhotse.features import FeatureSet, Features
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.testing.dummies import dummy_cut, dummy_supervision


@pytest.fixture()
def libri_recording_set():
    return RecordingSet.from_json('test/fixtures/libri/audio.json')


@pytest.fixture
def libri_cut_set():
    return CutSet.from_json('test/fixtures/libri/cuts.json')


@pytest.fixture
def libri_features_set():
    return FeatureSet.from_json('test/fixtures/libri/feature_manifest.json.gz')


@pytest.fixture
def supervision_set():
    return SupervisionSet.from_json('test/fixtures/supervision.yml')


@pytest.fixture
def libri_cut(libri_cut_set) -> Cut:
    return libri_cut_set['e3e70682-c209-4cac-629f-6fbed82c07cd']


def test_load_none_feats_cut_set():
    cutset = CutSet.from_json('test/fixtures/libri/cuts_no_feats.json')
    cut = list(cutset.cuts.values())[0]
    assert cut.features is None
    assert cut.recording is not None


def test_load_none_recording_cut_set():
    cutset = CutSet.from_json('test/fixtures/libri/cuts_no_recording.json')
    cut = list(cutset.cuts.values())[0]
    assert cut.recording is None
    assert cut.features is not None


def test_load_audio(libri_cut):
    samples = libri_cut.load_audio()
    assert samples.shape[0] == 1  # single channel
    assert samples.shape[1] == 10 * 16000  # samples count = duration * sampling_rate


def test_load_none_audio(libri_cut):
    libri_cut.recording = None
    samples = libri_cut.load_audio()
    assert samples is None


def test_num_frames(libri_cut):
    expected_features_frame_count = round(16.04 / 0.01)  # duration / frame_shift
    assert libri_cut.features.num_frames == expected_features_frame_count

    expected_cut_frame_count = round(10 / 0.01)  # duration / frame_shift
    assert libri_cut.num_frames == expected_cut_frame_count


def test_load_features(libri_cut):
    feats = libri_cut.load_features()
    assert feats.shape[0] == libri_cut.num_frames
    assert feats.shape[1] == libri_cut.features.num_features


def test_load_none_features(libri_cut):
    libri_cut.features = None
    feats = libri_cut.load_features()
    assert feats is None


def test_store_audio(libri_cut):
    with NamedTemporaryFile() as f:
        stored_cut = libri_cut.compute_and_store_recording(f.name)
        samples1 = libri_cut.load_audio()
        rec = Recording.from_file(f.name)
        samples2 = rec.load_audio()
        assert np.array_equal(samples1, samples2)
        assert rec.duration == libri_cut.duration
        assert rec.duration == stored_cut.duration
        assert libri_cut.duration == stored_cut.duration


@pytest.fixture
def dummy_recording_set():
    return RecordingSet.from_recordings([
        Recording(
            id='rec1', sampling_rate=16000, num_samples=160000, duration=10, sources=[
                AudioSource(type='file', channels=[0], source='dummy.wav')
            ]
        )
    ])


@pytest.fixture
def dummy_supervision_set():
    return SupervisionSet.from_segments([
        SupervisionSegment(
            id='sup1', recording_id='rec1', start=3, duration=4, channel=0, text='dummy text'
        )
    ])


@pytest.fixture
def dummy_feature_set():
    return FeatureSet.from_features([
        Features(
            recording_id='rec1', channels=0, start=0, duration=10, type='fbank', num_frames=1000,
            num_features=23, sampling_rate=16000, storage_type='lilcom_files', storage_path='feats',
            storage_key='dummy.llc', frame_shift=0.01
        )
    ])


def test_make_cuts_from_recordings(dummy_recording_set):
    cut_set = CutSet.from_manifests(recordings=dummy_recording_set)
    cut1 = cut_set[0]
    assert cut1.start == 0
    assert cut1.duration == 10.0
    assert cut1.end == 10.0
    assert cut1.channel == 0

    assert len(cut1.supervisions) == 0

    assert cut1.has_recording
    assert cut1.recording == dummy_recording_set.recordings['rec1']
    assert cut1.sampling_rate == 16000
    assert cut1.recording_id == 'rec1'
    assert cut1.num_samples == 160000

    assert not cut1.has_features
    assert cut1.features is None
    assert cut1.frame_shift is None
    assert cut1.num_frames is None
    assert cut1.num_features is None
    assert cut1.features_type is None


def test_make_cuts_from_features(dummy_feature_set):
    cut_set = CutSet.from_manifests(features=dummy_feature_set)
    cut1 = cut_set[0]
    assert cut1.start == 0
    assert cut1.duration == 10.0
    assert cut1.end == 10.0
    assert cut1.channel == 0

    assert len(cut1.supervisions) == 0

    assert not cut1.has_recording
    assert cut1.recording is None
    assert cut1.sampling_rate == 16000
    assert cut1.recording_id == 'rec1'
    assert cut1.num_samples is None

    assert cut1.has_features
    assert cut1.features == dummy_feature_set.features[0]
    assert cut1.frame_shift == 0.01
    assert cut1.num_frames == 1000
    assert cut1.num_features == 23
    assert cut1.features_type == 'fbank'


def test_make_cuts_from_features_recordings(dummy_recording_set, dummy_feature_set):
    cut_set = CutSet.from_manifests(recordings=dummy_recording_set, features=dummy_feature_set)
    cut1 = cut_set[0]
    assert cut1.start == 0
    assert cut1.duration == 10.0
    assert cut1.end == 10.0
    assert cut1.channel == 0

    assert len(cut1.supervisions) == 0

    assert cut1.has_recording
    assert cut1.recording == dummy_recording_set.recordings['rec1']
    assert cut1.sampling_rate == 16000
    assert cut1.recording_id == 'rec1'
    assert cut1.num_samples == 160000

    assert cut1.has_features
    assert cut1.features == dummy_feature_set.features[0]
    assert cut1.frame_shift == 0.01
    assert cut1.num_frames == 1000
    assert cut1.num_features == 23
    assert cut1.features_type == 'fbank'


def test_make_cuts_from_recordings_with_deterministic_ids(dummy_recording_set):
    cut_set = CutSet.from_manifests(recordings=dummy_recording_set, random_ids=False)
    for idx, cut in enumerate(cut_set):
        assert cut.id == f'{cut.recording_id}-{idx}-{cut.channel}'


def test_make_cuts_from_recordings_with_random_ids(dummy_recording_set):
    cut_set = CutSet.from_manifests(recordings=dummy_recording_set, random_ids=True)
    for idx, cut in enumerate(cut_set):
        assert cut.id != f'{cut.recording_id}-{idx}-{cut.channel}'


def test_make_cuts_from_features_with_deterministic_ids(dummy_feature_set):
    cut_set = CutSet.from_manifests(features=dummy_feature_set, random_ids=False)
    for idx, cut in enumerate(cut_set):
        assert cut.id == f'{cut.recording_id}-{idx}-{cut.channel}'


def test_make_cuts_from_features_with_random_ids(dummy_feature_set):
    cut_set = CutSet.from_manifests(features=dummy_feature_set, random_ids=True)
    for idx, cut in enumerate(cut_set):
        assert cut.id != f'{cut.recording_id}-{idx}-{cut.channel}'


class TestCutOnSupervisions:
    def test_make_cuts_from_recordings_supervisions(self, dummy_recording_set, dummy_supervision_set):
        cut_set = CutSet.from_manifests(recordings=dummy_recording_set,
                                        supervisions=dummy_supervision_set).trim_to_supervisions()
        cut1 = cut_set[0]
        assert cut1.start == 3.0
        assert cut1.duration == 4.0
        assert cut1.end == 7.0
        assert cut1.channel == 0

        assert len(cut1.supervisions) == 1
        assert cut1.supervisions[0].id == 'sup1'
        assert cut1.supervisions[0].recording_id == 'rec1'
        assert cut1.supervisions[0].start == 0.0
        assert cut1.supervisions[0].duration == 4.0
        assert cut1.supervisions[0].end == 4.0
        assert cut1.supervisions[0].channel == 0
        assert cut1.supervisions[0].text == 'dummy text'

        assert cut1.has_recording
        assert cut1.recording == dummy_recording_set.recordings['rec1']
        assert cut1.sampling_rate == 16000
        assert cut1.recording_id == 'rec1'
        assert cut1.num_samples == 16000 * 4

        assert not cut1.has_features
        assert cut1.features is None
        assert cut1.frame_shift is None
        assert cut1.num_frames is None
        assert cut1.num_features is None
        assert cut1.features_type is None

    def test_make_cuts_from_features_supervisions(self, dummy_feature_set, dummy_supervision_set):
        cut_set = CutSet.from_manifests(supervisions=dummy_supervision_set,
                                        features=dummy_feature_set).trim_to_supervisions()
        cut1 = cut_set[0]
        assert cut1.start == 3.0
        assert cut1.duration == 4.0
        assert cut1.end == 7.0
        assert cut1.channel == 0

        assert len(cut1.supervisions) == 1
        assert cut1.supervisions[0].id == 'sup1'
        assert cut1.supervisions[0].recording_id == 'rec1'
        assert cut1.supervisions[0].start == 0.0
        assert cut1.supervisions[0].duration == 4.0
        assert cut1.supervisions[0].end == 4.0
        assert cut1.supervisions[0].channel == 0
        assert cut1.supervisions[0].text == 'dummy text'

        assert not cut1.has_recording
        assert cut1.recording is None
        assert cut1.sampling_rate == 16000
        assert cut1.recording_id == 'rec1'
        assert cut1.num_samples is None

        assert cut1.has_features
        assert cut1.features == dummy_feature_set.features[0]
        assert cut1.frame_shift == 0.01
        assert cut1.num_frames == 400
        assert cut1.num_features == 23
        assert cut1.features_type == 'fbank'

    def test_make_cuts_from_recordings_features_supervisions(
            self,
            dummy_recording_set,
            dummy_feature_set,
            dummy_supervision_set
    ):
        cut_set = CutSet.from_manifests(recordings=dummy_recording_set, supervisions=dummy_supervision_set,
                                        features=dummy_feature_set).trim_to_supervisions()
        cut1 = cut_set[0]
        assert cut1.start == 3.0
        assert cut1.duration == 4.0
        assert cut1.end == 7.0
        assert cut1.channel == 0

        assert len(cut1.supervisions) == 1
        assert cut1.supervisions[0].id == 'sup1'
        assert cut1.supervisions[0].recording_id == 'rec1'
        assert cut1.supervisions[0].start == 0.0
        assert cut1.supervisions[0].duration == 4.0
        assert cut1.supervisions[0].end == 4.0
        assert cut1.supervisions[0].channel == 0
        assert cut1.supervisions[0].text == 'dummy text'

        assert cut1.has_recording
        assert cut1.recording == dummy_recording_set.recordings['rec1']
        assert cut1.sampling_rate == 16000
        assert cut1.recording_id == 'rec1'
        assert cut1.num_samples == 16000 * 4

        assert cut1.has_features
        assert cut1.features == dummy_feature_set.features[0]
        assert cut1.frame_shift == 0.01
        assert cut1.num_frames == 400
        assert cut1.num_features == 23
        assert cut1.features_type == 'fbank'


class TestNoCutOnSupervisions:
    def test_make_cuts_from_recordings_supervisions(self, dummy_recording_set, dummy_supervision_set):
        cut_set = CutSet.from_manifests(recordings=dummy_recording_set, supervisions=dummy_supervision_set)
        cut1 = cut_set[0]
        assert cut1.start == 0
        assert cut1.duration == 10.0
        assert cut1.end == 10.0
        assert cut1.channel == 0

        assert len(cut1.supervisions) == 1
        assert cut1.supervisions[0].id == 'sup1'
        assert cut1.supervisions[0].recording_id == 'rec1'
        assert cut1.supervisions[0].start == 3.0
        assert cut1.supervisions[0].end == 7.0
        assert cut1.supervisions[0].channel == 0
        assert cut1.supervisions[0].text == 'dummy text'

        assert cut1.has_recording
        assert cut1.recording == dummy_recording_set.recordings['rec1']
        assert cut1.sampling_rate == 16000
        assert cut1.recording_id == 'rec1'
        assert cut1.num_samples == 160000

        assert not cut1.has_features
        assert cut1.features is None
        assert cut1.frame_shift is None
        assert cut1.num_frames is None
        assert cut1.num_features is None
        assert cut1.features_type is None

    def test_make_cuts_from_features_supervisions(self, dummy_feature_set, dummy_supervision_set):
        cut_set = CutSet.from_manifests(supervisions=dummy_supervision_set, features=dummy_feature_set)
        cut1 = cut_set[0]
        assert cut1.start == 0
        assert cut1.duration == 10.0
        assert cut1.end == 10.0
        assert cut1.channel == 0

        assert len(cut1.supervisions) == 1
        assert cut1.supervisions[0].id == 'sup1'
        assert cut1.supervisions[0].recording_id == 'rec1'
        assert cut1.supervisions[0].start == 3.0
        assert cut1.supervisions[0].end == 7.0
        assert cut1.supervisions[0].channel == 0
        assert cut1.supervisions[0].text == 'dummy text'

        assert not cut1.has_recording
        assert cut1.recording is None
        assert cut1.sampling_rate == 16000
        assert cut1.recording_id == 'rec1'
        assert cut1.num_samples is None

        assert cut1.has_features
        assert cut1.features == dummy_feature_set.features[0]
        assert cut1.frame_shift == 0.01
        assert cut1.num_frames == 1000
        assert cut1.num_features == 23
        assert cut1.features_type == 'fbank'

    def test_make_cuts_from_recordings_features_supervisions(
            self,
            dummy_recording_set,
            dummy_feature_set,
            dummy_supervision_set
    ):
        cut_set = CutSet.from_manifests(recordings=dummy_recording_set, supervisions=dummy_supervision_set,
                                        features=dummy_feature_set)
        cut1 = cut_set[0]
        assert cut1.start == 0
        assert cut1.duration == 10.0
        assert cut1.end == 10.0
        assert cut1.channel == 0

        assert len(cut1.supervisions) == 1
        assert cut1.supervisions[0].id == 'sup1'
        assert cut1.supervisions[0].recording_id == 'rec1'
        assert cut1.supervisions[0].start == 3.0
        assert cut1.supervisions[0].end == 7.0
        assert cut1.supervisions[0].channel == 0
        assert cut1.supervisions[0].text == 'dummy text'

        assert cut1.has_recording
        assert cut1.recording == dummy_recording_set.recordings['rec1']
        assert cut1.sampling_rate == 16000
        assert cut1.recording_id == 'rec1'
        assert cut1.num_samples == 160000

        assert cut1.has_features
        assert cut1.features == dummy_feature_set.features[0]
        assert cut1.frame_shift == 0.01
        assert cut1.num_frames == 1000
        assert cut1.num_features == 23
        assert cut1.features_type == 'fbank'


@pytest.fixture()
def dummy_cut_with_supervisions():
    return dummy_cut(
        unique_id=0,
        supervisions=[
            dummy_supervision(unique_id=i, duration=i) for i in range(1, 7)
        ]
    )


def test_cut_filter_supervisions(dummy_cut_with_supervisions):
    cut = dummy_cut_with_supervisions

    # test id filtering
    cut_first_three = cut.filter_supervisions(lambda s: s.id.endswith(("1", "2", "3")))
    cut_last_three = cut.filter_supervisions(lambda s: s.id.endswith(("4", "5", "6")))

    assert not set(s.id for s in cut_first_three.supervisions) & set(s.id for s in cut_last_three.supervisions)
    assert (set(s.id for s in cut_first_three.supervisions) | set(s.id for s in cut_last_three.supervisions)) \
           == set(s.id for s in cut.supervisions)

    # test duration filtering
    cut_first_three = cut.filter_supervisions(lambda s: s.duration <= 3)
    cut_last_three = cut.filter_supervisions(lambda s: s.duration > 3)

    assert not set(s.id for s in cut_first_three.supervisions) & set(s.id for s in cut_last_three.supervisions)
    assert (set(s.id for s in cut_first_three.supervisions) | set(s.id for s in cut_last_three.supervisions)) \
           == set(s.id for s in cut.supervisions)
