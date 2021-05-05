import pickle
from tempfile import NamedTemporaryFile

import pytest

from lhotse import AudioSource, Cut, CutSet, FeatureSet, Features, Recording, RecordingSet, SupervisionSegment, \
    SupervisionSet, load_manifest, store_manifest
from lhotse.utils import fastcopy, is_module_available, nullcontext as does_not_raise


@pytest.mark.parametrize(
    ['path', 'exception_expectation'],
    [
        ('test/fixtures/audio.json', does_not_raise()),
        ('test/fixtures/supervision.json', does_not_raise()),
        ('test/fixtures/dummy_feats/feature_manifest.json', does_not_raise()),
        ('test/fixtures/libri/cuts.json', does_not_raise()),
        ('test/fixtures/feature_config.yml', pytest.raises(ValueError)),
        ('no/such/path.xd', pytest.raises(AssertionError)),
    ]
)
def test_load_any_lhotse_manifest(path, exception_expectation):
    with exception_expectation:
        load_manifest(path)


@pytest.fixture
def recording_set():
    return RecordingSet.from_recordings([
        Recording(
            id='x',
            sources=[
                AudioSource(
                    type='file',
                    channels=[0],
                    source='text/fixtures/mono_c0.wav'
                ),
                AudioSource(
                    type='command',
                    channels=[1],
                    source='cat text/fixtures/mono_c1.wav'
                )
            ],
            sampling_rate=8000,
            num_samples=4000,
            duration=0.5
        )
    ])


@pytest.fixture
def supervision_set():
    return SupervisionSet.from_segments([
        SupervisionSegment(
            id='segment-1',
            recording_id='recording-1',
            channel=0,
            start=0.1,
            duration=0.3,
            text='transcript of the first segment',
            language='english',
            speaker='Norman Dyhrentfurth',
            gender='male'
        )
    ])


@pytest.fixture
def feature_set():
    return FeatureSet(
        features=[
            Features(
                recording_id='irrelevant',
                channels=0,
                start=0.0,
                duration=20.0,
                type='fbank',
                num_frames=2000,
                num_features=20,
                frame_shift=0.01,
                sampling_rate=16000,
                storage_type='lilcom',
                storage_path='/irrelevant/',
                storage_key='path.llc'
            )
        ]
    )


@pytest.fixture
def cut_set():
    cut = Cut(
        id='cut-1',
        start=0.0,
        duration=10.0,
        channel=0,
        features=Features(
            type='fbank',
            num_frames=100,
            num_features=40,
            frame_shift=0.01,
            sampling_rate=16000,
            start=0.0,
            duration=10.0,
            storage_type='lilcom',
            storage_path='irrelevant',
            storage_key='irrelevant',
        ),
        recording=Recording(
            id='rec-1',
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
            sources=[
                AudioSource(
                    type='file',
                    channels=[0],
                    source='irrelevant'
                )
            ]
        ),
        supervisions=[
            SupervisionSegment(id='sup-1', recording_id='irrelevant', start=0.5, duration=6.0),
            SupervisionSegment(id='sup-2', recording_id='irrelevant', start=7.0, duration=2.0)
        ])
    return CutSet.from_cuts([
        cut,
        fastcopy(cut, id='cut-nosup', supervisions=[]),
        fastcopy(cut, id='cut-norec', recording=None),
        fastcopy(cut, id='cut-nofeat', features=None),
        cut.pad(duration=30.0, direction='left'),
        cut.pad(duration=30.0, direction='right'),
        cut.pad(duration=30.0, direction='both'),
        cut.mix(cut, offset_other_by=5.0, snr=8)
    ])


@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('yaml', False),
        ('yaml', True),
        ('json', False),
        ('json', True),
        ('jsonl', False),
        ('jsonl', True),
    ]
)
def test_feature_set_serialization(feature_set, format, compressed):
    with NamedTemporaryFile(suffix='.gz' if compressed else '') as f:
        if format == 'jsonl':
            feature_set.to_jsonl(f.name)
            feature_set_deserialized = FeatureSet.from_jsonl(f.name)
        if format == 'json':
            feature_set.to_json(f.name)
            feature_set_deserialized = FeatureSet.from_json(f.name)
        if format == 'yaml':
            feature_set.to_yaml(f.name)
            feature_set_deserialized = FeatureSet.from_yaml(f.name)
    assert feature_set_deserialized == feature_set


@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('yaml', False),
        ('yaml', True),
        ('json', False),
        ('json', True),
        ('jsonl', False),
        ('jsonl', True),
    ]
)
def test_serialization(recording_set, format, compressed):
    with NamedTemporaryFile(suffix='.gz' if compressed else '') as f:
        if format == 'jsonl':
            recording_set.to_jsonl(f.name)
            deserialized = RecordingSet.from_jsonl(f.name)
        if format == 'yaml':
            recording_set.to_yaml(f.name)
            deserialized = RecordingSet.from_yaml(f.name)
        if format == 'json':
            recording_set.to_json(f.name)
            deserialized = RecordingSet.from_json(f.name)
    assert deserialized == recording_set


@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('yaml', False),
        ('yaml', True),
        ('json', False),
        ('json', True),
        ('jsonl', False),
        ('jsonl', True),
    ]
)
def test_supervision_set_serialization(supervision_set, format, compressed):
    with NamedTemporaryFile(suffix='.gz' if compressed else '') as f:
        if format == 'yaml':
            supervision_set.to_yaml(f.name)
            restored = supervision_set.from_yaml(f.name)
        if format == 'json':
            supervision_set.to_json(f.name)
            restored = supervision_set.from_json(f.name)
        if format == 'jsonl':
            supervision_set.to_jsonl(f.name)
            restored = supervision_set.from_jsonl(f.name)
    assert supervision_set == restored


@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('yaml', False),
        ('yaml', True),
        ('json', False),
        ('json', True),
        ('jsonl', False),
        ('jsonl', True),
    ]
)
def test_cut_set_serialization(cut_set, format, compressed):
    with NamedTemporaryFile(suffix='.gz' if compressed else '') as f:
        if format == 'yaml':
            cut_set.to_yaml(f.name)
            restored = CutSet.from_yaml(f.name)
        if format == 'json':
            cut_set.to_json(f.name)
            restored = CutSet.from_json(f.name)
        if format == 'jsonl':
            cut_set.to_jsonl(f.name)
            restored = CutSet.from_jsonl(f.name)
    assert cut_set == restored


@pytest.fixture
def manifests(recording_set, supervision_set, feature_set, cut_set):
    return {
        'recording_set': recording_set,
        'supervision_set': supervision_set,
        'feature_set': feature_set,
        'cut_set': cut_set
    }


@pytest.mark.parametrize(
    'manifest_type',
    ['recording_set', 'supervision_set', 'feature_set', 'cut_set']
)
@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('yaml', False),
        ('yaml', True),
        ('json', False),
        ('json', True),
        ('jsonl', False),
        ('jsonl', True),
    ]
)
def test_generic_serialization_classmethod(manifests, manifest_type, format, compressed):
    manifest = manifests[manifest_type]
    with NamedTemporaryFile(suffix='.' + format + ('.gz' if compressed else '')) as f:
        manifest.to_file(f.name)
        restored = type(manifest).from_file(f.name)
    assert manifest == restored


@pytest.mark.parametrize(
    'manifest_type',
    ['recording_set', 'supervision_set', 'feature_set', 'cut_set']
)
@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('yaml', False),
        ('yaml', True),
        ('json', False),
        ('json', True),
        ('jsonl', False),
        ('jsonl', True),
    ]
)
def test_generic_serialization(manifests, manifest_type, format, compressed):
    manifest = manifests[manifest_type]
    with NamedTemporaryFile(suffix='.' + format + ('.gz' if compressed else '')) as f:
        store_manifest(manifest, f.name)
        restored = load_manifest(f.name)
        assert manifest == restored


@pytest.mark.skipif(not is_module_available('pyarrow'), reason='Requires pyarrow')
@pytest.mark.parametrize(
    'manifest_type',
    ['recording_set', 'supervision_set', 'cut_set']
)
@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('jsonl', False),
        ('jsonl', True),
    ]
)
def test_lazy_jsonl_deserialization(manifests, manifest_type, format, compressed):
    manifest = manifests[manifest_type]
    with NamedTemporaryFile(suffix='.' + format + ('.gz' if compressed else '')) as f:
        store_manifest(manifest, f.name)
        lazy_manifest = type(manifest).from_jsonl_lazy(f.name)
        # Test iteration
        for eager_obj, lazy_obj in zip(manifest, lazy_manifest):
            assert eager_obj == lazy_obj
        # Test accessing elements by ID
        for lazy_obj in lazy_manifest:
            lazy_manifest[lazy_obj.id]


@pytest.mark.skipif(not is_module_available('pyarrow'), reason='Requires pyarrow')
@pytest.mark.parametrize(
    'manifest_type',
    ['recording_set', 'supervision_set', 'cut_set']
)
def test_lazy_arrow_serialization(manifests, manifest_type):
    manifest = manifests[manifest_type]
    with NamedTemporaryFile(suffix='.arrow') as f:
        manifest.to_file(f.name)
        lazy_manifest = type(manifest).from_file(f.name)
        # Test iteration
        for eager_obj, lazy_obj in zip(manifest, lazy_manifest):
            assert eager_obj == lazy_obj
        # Test accessing elements by ID
        for lazy_obj in lazy_manifest:
            lazy_manifest[lazy_obj.id]


@pytest.mark.skipif(not is_module_available('pyarrow'), reason='Requires pyarrow')
@pytest.mark.parametrize(
    'manifest_type',
    ['recording_set', 'supervision_set', 'cut_set']
)
def test_lazy_arrow_pickling(manifests, manifest_type):
    manifest = manifests[manifest_type]
    with NamedTemporaryFile(suffix='.arrow') as f, NamedTemporaryFile(suffix='.pkl') as f_pkl:
        # Create an .arrow file that can be mmapped
        manifest.to_file(f.name)
        lazy_manifest = type(manifest).from_file(f.name)
        # Create a pickle with a manifest that refers to an mmapped file
        pickle.dump(lazy_manifest, f_pkl)
        f_pkl.flush()
        f_pkl.seek(0)
        # Unpickle
        unpickled_manifest = pickle.load(f_pkl)
        # Lengths are the same
        assert len(lazy_manifest) == len(manifest)
        assert len(unpickled_manifest) == len(manifest)
        # Test iteration
        for eager_obj, lazy_obj, unpickled_obj in zip(manifest, lazy_manifest, unpickled_manifest):
            assert eager_obj == lazy_obj
            assert eager_obj == unpickled_obj
        # Test accessing elements by ID
        for unpickled_obj in unpickled_manifest:
            unpickled_manifest[unpickled_obj.id]


@pytest.mark.skipif(not is_module_available('pyarrow'), reason='Requires pyarrow')
@pytest.mark.parametrize(
    'manifest_type',
    ['recording_set', 'supervision_set', 'cut_set']
)
@pytest.mark.parametrize(
    ['format', 'compressed'],
    [
        ('jsonl', False),
        ('jsonl', True),
    ]
)
def test_lazy_jsonl_to_arrow_serialization(manifests, manifest_type, format, compressed):
    manifest = manifests[manifest_type]
    with NamedTemporaryFile(suffix='.' + format + ('.gz' if compressed else '')) as jsonl_f, \
            NamedTemporaryFile(suffix='.arrow') as arrow_f:
        store_manifest(manifest, jsonl_f.name)
        # For now, we have to first create a JSONL so that we can create mmapped arrow...
        lazy_temp_manifest = type(manifest).from_jsonl_lazy(jsonl_f.name)
        lazy_temp_manifest.to_arrow(arrow_f.name)
        # Now read the real mmapped arrow manifest.
        lazy_manifest = type(manifest).from_arrow(arrow_f.name)
        for eager_obj, lazy_obj in zip(manifest, lazy_manifest):
            assert eager_obj == lazy_obj
