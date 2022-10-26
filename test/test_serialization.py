import os
from tempfile import NamedTemporaryFile

import pytest

from lhotse import (
    AudioSource,
    CutSet,
    Features,
    FeatureSet,
    MonoCut,
    MultiCut,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    load_manifest,
    store_manifest,
)
from lhotse.serialization import load_manifest_lazy, open_best
from lhotse.supervision import AlignmentItem
from lhotse.testing.dummies import DummyManifest
from lhotse.utils import fastcopy
from lhotse.utils import nullcontext as does_not_raise


@pytest.mark.parametrize(
    ["path", "exception_expectation"],
    [
        ("test/fixtures/audio.json", does_not_raise()),
        ("test/fixtures/supervision.json", does_not_raise()),
        ("test/fixtures/dummy_feats/feature_manifest.json", does_not_raise()),
        ("test/fixtures/libri/cuts.json", does_not_raise()),
        ("test/fixtures/libri/cuts_multi.json", does_not_raise()),
        ("test/fixtures/feature_config.yml", pytest.raises(ValueError)),
        ("no/such/path.xd", pytest.raises(ValueError)),
    ],
)
def test_load_any_lhotse_manifest(path, exception_expectation):
    with exception_expectation:
        load_manifest(path)


@pytest.mark.parametrize(
    ["path", "exception_expectation"],
    [
        ("test/fixtures/audio.json", does_not_raise()),
        ("test/fixtures/supervision.json", does_not_raise()),
        ("test/fixtures/dummy_feats/feature_manifest.json", does_not_raise()),
        ("test/fixtures/libri/cuts.json", does_not_raise()),
        ("test/fixtures/libri/cuts_multi.json", does_not_raise()),
        ("test/fixtures/feature_config.yml", pytest.raises(ValueError)),
        ("no/such/path.xd", pytest.raises(ValueError)),
    ],
)
def test_load_any_lhotse_manifest_lazy(path, exception_expectation):
    with exception_expectation:
        me = load_manifest(path)
        # some temporary files are needed to convert JSON to JSONL
        with NamedTemporaryFile(suffix=".jsonl.gz") as f:
            me.to_file(f.name)
            f.flush()
            ml = load_manifest_lazy(f.name)
            assert list(me) == list(ml)  # equal under iteration


@pytest.fixture
def recording_set():
    return RecordingSet.from_recordings(
        [
            Recording(
                id="x",
                sources=[
                    AudioSource(
                        type="file", channels=[0], source="text/fixtures/mono_c0.wav"
                    ),
                    AudioSource(
                        type="command",
                        channels=[1],
                        source="cat text/fixtures/mono_c1.wav",
                    ),
                ],
                sampling_rate=8000,
                num_samples=4000,
                duration=0.5,
            )
        ]
    )


@pytest.fixture
def supervision_set():
    return SupervisionSet.from_segments(
        [
            SupervisionSegment(
                id="segment-1",
                recording_id="recording-1",
                channel=0,
                start=0.1,
                duration=0.3,
                text="transcript of the first segment",
                language="english",
                speaker="Norman Dyhrentfurth",
                gender="male",
                alignment={
                    "word": [
                        AlignmentItem(symbol="transcript", start=0.1, duration=0.08),
                        AlignmentItem(symbol="of", start=0.18, duration=0.02),
                        AlignmentItem(symbol="the", start=0.2, duration=0.03),
                        AlignmentItem(symbol="first", start=0.23, duration=0.07),
                        AlignmentItem(symbol="segment", start=0.3, duration=0.1),
                    ]
                },
            )
        ]
    )


@pytest.fixture
def feature_set():
    return FeatureSet(
        features=[
            Features(
                recording_id="irrelevant",
                channels=0,
                start=0.0,
                duration=20.0,
                type="fbank",
                num_frames=2000,
                num_features=20,
                frame_shift=0.01,
                sampling_rate=16000,
                storage_type="lilcom",
                storage_path="/irrelevant/",
                storage_key="path.llc",
            )
        ]
    )


@pytest.fixture
def cut_set():
    cut = MonoCut(
        id="cut-1",
        start=0.0,
        duration=10.0,
        channel=0,
        features=Features(
            type="fbank",
            num_frames=100,
            num_features=40,
            frame_shift=0.01,
            sampling_rate=16000,
            start=0.0,
            duration=10.0,
            storage_type="lilcom",
            storage_path="irrelevant",
            storage_key="irrelevant",
        ),
        recording=Recording(
            id="rec-1",
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
            sources=[AudioSource(type="file", channels=[0], source="irrelevant")],
        ),
        supervisions=[
            SupervisionSegment(
                id="sup-1", recording_id="irrelevant", start=0.5, duration=6.0
            ),
            SupervisionSegment(
                id="sup-2", recording_id="irrelevant", start=7.0, duration=2.0
            ),
        ],
    )
    multi_cut = MultiCut(
        id="cut-2",
        start=0.0,
        duration=10.0,
        channel=0,
        recording=Recording(
            id="rec-2",
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
            channel_ids=[0, 1],
            sources=[AudioSource(type="file", channels=[0, 1], source="irrelevant")],
        ),
        supervisions=[
            SupervisionSegment(
                id="sup-3", recording_id="irrelevant", start=0.5, duration=6.0
            ),
            SupervisionSegment(
                id="sup-4", recording_id="irrelevant", start=7.0, duration=2.0
            ),
        ],
    )
    return CutSet.from_cuts(
        [
            cut,
            fastcopy(cut, id="cut-nosup", supervisions=[]),
            fastcopy(cut, id="cut-norec", recording=None),
            fastcopy(cut, id="cut-nofeat", features=None),
            cut.pad(duration=30.0, direction="left"),
            cut.pad(duration=30.0, direction="right"),
            cut.pad(duration=30.0, direction="both"),
            cut.mix(cut, offset_other_by=5.0, snr=8),
            multi_cut,
        ]
    )


@pytest.mark.parametrize(
    ["format", "compressed"],
    [
        ("yaml", False),
        ("yaml", True),
        ("json", False),
        ("json", True),
        ("jsonl", False),
        ("jsonl", True),
    ],
)
def test_feature_set_serialization(feature_set, format, compressed):
    with NamedTemporaryFile(suffix=".gz" if compressed else "") as f:
        if format == "jsonl":
            feature_set.to_jsonl(f.name)
            feature_set_deserialized = FeatureSet.from_jsonl(f.name)
        if format == "json":
            feature_set.to_json(f.name)
            feature_set_deserialized = FeatureSet.from_json(f.name)
        if format == "yaml":
            feature_set.to_yaml(f.name)
            feature_set_deserialized = FeatureSet.from_yaml(f.name)
    assert feature_set_deserialized == feature_set


@pytest.mark.parametrize(
    ["format", "compressed"],
    [
        ("yaml", False),
        ("yaml", True),
        ("json", False),
        ("json", True),
        ("jsonl", False),
        ("jsonl", True),
    ],
)
def test_serialization(recording_set, format, compressed):
    with NamedTemporaryFile(suffix=".gz" if compressed else "") as f:
        if format == "jsonl":
            recording_set.to_jsonl(f.name)
            deserialized = RecordingSet.from_jsonl(f.name)
        if format == "yaml":
            recording_set.to_yaml(f.name)
            deserialized = RecordingSet.from_yaml(f.name)
        if format == "json":
            recording_set.to_json(f.name)
            deserialized = RecordingSet.from_json(f.name)
    assert deserialized == recording_set


@pytest.mark.parametrize(
    ["format", "compressed"],
    [
        ("yaml", False),
        ("yaml", True),
        ("json", False),
        ("json", True),
        ("jsonl", False),
        ("jsonl", True),
    ],
)
def test_supervision_set_serialization(supervision_set, format, compressed):
    with NamedTemporaryFile(suffix=".gz" if compressed else "") as f:
        if format == "yaml":
            supervision_set.to_yaml(f.name)
            restored = supervision_set.from_yaml(f.name)
        if format == "json":
            supervision_set.to_json(f.name)
            restored = supervision_set.from_json(f.name)
        if format == "jsonl":
            supervision_set.to_jsonl(f.name)
            restored = supervision_set.from_jsonl(f.name)
    assert supervision_set == restored


@pytest.mark.parametrize(
    ["format", "compressed"],
    [
        ("yaml", False),
        ("yaml", True),
        ("json", False),
        ("json", True),
        ("jsonl", False),
        ("jsonl", True),
    ],
)
def test_cut_set_serialization(cut_set, format, compressed):
    with NamedTemporaryFile(suffix=".gz" if compressed else "") as f:
        if format == "yaml":
            cut_set.to_yaml(f.name)
            restored = CutSet.from_yaml(f.name)
        if format == "json":
            cut_set.to_json(f.name)
            restored = CutSet.from_json(f.name)
        if format == "jsonl":
            cut_set.to_jsonl(f.name)
            restored = CutSet.from_jsonl(f.name)
    assert cut_set == restored


@pytest.fixture
def manifests(recording_set, supervision_set, feature_set, cut_set):
    return {
        "recording_set": recording_set,
        "supervision_set": supervision_set,
        "feature_set": feature_set,
        "cut_set": cut_set,
    }


@pytest.mark.parametrize(
    "manifest_type", ["recording_set", "supervision_set", "feature_set", "cut_set"]
)
@pytest.mark.parametrize(
    ["format", "compressed"],
    [
        ("yaml", False),
        ("yaml", True),
        ("json", False),
        ("json", True),
        ("jsonl", False),
        ("jsonl", True),
    ],
)
def test_generic_serialization_classmethod(
    manifests, manifest_type, format, compressed
):
    manifest = manifests[manifest_type]
    with NamedTemporaryFile(suffix="." + format + (".gz" if compressed else "")) as f:
        manifest.to_file(f.name)
        restored = type(manifest).from_file(f.name).to_eager()
    assert manifest == restored


@pytest.mark.parametrize(
    "manifest_type", ["recording_set", "supervision_set", "feature_set", "cut_set"]
)
@pytest.mark.parametrize(
    ["format", "compressed"],
    [
        ("yaml", False),
        ("yaml", True),
        ("json", False),
        ("json", True),
        ("jsonl", False),
        ("jsonl", True),
    ],
)
def test_generic_serialization(manifests, manifest_type, format, compressed):
    manifest = manifests[manifest_type]
    with NamedTemporaryFile(suffix="." + format + (".gz" if compressed else "")) as f:
        store_manifest(manifest, f.name)
        restored = load_manifest(f.name)
        assert manifest == restored


@pytest.mark.parametrize(
    "manifest_type", ["recording_set", "supervision_set", "cut_set", "feature_set"]
)
@pytest.mark.parametrize(
    ["format", "compressed"],
    [
        ("jsonl", False),
        ("jsonl", True),
    ],
)
def test_sequential_jsonl_writer(manifests, manifest_type, format, compressed):
    manifest = manifests[manifest_type]
    with NamedTemporaryFile(
        suffix="." + format + (".gz" if compressed else "")
    ) as jsonl_f:
        with manifest.open_writer(jsonl_f.name) as writer:
            for item in manifest:
                writer.write(item)
        restored = writer.open_manifest()
        # Same manifest type
        assert type(manifest) == type(restored)
        # Equal under iteration
        assert list(manifest) == list(restored)


@pytest.mark.parametrize(
    "manifest_type", ["recording_set", "supervision_set", "cut_set", "feature_set"]
)
def test_in_memory_writer(manifests, manifest_type):
    manifest = manifests[manifest_type]
    with manifest.open_writer(None) as writer:
        for item in manifest:
            writer.write(item)
    restored = writer.open_manifest()
    assert manifest == restored


@pytest.mark.parametrize("overwrite", [True, False])
def test_sequential_jsonl_writer_overwrite(overwrite):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=100)
    half = cuts.split(num_splits=2)[0]
    with NamedTemporaryFile(suffix=".jsonl") as jsonl_f:
        # Store the first half
        half.to_file(jsonl_f.name)

        # Open sequential writer
        with CutSet.open_writer(jsonl_f.name, overwrite=overwrite) as writer:
            if overwrite:
                assert all(not writer.contains(id_) for id_ in half.ids)
            else:
                assert all(writer.contains(id_) for id_ in half.ids)


@pytest.mark.parametrize(
    "manifest_type", ["recording_set", "supervision_set", "cut_set", "feature_set"]
)
def test_manifest_is_lazy(manifests, manifest_type):
    # Eager manifest is not lazy
    eager = manifests[manifest_type]
    cls = type(eager)
    assert not eager.is_lazy

    # Save the manifest to JSONL and open it lazily
    with NamedTemporaryFile(suffix=".jsonl") as f, cls.open_writer(f.name) as writer:
        for item in eager:
            writer.write(item)
        f.flush()

        lazy = writer.open_manifest()

        # Lazy manifest is lazy
        assert lazy.is_lazy

        # Concatenation of eager + eager manifests is eager
        eager_eager_cat = eager + eager
        assert eager_eager_cat.is_lazy

        # Concatenation of lazy + eager manifests is lazy
        lazy_eager_cat = lazy + eager
        assert lazy_eager_cat.is_lazy

        # Concatenation of eager + lazy manifests is lazy
        eager_lazy_cat = eager + lazy
        assert eager_lazy_cat.is_lazy

        # Concatenation of eager + lazy manifests is lazy
        lazy_lazy_cat = eager + lazy
        assert lazy_lazy_cat.is_lazy

        # Muxing of eager + eager manifests is lazy
        eager_eager_mux = cls.mux(eager, eager)
        assert eager_eager_mux.is_lazy

        # Muxing of lazy + eager manifests is lazy
        lazy_eager_mux = cls.mux(lazy, eager)
        assert lazy_eager_mux.is_lazy

        # Muxing of eager + lazy manifests is lazy
        eager_lazy_mux = cls.mux(eager, lazy)
        assert eager_lazy_mux.is_lazy

        # Muxing of eager + lazy manifests is lazy
        lazy_lazy_mux = cls.mux(lazy, lazy)
        assert lazy_lazy_mux.is_lazy


@pytest.mark.skipif(os.name == "nt", reason="This test cannot be run on Windows.")
def test_open_pipe(tmp_path):
    data = "text"
    with open_best(f"pipe:gzip -c > {tmp_path}/text.gz", mode="w") as f:
        print(data, file=f)

    with open_best(f"pipe:gunzip -c {tmp_path}/text.gz", mode="r") as f:
        data_read = f.read().strip()

    assert data_read == data


@pytest.mark.skipif(os.name == "nt", reason="This test cannot be run on Windows.")
def test_open_pipe_iter(tmp_path):
    lines = [
        "line0",
        "line1",
        "line2",
    ]
    with open_best(f"pipe:gzip -c > {tmp_path}/text.gz", mode="w") as f:
        for l in lines:
            print(l, file=f)

    lines_read = []
    with open_best(f"pipe:gunzip -c {tmp_path}/text.gz", mode="r") as f:
        for l in f:
            lines_read.append(l.strip())

    assert lines_read == lines
