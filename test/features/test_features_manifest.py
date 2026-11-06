import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lilcom", reason="The legacy fixtures store lilcom features.")

from lhotse import CutSet, Features, MonoCut
from lhotse.array import Array, TemporalArray
from lhotse.serialization import deserialize_custom_field, deserialize_item, load_json
from lhotse.shar.utils import to_shar_placeholder
from lhotse.testing.dummies import dummy_cut, dummy_features
from lhotse.utils import fastcopy

# Written by lhotse before Features became a TemporalArray; see generate.py there.
LEGACY = Path("test/fixtures/legacy_features")
LEGACY_CUT_IDS = ["full", "sub", "feats-only", "multi", "mixed", "padded"]
LEGACY_FEATURES_DICT = {
    "type": "kaldi-fbank",
    "num_frames": 50,
    "num_features": 8,
    "frame_shift": 0.01,
    "sampling_rate": 8000,
    "start": 0,
    "duration": 0.5,
    "storage_type": "lilcom_chunky",
    "storage_path": "test/fixtures/legacy_features/feats.lca",
    "storage_key": "0,522",
    "channels": 0,
}


@pytest.fixture(scope="module")
def expected():
    with np.load(LEGACY / "expected.npz") as data:
        return dict(data)


@pytest.fixture(scope="module")
def legacy_cuts():
    return CutSet.from_file(LEGACY / "cuts.jsonl").to_eager()


def assert_features_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_features_is_a_temporal_array():
    feats = Features.from_dict(dict(LEGACY_FEATURES_DICT))
    assert isinstance(feats, TemporalArray)
    assert feats.array == Array(
        storage_type="lilcom_chunky",
        storage_path="test/fixtures/legacy_features/feats.lca",
        storage_key="0,522",
        shape=[50, 8],
    )
    assert feats.temporal_dim == 0
    assert feats.shape == [50, 8]
    assert (feats.num_frames, feats.num_features) == (50, 8)
    assert (feats.storage_type, feats.storage_key) == ("lilcom_chunky", "0,522")
    assert (feats.duration, feats.end) == (0.5, 0.5)


def test_features_legacy_dict_round_trips_through_new_layout():
    feats = Features.from_dict(dict(LEGACY_FEATURES_DICT))
    data = feats.to_dict()
    assert data == {
        "array": {
            "storage_type": "lilcom_chunky",
            "storage_path": "test/fixtures/legacy_features/feats.lca",
            "storage_key": "0,522",
            "shape": [50, 8],
        },
        "temporal_dim": 0,
        "frame_shift": 0.01,
        "start": 0,
        "type": "kaldi-fbank",
        "sampling_rate": 8000,
        "duration": 0.5,
        "channels": 0,
    }
    restored = deserialize_item(data)
    assert type(restored) is Features
    assert restored == feats


def test_features_legacy_dict_without_storage_path():
    data = dict(LEGACY_FEATURES_DICT)
    del data["storage_path"]
    feats = Features.from_dict(data)
    assert feats.storage_path is None
    assert Features.from_dict(feats.to_dict()) == feats


def test_features_legacy_dict_without_frame_shift_infers_it():
    data = dict(LEGACY_FEATURES_DICT)
    del data["frame_shift"]
    with pytest.warns(UserWarning, match="frame_shift"):
        feats = Features.from_dict(data)
    assert feats.frame_shift == 0.01


def test_features_legacy_constructor_matches_array_constructor():
    legacy = Features(**LEGACY_FEATURES_DICT)
    unified = Features(
        array=Array(
            storage_type="lilcom_chunky",
            storage_path="test/fixtures/legacy_features/feats.lca",
            storage_key="0,522",
            shape=[50, 8],
        ),
        type="kaldi-fbank",
        frame_shift=0.01,
        sampling_rate=8000,
        start=0,
        duration=0.5,
        channels=0,
    )
    assert legacy == unified


def test_features_duration_defaults_to_num_frames_times_frame_shift():
    data = {k: v for k, v in LEGACY_FEATURES_DICT.items() if k != "duration"}
    assert Features(**data).duration == 0.5


def test_features_copy_with_legacy_fields_does_not_alias_the_array():
    feats = dummy_features(0)
    copied = fastcopy(feats, storage_path="elsewhere", num_frames=7)
    assert (copied.storage_path, copied.shape) == ("elsewhere", [7, 23])
    assert feats.storage_path == "test/fixtures/dummy_feats/storage"
    assert feats.shape == [100, 23]

    copied.storage_key = "new-key"
    assert copied.storage_key == "new-key"
    assert feats.storage_key == "dbf9a0ec-f79d-4eb8-ae83-143a6d5de64d.npy"


def test_features_in_custom_field_are_deserialized_as_features():
    feats = dummy_features(0)
    custom = deserialize_custom_field({"feats": feats.to_dict()})
    assert type(custom["feats"]) is Features
    assert custom["feats"] == feats


def test_features_shar_placeholder_keeps_feature_metadata():
    cut = dummy_cut(0, duration=0.5, with_data=True)
    placeholder = to_shar_placeholder(cut.features, cut)
    assert type(placeholder) is Features
    assert placeholder.array == Array(
        storage_type="shar", storage_path="", storage_key="", shape=[100, 23]
    )
    assert (placeholder.start, placeholder.duration) == (0, 0.5)
    assert placeholder.type == cut.features.type
    assert cut.features.storage_type == "numpy_files"


def test_features_move_to_memory_returns_features():
    feats = dummy_features(0)
    in_memory = feats.move_to_memory(start=0.2, duration=0.5)
    assert type(in_memory) is Features
    assert (in_memory.start, in_memory.duration) == (0, 0.5)
    assert in_memory.shape == [50, 23]
    assert in_memory.storage_type == "memory_raw"
    np.testing.assert_array_equal(in_memory.load(), feats.load(0.2, 0.5))


@pytest.mark.parametrize("cut_id", LEGACY_CUT_IDS)
def test_legacy_cut_manifest_loads_the_same_features(legacy_cuts, expected, cut_id):
    cut = legacy_cuts[cut_id]
    assert_features_equal(cut.load_features(), expected[cut_id])
    assert_features_equal(
        cut.truncate(offset=0.05, duration=0.2).load_features(),
        expected[f"{cut_id}-trunc"],
    )
    assert_features_equal(
        cut.move_to_memory().load_features(), expected[f"{cut_id}-memory"]
    )


@pytest.mark.parametrize("cut_id", LEGACY_CUT_IDS)
def test_legacy_cut_manifest_round_trips(legacy_cuts, expected, tmp_path, cut_id):
    path = tmp_path / "cuts.jsonl"
    legacy_cuts.subset(cut_ids=[cut_id]).to_file(path)
    restored = CutSet.from_file(path).to_eager()[cut_id]
    assert restored == legacy_cuts[cut_id]
    assert_features_equal(restored.load_features(), expected[cut_id])


def test_legacy_feature_only_cut_keeps_metadata(legacy_cuts):
    cut = legacy_cuts["feats-only"]
    assert not cut.has_recording
    assert isinstance(cut.features, Features)
    assert (cut.sampling_rate, cut.frame_shift, cut.num_features) == (8000, 0.01, 8)
    assert cut.features_type == "kaldi-fbank"
    assert cut.num_frames == 30


def test_legacy_shar_loads_the_same_features(expected):
    cuts = CutSet.from_shar(in_dir=LEGACY / "shar")
    loaded = {cut.id: cut.load_features() for cut in cuts}
    assert sorted(loaded) == ["full", "multi", "sub"]
    for cut_id, feats in loaded.items():
        assert_features_equal(feats, expected[cut_id])


def test_legacy_webdataset_loads_the_same_features(expected):
    pytest.importorskip("webdataset")
    cuts = CutSet.from_webdataset(str(LEGACY / "memory_cuts.tar"))
    loaded = {cut.id: cut.load_features() for cut in cuts}
    assert sorted(loaded) == ["feats-only", "full", "sub"]
    for cut_id, feats in loaded.items():
        assert_features_equal(feats, expected[f"{cut_id}-memory"])


def test_legacy_pickle_loads_the_same_features(expected):
    with open(LEGACY / "memory_cuts.pkl", "rb") as f:
        cuts = pickle.load(f)
    assert [cut.id for cut in cuts] == ["full", "sub", "feats-only"]
    for cut in cuts:
        assert isinstance(cut.features, Features)
        assert cut.features.storage_type == "memory_raw"
        assert_features_equal(cut.load_features(), expected[f"{cut.id}-memory"])
        assert pickle.loads(pickle.dumps(cut)) == cut


def test_new_shar_round_trips_features(legacy_cuts, expected, tmp_path):
    cuts = legacy_cuts.subset(cut_ids=["full", "sub", "multi"])
    cuts.to_shar(tmp_path, fields={"features": "lilcom"}, shard_size=10)
    restored = CutSet.from_shar(in_dir=tmp_path)
    for cut in restored:
        assert isinstance(cut.features, Features)
        assert cut.features.type == "kaldi-fbank"
        assert_features_equal(cut.load_features(), expected[cut.id])


def test_features_load_rejects_start_before_features_beyond_tolerance():
    feats = Features.from_dict(dict(LEGACY_FEATURES_DICT, start=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(feats.load(start=0.9995), feats.load())
    with pytest.raises(ValueError, match="Cannot load"):
        feats.load(start=0.99)


def test_mono_cut_from_dict_accepts_both_feature_layouts():
    cut = dummy_cut(0, with_data=True)
    new = MonoCut.from_dict(cut.to_dict())
    legacy_dict = cut.to_dict()
    legacy_dict["features"] = {
        "type": cut.features.type,
        "num_frames": cut.features.num_frames,
        "num_features": cut.features.num_features,
        "frame_shift": cut.features.frame_shift,
        "sampling_rate": cut.features.sampling_rate,
        "start": cut.features.start,
        "duration": cut.features.duration,
        "storage_type": cut.features.storage_type,
        "storage_path": cut.features.storage_path,
        "storage_key": cut.features.storage_key,
        "recording_id": cut.features.recording_id,
        "channels": cut.features.channels,
    }
    assert MonoCut.from_dict(legacy_dict) == new == cut


def test_cli_cut_simple_reads_legacy_feature_manifest(tmp_path):
    from click.testing import CliRunner

    from lhotse.bin.modes.cut import simple

    legacy_manifest = "test/fixtures/dummy_feats/feature_manifest.json"
    output = tmp_path / "cuts.jsonl.gz"
    result = CliRunner().invoke(simple, ["-f", legacy_manifest, str(output)])
    assert result.exit_code == 0, result.output
    cuts = CutSet.from_file(output).to_eager()
    expected = [Features.from_dict(d) for d in load_json(legacy_manifest)]
    assert [cut.features for cut in cuts] == expected
    assert [cut.duration for cut in cuts] == [0.5, 0.5, 1.0, 1.0]
