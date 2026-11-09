from copy import deepcopy
from functools import cached_property

import numpy as np
import pytest

from lhotse import CutSet, Features, MonoCut, MultiCut, NumpyFilesWriter, Recording
from lhotse.array import Array, TemporalArray
from lhotse.features.io import MemoryRawWriter
from lhotse.testing.dummies import dummy_alignment, dummy_recording, dummy_supervision


@pytest.mark.parametrize("channels", [[1, 0], [1]])
@pytest.mark.parametrize("combine_cut_set", [False, True])
def test_multi_cut_from_mono_audio(channels, combine_cut_set, tmp_path):
    recording = Recording.from_file("test/fixtures/stereo.wav")
    cuts = [
        MonoCut(
            id=f"cut-{channel}",
            start=0.25,
            duration=0.5,
            channel=channel,
            recording=recording,
            supervisions=[
                dummy_supervision(
                    channel,
                    start=0.1,
                    duration=0.2,
                    channel=channel,
                    alignment=dummy_alignment(text="hello", start=0.1, duration=0.2),
                ).copy_with(recording_id=recording.id)
            ],
        )
        for channel in channels
    ]
    original = deepcopy(cuts)
    expected_audio = np.concatenate(
        [cut.load_audio() for cut in sorted(cuts, key=lambda cut: cut.channel)]
    )

    if combine_cut_set:
        combined = CutSet.from_cuts(cuts).combine_same_recording_channels()
        assert len(combined) == 1
        multi = combined[0]
    else:
        multi = MultiCut.from_mono(*cuts)

    audio = multi.load_audio()
    np.testing.assert_array_equal(audio, expected_audio)
    assert audio.shape == (len(channels), 4000)
    assert audio.dtype == np.float32
    assert isinstance(multi, MultiCut)
    assert multi.recording == recording
    assert multi.recording_id == recording.id
    assert multi.sampling_rate == recording.sampling_rate
    assert multi.num_samples == 4000
    assert multi.id == cuts[0].id
    assert multi.start == 0.25
    assert multi.duration == 0.5
    assert multi.channel == sorted(channels)
    assert multi.supervisions == [s for cut in cuts for s in cut.supervisions]
    assert multi.features is None
    assert multi.custom is None
    for cut in cuts:
        np.testing.assert_array_equal(
            multi.load_audio(channel=cut.channel), cut.load_audio()
        )

    path = tmp_path / "cuts.jsonl.gz"
    CutSet.from_cuts([multi]).to_file(path)
    restored = CutSet.from_file(path)[0]
    assert restored == multi
    np.testing.assert_array_equal(restored.load_audio(), expected_audio)
    assert cuts == original


@pytest.mark.parametrize("in_memory", [False, True])
@pytest.mark.parametrize("with_recording", [False, True])
def test_multi_cut_from_mono_preserves_manifests(in_memory, with_recording, tmp_path):
    recording = (
        dummy_recording(0, with_data=True)
        if in_memory
        else Recording.from_file("test/fixtures/libri/libri-1088-134315-0000.wav")
    )
    feature_data = np.arange(200, dtype=np.float32).reshape(100, 2)
    embedding = np.arange(3, dtype=np.float32)
    indexes = np.arange(100, dtype=np.uint8)
    writer = MemoryRawWriter() if in_memory else NumpyFilesWriter(tmp_path)
    with writer:
        features = Features(
            type="fbank",
            num_frames=100,
            num_features=2,
            frame_shift=0.01,
            sampling_rate=recording.sampling_rate,
            start=0.0,
            duration=1.0,
            storage_type=writer.name,
            storage_path=writer.storage_path,
            storage_key=writer.write("features", feature_data),
            recording_id=recording.id,
            channels=0,
        )
        custom = {
            "embedding": writer.store_array("embedding", embedding),
            "indexes": writer.store_array(
                "indexes", indexes, frame_shift=0.01, temporal_dim=0
            ),
            "target": recording,
            "metadata": {"text": "hello", "labels": [1, 2]},
        }
    cut = MonoCut(
        id="cut",
        start=0.25,
        duration=0.5,
        channel=0,
        recording=recording if with_recording else None,
        features=features,
        custom=custom,
    )
    original = deepcopy(cut)

    multi = MultiCut.from_mono(cut)

    assert multi.features == features
    assert multi.recording == cut.recording
    assert multi.custom == custom
    assert isinstance(multi.features, Features)
    assert isinstance(multi.embedding, Array)
    assert isinstance(multi.indexes, TemporalArray)
    assert isinstance(multi.target, Recording)
    assert multi is not cut
    assert multi.channel == [0]
    assert multi.num_frames == 50
    assert multi.num_features == 2
    assert multi.recording_id == recording.id
    assert multi.sampling_rate == recording.sampling_rate

    restored = MultiCut.from_dict(multi.to_dict())
    if not in_memory:
        path = tmp_path / "cuts.jsonl"
        CutSet.from_cuts([multi]).to_file(path)
        restored = CutSet.from_file(path)[0]
    assert restored == multi
    for result in (multi, restored):
        np.testing.assert_array_equal(result.load_features(), feature_data[25:75])
        assert result.load_features().dtype == np.float32
        np.testing.assert_array_equal(result.load_embedding(), embedding)
        assert result.load_embedding().dtype == np.float32
        np.testing.assert_array_equal(result.load_indexes(), indexes[25:75])
        assert result.load_indexes().dtype == np.uint8
        expected_audio = recording.load_audio(offset=0.25, duration=0.5)
        np.testing.assert_array_equal(result.load_target(), expected_audio)
        if with_recording:
            np.testing.assert_array_equal(result.load_audio(), expected_audio)
        else:
            assert result.load_audio() is None
    assert cut == original


def test_multi_cut_from_mono_requires_at_least_one_cut():
    with pytest.raises(AssertionError, match="Cuts must match"):
        MultiCut.from_mono()


def test_multi_cut_from_mono_ignores_cached_attributes():
    class CachedMonoCut(MonoCut):
        @cached_property
        def cached_duration(self):
            return self.duration

    recording = Recording.from_file("test/fixtures/stereo.wav")
    cut = CachedMonoCut(
        id="cut", start=0.25, duration=0.5, channel=0, recording=recording
    )
    assert cut.cached_duration == 0.5

    multi = MultiCut.from_mono(cut)

    assert multi.duration == cut.duration
    assert multi.recording is recording
    np.testing.assert_array_equal(multi.load_audio(), cut.load_audio())
    assert cut.cached_duration == 0.5
