from pathlib import Path

import pytest

from lhotse import CutSet
from lhotse.shar.writers.shar import SharWriter
from lhotse.testing.dummies import (
    DummyManifest,
    dummy_cut,
    dummy_features,
    dummy_in_memory_features,
    dummy_multi_cut,
    dummy_recording,
    dummy_temporal_array,
    dummy_temporal_array_uint8,
)


@pytest.fixture
def cuts():
    return DummyManifest(CutSet, begin_id=0, end_id=20, with_data=True)


@pytest.fixture
def shar_dir(tmpdir, cuts):
    tmpdir = Path(tmpdir)
    # Prepare data
    writer = SharWriter(
        tmpdir,
        fields={
            "recording": "wav",
            "features": "lilcom",
            "custom_embedding": "numpy",
            "custom_features": "lilcom",
            "custom_indexes": "numpy",
            "custom_recording": "wav",
        },
        shard_size=10,
    )
    with writer:
        for c in cuts:
            writer.write(c)
    return tmpdir


@pytest.fixture
def multi_cuts():
    cuts = CutSet.from_cuts([dummy_multi_cut(i, with_data=True) for i in range(20)])
    for cut in cuts:
        cut.custom_indexes = dummy_temporal_array_uint8()
    return cuts


@pytest.fixture
def multi_cuts_multi_audio_source():
    cuts = CutSet.from_cuts(
        [dummy_multi_cut(i, with_data=True, source_per_channel=True) for i in range(20)]
    )
    for cut in cuts:
        cut.custom_indexes = dummy_temporal_array_uint8()
    return cuts


@pytest.fixture()
def multi_cut_shar_dir(tmpdir, multi_cuts):
    tmpdir = Path(tmpdir)
    # Prepare data
    writer = SharWriter(
        tmpdir,
        fields={
            "recording": "wav",
            "features": "lilcom",
            "custom_indexes": "numpy",
        },
        shard_size=10,
    )
    with writer:
        for c in multi_cuts:
            writer.write(c)
    return tmpdir


@pytest.fixture()
def multi_cut_multi_audio_source_shar_dir(tmpdir, multi_cuts_multi_audio_source):
    tmpdir = Path(tmpdir)
    # Prepare data
    writer = SharWriter(
        tmpdir,
        fields={
            "recording": "wav",
            "features": "lilcom",
            "custom_indexes": "numpy",
        },
        shard_size=10,
    )
    with writer:
        for c in multi_cuts_multi_audio_source:
            writer.write(c)
    return tmpdir


@pytest.fixture
def cuts_from_long_recordings():
    cuts = CutSet.from_cuts(
        [
            dummy_cut(
                i,
                start=3.0,
                duration=5.0,
                recording=dummy_recording(i, duration=10, with_data=True),
                features=dummy_in_memory_features(i, start=0.0, duration=10.0),
            )
            for i in range(20)
        ]
    )
    for i, c in enumerate(cuts):
        c.custom_features = dummy_temporal_array(
            start=0.0, num_frames=1000, frame_shift=0.01
        )
        c.custom_indexes = dummy_temporal_array_uint8(
            start=0.0, num_frames=1000, frame_shift=0.01
        )
    return cuts
