from pathlib import Path

import pytest

from lhotse import CutSet
from lhotse.shar.writers.shar import SharWriter
from lhotse.testing.dummies import DummyManifest


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
