import pytest

from lhotse import CutSet
from lhotse.testing.dummies import DummyManifest


@pytest.mark.parametrize("drop_everything", [True, False])
def test_cut_set_from_shar(tmp_path, drop_everything):
    # Prepare data -- it needs to have missing values for some cuts
    cuts = DummyManifest(CutSet, begin_id=0, end_id=20, with_data=True)
    cuts[0].recording = None
    cuts[0].features = None
    cuts[0].custom_indexes = None
    cuts[0].custom_recording = None
    cuts[0].custom_features = None
    if drop_everything:
        cuts[0].custom_embedding = None

    # Prepare system under test
    cuts.to_shar(
        tmp_path,
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
    cuts_shar = CutSet.from_shar(in_dir=tmp_path).to_eager()

    assert not cuts_shar[0].has_recording
    assert not cuts_shar[0].has_features
    assert not cuts_shar[0].has_custom("custom_indexes")
    assert not cuts_shar[0].has_custom("custom_recording")
    assert not cuts_shar[0].has_custom("custom_features")
    assert cuts_shar[0].has_custom("custom_embedding") == (not drop_everything)
    for cut in cuts_shar.subset(last=19):
        assert cut.has_recording
        assert cut.has_features
        assert cut.has_custom("custom_indexes")
        assert cut.has_custom("custom_recording")
        assert cut.has_custom("custom_features")
        assert cut.has_custom("custom_embedding")
