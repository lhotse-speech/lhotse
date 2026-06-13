"""Tests for the ``.copy_with(**kwargs)`` convenience method added to Lhotse manifests.

See https://github.com/lhotse-speech/lhotse/issues/242.
"""

import pytest

from lhotse.cut import MixedCut, PaddingCut
from lhotse.testing.dummies import (
    dummy_array,
    dummy_cut,
    dummy_features,
    dummy_multi_cut,
    dummy_recording,
    dummy_supervision,
    dummy_temporal_array,
)
from lhotse.utils import fastcopy


def _padding_cut() -> PaddingCut:
    return PaddingCut(
        id="pad",
        duration=1.0,
        sampling_rate=16000,
        num_samples=16000,
        num_frames=100,
        num_features=80,
        frame_shift=0.01,
        feat_value=0.0,
    )


def _mixed_cut() -> MixedCut:
    return dummy_cut(0, with_data=False).pad(duration=2.0)


# (factory, field_to_overwrite, new_value)
MANIFESTS = [
    pytest.param(lambda: dummy_recording(0), "id", "new-rec", id="Recording"),
    pytest.param(
        lambda: dummy_supervision(0), "text", "new text", id="SupervisionSegment"
    ),
    pytest.param(lambda: dummy_features(0), "start", 5.0, id="Features"),
    pytest.param(lambda: dummy_array(), "storage_key", "new-key", id="Array"),
    pytest.param(lambda: dummy_temporal_array(), "start", 2.0, id="TemporalArray"),
    pytest.param(lambda: dummy_cut(0), "id", "new-cut", id="MonoCut"),
    pytest.param(lambda: dummy_multi_cut(0), "id", "new-multi", id="MultiCut"),
    pytest.param(_padding_cut, "id", "new-pad", id="PaddingCut"),
    pytest.param(_mixed_cut, "id", "new-mixed", id="MixedCut"),
]


@pytest.mark.parametrize("factory,field,new_value", MANIFESTS)
def test_copy_with_overwrites_field(factory, field, new_value):
    obj = factory()
    copy = obj.copy_with(**{field: new_value})
    assert copy is not obj
    assert getattr(copy, field) == new_value


@pytest.mark.parametrize("factory,field,new_value", MANIFESTS)
def test_copy_with_does_not_mutate_original(factory, field, new_value):
    obj = factory()
    original_value = getattr(obj, field)
    obj.copy_with(**{field: new_value})
    assert getattr(obj, field) == original_value


@pytest.mark.parametrize("factory,field,new_value", MANIFESTS)
def test_copy_with_matches_fastcopy(factory, field, new_value):
    obj = factory()
    assert obj.copy_with(**{field: new_value}) == fastcopy(obj, **{field: new_value})


@pytest.mark.parametrize("factory,field,new_value", MANIFESTS)
def test_copy_with_no_kwargs_is_equal_copy(factory, field, new_value):
    obj = factory()
    assert obj.copy_with() == obj


def test_copy_with_can_set_custom_field_on_supervision():
    supervision = dummy_supervision(0)
    modified = supervision.copy_with(custom={"speaker_age": 42})
    assert modified.custom == {"speaker_age": 42}
    assert supervision.custom != {"speaker_age": 42}
