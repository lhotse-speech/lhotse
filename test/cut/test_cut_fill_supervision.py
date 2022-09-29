import pytest

from lhotse.testing.dummies import dummy_cut, dummy_multi_cut, dummy_supervision

# Note: dummy_cut, dummy_multi_cut, and dummy_supervision have a duration of 1.0 by default.


@pytest.mark.parametrize(
    "cut",
    [
        # MonoCut with single supervision
        dummy_cut(0, supervisions=[dummy_supervision(0)]),
        # MultiCut with single supervision
        dummy_multi_cut(0, supervisions=[dummy_supervision(0)]),
    ],
)
def test_cut_fill_supervision_identity(cut):
    fcut = cut.fill_supervision()
    assert cut == fcut


@pytest.mark.parametrize(
    "cut",
    [
        # MonoCut with single supervision
        dummy_cut(0, supervisions=[dummy_supervision(0)]),
        # MultiCut with single supervision
        dummy_multi_cut(0, supervisions=[dummy_supervision(0)]),
    ],
)
def test_cut_fill_supervision_expand(cut):
    cut.duration = 7.51
    fcut = cut.fill_supervision()
    # Original is not modified
    assert cut.supervisions[0].start == 0
    assert cut.supervisions[0].duration == 1
    # Result is modified
    assert fcut.supervisions[0].start == 0
    assert fcut.supervisions[0].duration == 7.51


@pytest.mark.parametrize(
    "cut",
    [
        # MonoCut with single supervision
        dummy_cut(0, supervisions=[dummy_supervision(0)]),
        # MultiCut with single supervision
        dummy_multi_cut(0, supervisions=[dummy_supervision(0)]),
    ],
)
def test_cut_fill_supervision_shrink(cut):
    cut.duration = 0.5
    fcut = cut.fill_supervision(shrink_ok=True)
    # Original is not modified
    assert cut.supervisions[0].start == 0
    assert cut.supervisions[0].duration == 1
    # Result is modified
    assert fcut.supervisions[0].start == 0
    assert fcut.supervisions[0].duration == 0.5


@pytest.mark.parametrize(
    "cut",
    [
        # MonoCut with single supervision
        dummy_cut(0, supervisions=[dummy_supervision(0)]),
        # MultiCut with single supervision
        dummy_multi_cut(0, supervisions=[dummy_supervision(0)]),
    ],
)
def test_cut_fill_supervision_shrink_raises_default(cut):
    cut.duration = 0.5
    with pytest.raises(ValueError):
        fcut = cut.fill_supervision()


@pytest.mark.parametrize(
    "cut",
    [
        # MonoCut with no supervision
        dummy_cut(0, supervisions=[]),
        # MultiCut with no supervision
        dummy_multi_cut(0, supervisions=[]),
    ],
)
def test_cut_fill_supervision_add_empty_true(cut):
    fcut = cut.fill_supervision()
    # Original is not modified
    assert len(cut.supervisions) == 0
    # Result is modified
    assert fcut.supervisions[0].start == 0
    assert fcut.supervisions[0].duration == 1


@pytest.mark.parametrize(
    "cut",
    [
        # MonoCut with no supervision
        dummy_cut(0, supervisions=[]),
        # MultiCut with no supervision
        dummy_multi_cut(0, supervisions=[]),
    ],
)
def test_cut_fill_supervision_add_empty_false(cut):
    fcut = cut.fill_supervision(add_empty=False)
    assert cut == fcut


def test_mono_cut_fill_supervision_raises_on_two_supervisions():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0), dummy_supervision(1)])
    with pytest.raises(AssertionError):
        fcut = cut.fill_supervision()


def test_multi_cut_fill_supervision_raises_on_two_supervisions():
    cut = dummy_multi_cut(0, supervisions=[dummy_supervision(0), dummy_supervision(1)])
    with pytest.raises(AssertionError):
        fcut = cut.fill_supervision()


def test_mixed_cut_fill_supervision_identity():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0)])
    cut = cut.mix(dummy_cut(1))  # cuts are 100% overlapping
    fcut = cut.fill_supervision()
    assert cut == fcut


def test_mixed_cut_fill_supervision_expand():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0)])
    cut = cut.pad(duration=7.51)
    fcut = cut.fill_supervision()
    # Original is not modified
    assert cut.supervisions[0].start == 0
    assert cut.supervisions[0].duration == 1
    # Result is modified
    assert fcut.supervisions[0].start == 0
    assert fcut.supervisions[0].duration == 7.51


def test_mixed_cut_fill_supervision_shrink():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0)])
    cut = cut.mix(dummy_cut(1)).truncate(duration=0.5)  # cuts are 100% overlapping
    fcut = cut.fill_supervision(shrink_ok=True)
    # Original is not modified
    assert cut.supervisions[0].start == 0
    assert cut.supervisions[0].duration == 1
    # Result is modified
    assert fcut.supervisions[0].start == 0
    assert fcut.supervisions[0].duration == 0.5


def test_mixed_cut_fill_supervision_shrink_raises_default():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0)])
    cut = cut.mix(dummy_cut(1)).truncate(duration=0.5)  # cuts are 100% overlapping
    with pytest.raises(ValueError):
        fcut = cut.fill_supervision()


def test_mixed_cut_fill_supervision_add_empty_true():
    cut = dummy_cut(0)
    cut = cut.pad(duration=10)
    fcut = cut.fill_supervision()
    # Original is not modified
    assert len(cut.supervisions) == 0
    # Result is modified
    assert fcut.supervisions[0].start == 0
    assert fcut.supervisions[0].duration == 10


def test_mixed_cut_fill_supervision_add_empty_false():
    cut = dummy_cut(0)
    cut = cut.pad(duration=10)
    fcut = cut.fill_supervision(add_empty=False)
    assert cut == fcut


def test_mixed_cut_fill_supervision_raises_on_two_supervisions():
    cut = dummy_cut(0, supervisions=[dummy_supervision(0), dummy_supervision(1)])
    cut = cut.pad(duration=10)
    with pytest.raises(AssertionError):
        fcut = cut.fill_supervision()
