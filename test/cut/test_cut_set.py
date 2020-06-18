from tempfile import NamedTemporaryFile

from lhotse.cut import Cut, MixedCut


def test_cut_set_iteration(cut_set_with_mixed_cut):
    cuts = list(cut_set_with_mixed_cut)
    assert len(cut_set_with_mixed_cut) == 3
    assert len(cuts) == 3


def test_cut_set_holds_both_simple_and_mixed_cuts(cut_set_with_mixed_cut):
    simple_cuts = cut_set_with_mixed_cut.simple_cuts.values()
    assert all(isinstance(c, Cut) for c in simple_cuts)
    assert len(simple_cuts) == 2
    mixed_cuts = cut_set_with_mixed_cut.mixed_cuts.values()
    assert all(isinstance(c, MixedCut) for c in mixed_cuts)
    assert len(mixed_cuts) == 1


def test_cut_set_can_be_attached_to_mixed_cuts(cut_set_with_mixed_cut, cut_set):
    cut_set_with_mixed_cut.with_source_cuts_from(cut_set)
    mixed_cut = cut_set_with_mixed_cut.cuts['mixed-cut-id']
    assert mixed_cut._cut_set == cut_set


def test_simple_cut_set_serialization(cut_set):
    with NamedTemporaryFile() as f:
        cut_set.to_yaml(f.name)
        restored = cut_set.from_yaml(f.name)
    assert cut_set == restored


def test_mixed_cut_set_serialization(cut_set_with_mixed_cut):
    with NamedTemporaryFile() as f:
        cut_set_with_mixed_cut.to_yaml(f.name)
        restored = cut_set_with_mixed_cut.from_yaml(f.name)
    assert cut_set_with_mixed_cut == restored
