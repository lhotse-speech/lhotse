from cytoolz import sliding_window

from lhotse.cut import CutSet


def test_cut_set():
    cut_set = CutSet(cuts={})

    # CutSet consists of elements with IDs
    cut = cut_set.cuts['cut-1']

    # Each Cut specifies standard time info
    assert 10.0 == cut.begin
    assert 15.0 == cut.duration
    assert 25.0 == cut.end

    # Each Cut consists of supervision segments
    supervisions = cut.supervisions
    assert 3 == len(supervisions)

    # Supervision segments cannot overlap
    for left_segment, right_segment in sliding_window(2, sorted(supervisions, key=lambda s: s.begin)):
        assert left_segment.end <= right_segment.begin

    # Each Cut contains a feature matrix
    features = cut.features
    # TODO: need to push the "trimming" capability from FeatureSet to Features
    feat_matrix = features.load(
        channel=cut.channel,
        begin=cut.begin,
        duration=cut.duration
    )

    # Append Cuts
    another_cut = cut_set.cuts['cut-2']
    concatenated_cuts = cut + another_cut

    # Truncate Cuts
    truncated_cut = cut.truncate(offset=0, duration=cut.duration - 0.5)

    # Overlay Cuts - meaning, add their feature matrices and gather supervisions into a common list
    overlayed_cut = cut.overlay(another_cut)
