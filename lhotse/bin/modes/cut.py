import random
from typing import Tuple

import click
import numpy as np

from lhotse.bin.modes.cli_base import cli
from lhotse.cut import make_trivial_cut_set, CutSet
from lhotse.features import FeatureSet
from lhotse.manipulation import split
from lhotse.supervision import SupervisionSet
from lhotse.utils import Pathlike


@cli.command()
@click.argument('supervision_manifest', type=click.Path(exists=True, dir_okay=False))
@click.argument('feature_manifest', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_cut_manifest', type=click.Path())
def make_trivial_cuts(
        supervision_manifest: Pathlike,
        feature_manifest: Pathlike,
        output_cut_manifest: Pathlike,
):
    """
    Create a CutSet stored in OUTPUT_CUT_MANIFEST that contains supervision regions from SUPERVISION_MANIFEST
    and features supplied by FEATURE_MANIFEST. This is the most trivial way to create Cuts.
    """
    supervision_set = SupervisionSet.from_yaml(supervision_manifest)
    feature_set = FeatureSet.from_yaml(feature_manifest)
    cut_set = make_trivial_cut_set(supervision_set=supervision_set, feature_set=feature_set)
    cut_set.to_yaml(output_cut_manifest)


@cli.command()
@click.argument('supervision_manifest', type=click.Path(exists=True, dir_okay=False))
@click.argument('feature_manifest', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_cut_manifest', type=click.Path())
@click.option('-r', '--random-seed', default=42, type=int, help='Random seed value.')
@click.option('-s', '--snr-range', type=(float, float), default=(20, 20),
              help='Range of SNR values (in dB) that will be uniformly sampled in order to overlay the signals.')
@click.option('-o', '--offset-range', type=(float, float), default=(0.5, 0.5),
              help='Range of relative offset values (0 - 1), which will offset the "right" signal by this many times '
                   'the duration of the "left" signal. It is uniformly sampled for each overlay operation.')
def make_overlayed_cuts(
        supervision_manifest: Pathlike,
        feature_manifest: Pathlike,
        output_cut_manifest: Pathlike,
        random_seed: int,
        snr_range: Tuple[float, float],
        offset_range: Tuple[float, float]
):
    """
    Create a CutSet stored in OUTPUT_CUT_MANIFEST that contains supervision regions from SUPERVISION_MANIFEST
    and features supplied by FEATURE_MANIFEST. It first creates a trivial CutSet, splits it into two equal parts,
    and overlays the signals features to create a mix. The parameters of the mix are controlled via CLI options.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    supervision_set = SupervisionSet.from_yaml(supervision_manifest)
    feature_set = FeatureSet.from_yaml(feature_manifest)

    source_cut_set = make_trivial_cut_set(supervision_set=supervision_set, feature_set=feature_set)
    left_cuts, right_cuts = split(source_cut_set, num_splits=2, randomize=True)

    snrs = np.random.uniform(*snr_range, size=len(left_cuts)).tolist()
    relative_offsets = np.random.uniform(*offset_range, size=len(left_cuts)).tolist()

    cuts = (
        left_cut.overlay(
            right_cut,
            offset_other_by=left_cut.duration * relative_offset,
            snr=snr
        )
        for left_cut, right_cut, snr, relative_offset in zip(left_cuts, right_cuts, snrs, relative_offsets)
    )

    # Make the overlayed cut set contain both the overlayed cuts and the source cuts
    overlayed_cut_set = CutSet(cuts={cut.id: cut for cut in cuts}) + source_cut_set
    overlayed_cut_set.to_yaml(output_cut_manifest)
