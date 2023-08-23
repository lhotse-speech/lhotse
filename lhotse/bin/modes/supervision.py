import click
from tqdm import tqdm

from lhotse.bin.modes.cli_base import cli
from lhotse.serialization import load_manifest_lazy_or_eager
from lhotse.supervision import SupervisionSet
from lhotse.utils import Pathlike


@cli.group()
def supervision():
    """Commands related to manipulating supervision manifests."""
    pass


@supervision.command()
@click.argument("in_supervision_manifest", type=click.Path(allow_dash=True))
@click.argument("out_supervision_manifest", type=click.Path(allow_dash=True))
@click.option(
    "--ctm-file",
    type=click.Path(exists=True, dir_okay=False),
    help="CTM file containing alignments to add.",
)
@click.option(
    "--alignment-type",
    type=str,
    default="word",
    help="Type of alignment to add (default = `word`).",
)
@click.option(
    "--match-channel/--no-match-channel",
    default=False,
    help="Whether to match channel between CTM and SupervisionSegment (default = False).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Whether to print verbose output.",
)
def with_alignment_from_ctm(
    in_supervision_manifest: Pathlike,
    out_supervision_manifest: Pathlike,
    ctm_file: Pathlike,
    alignment_type: str,
    match_channel: bool,
    verbose: bool,
):
    """
    Add alignments from CTM file to the supervision set.

    :param in_supervision_manifest: Path to input supervision manifest.
    :param out_supervision_manifest: Path to output supervision manifest.
    :param ctm_file: Path to CTM file.
    :param alignment_type: Alignment type (optional, default = `word`).
    :param match_channel: if True, also match channel between CTM and SupervisionSegment
    :param verbose: Whether to print verbose output.
    :return: A new SupervisionSet with AlignmentItem objects added to the segments.
    """
    supervisions = load_manifest_lazy_or_eager(in_supervision_manifest, SupervisionSet)
    supervisions = supervisions.with_alignment_from_ctm(
        ctm_file=ctm_file,
        type=alignment_type,
        match_channel=match_channel,
        verbose=verbose,
    )
    with SupervisionSet.open_writer(out_supervision_manifest, overwrite=True) as writer:
        supervisions = (
            tqdm(supervisions, desc="Writing supervisions") if verbose else supervisions
        )
        for s in supervisions:
            writer.write(s)
