from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.seamless_interaction import (
    LABELS,
    SPLITS,
    download_seamless_interaction,
    prepare_seamless_interaction,
)
from lhotse.utils import Pathlike

__all__ = ["seamless_interaction"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--labels",
    type=click.Choice(LABELS),
    multiple=True,
    default=LABELS,
    help="Conversation types to prepare.",
)
@click.option(
    "-s",
    "--splits",
    type=click.Choice(SPLITS),
    multiple=True,
    default=SPLITS,
    help="Splits to prepare.",
)
def seamless_interaction(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    labels: Sequence[str],
    splits: Sequence[str],
):
    """Seamless Interaction data preparation (one two-channel recording per conversation)."""
    for label in labels:
        prepare_seamless_interaction(
            corpus_dir, output_dir=output_dir, label=label, splits=splits
        )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-l",
    "--labels",
    type=click.Choice(LABELS),
    multiple=True,
    default=LABELS,
    help="Conversation types to download.",
)
@click.option(
    "-s",
    "--splits",
    type=click.Choice(SPLITS),
    multiple=True,
    default=SPLITS,
    help="Splits to download.",
)
@click.option(
    "--force-download",
    is_flag=True,
    default=False,
    help="Download the files even if they already exist.",
)
def seamless_interaction(
    target_dir: Pathlike,
    labels: Sequence[str],
    splits: Sequence[str],
    force_download: bool,
):
    """Seamless Interaction download (audio, transcripts and voice activity, no video)."""
    download_seamless_interaction(
        target_dir, labels=labels, splits=splits, force_download=force_download
    )
