from typing import Optional, Sequence, Union

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.mtedx import download_mtedx, prepare_mtedx
from lhotse.utils import Pathlike

__all__ = ["mtedx"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
@click.option(
    "-l",
    "--lang",
    multiple=True,
    default=["all"],
    help="Specify which languages to prepare, e.g., "
    "        lhoste prepare librispeech mtedx_corpus data -l de -l fr -l es ",
)
def mtedx(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    num_jobs: int,
    lang: Optional[Union[str, Sequence[str]]],
):
    """MTEDx ASR data preparation."""
    prepare_mtedx(corpus_dir, output_dir=output_dir, num_jobs=num_jobs, languages=lang)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-l",
    "--lang",
    multiple=True,
    default=["all"],
    help="Specify which languages to download, e.g., "
    "        lhoste download mtedx . -l de -l fr -l es "
    "        lhoste download mtedx",
)
def mtedx(
    target_dir: Pathlike,
    lang: Optional[Union[str, Sequence[str]]],
):
    """MTEDx download."""
    download_mtedx(target_dir, languages=lang)
