from typing import Optional, Sequence, Union

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.fleurs import download_fleurs, prepare_fleurs
from lhotse.utils import Pathlike

__all__ = ["fleurs"]


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
def fleurs(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    num_jobs: int,
    lang: Optional[Union[str, Sequence[str]]],
):
    """Fleurs ASR data preparation."""
    prepare_fleurs(corpus_dir, output_dir=output_dir, num_jobs=num_jobs, languages=lang)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-l",
    "--lang",
    multiple=True,
    default=["all"],
    help="Specify which languages to download, e.g., "
    "        lhotse download fleurs . -l hi_in -l en_us "
    "        lhotse download fleurs",
)
@click.option(
    "--force-download",
    type=bool,
    is_flag=True,
    default=False,
    help="Specify whether to overwrite an existing archive",
)
def fleurs(
    target_dir: Pathlike,
    lang: Optional[Union[str, Sequence[str]]],
    force_download: bool = False,
):
    """FLEURS download."""
    download_fleurs(
        target_dir,
        languages=lang,
        force_download=force_download,
    )
