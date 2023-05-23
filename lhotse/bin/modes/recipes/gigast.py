import logging
from typing import List

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.gigast import (
    GIGASPEECH_PARTS,
    GIGAST_LANGS,
    download_gigast,
    prepare_gigast,
)
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("manifests_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--language",
    type=click.Choice(("auto",) + GIGAST_LANGS),
    default=["auto"],
    multiple=True,
    help="Languages to download. one of: 'all' (downloads all known languages); a single language code (e.g., 'en')",
)
@click.option(
    "--subset",
    type=click.Choice(("auto",) + GIGASPEECH_PARTS),
    multiple=True,
    default=["auto"],
    help="Which parts of Gigaspeech to download (by default XL + DEV + TEST).",
)
def gigast(
    corpus_dir: Pathlike,
    manifests_dir: Pathlike,
    output_dir: Pathlike,
    language: List[str],
    subset: List[str],
):
    """GigaST data preparation."""
    languages = language[0] if len(language) == 1 else language
    if "auto" in subset:
        subset = "auto"
    prepare_gigast(
        corpus_dir=corpus_dir,
        manifests_dir=manifests_dir,
        output_dir=output_dir,
        languages=languages,
        dataset_parts=subset,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-l",
    "--languages",
    default="all",
    help="Languages to download. one of: 'all' (downloads all known languages); a single language code (e.g., 'en')",
)
@click.option("--force-download", is_flag=True, default=False, help="Force download")
def gigast(
    target_dir: Pathlike,
    languages: List[str],
    force_download: bool = False,
):
    """GigaST download."""
    download_gigast(
        target_dir=target_dir,
        languages=languages,
        force_download=force_download,
    )
