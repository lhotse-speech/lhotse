from typing import List

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.commonvoice import download_commonvoice, prepare_commonvoice
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--language",
    default=["auto"],
    multiple=True,
    help="Languages to prepare (scans CORPUS_DIR for language codes by default).",
)
@click.option(
    "-s",
    "--split",
    default=["train", "dev", "test"],
    multiple=True,
    help="Splits to prepare (available options: train, dev, test, validated, invalidated, other)",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def commonvoice(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    language: List[str],
    split: List[str],
    num_jobs: int,
):
    """
    Mozilla CommonVoice manifest preparation script.
    CORPUS_DIR is expected to contain sub-directories that are named with CommonVoice language codes,
    e.g., "en", "pl", etc.
    """
    languages = language[0] if len(language) == 1 else language
    prepare_commonvoice(
        corpus_dir=corpus_dir,
        languages=languages,
        splits=split,
        output_dir=output_dir,
        num_jobs=num_jobs,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "-l",
    "--languages",
    default="all",
    help="Languages to prepare (scans CORPUS_DIR for language codes by default).",
)
@click.option(
    "-v",
    "--release",
    default="cv-corpus-13.0-2023-03-09",
    help="the name of the CommonVoice release (e.g., 'cv-corpus-13.0-2023-03-09').It is used as part of the download URL.",
)
def commonvoice(
    target_dir: Pathlike,
    languages: List[str],
    release: str,
):
    """Commonvoice download."""
    download_commonvoice(
        target_dir=target_dir,
        languages=languages,
        release=release,
    )
