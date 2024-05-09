import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_voxpopuli, prepare_voxpopuli
from lhotse.recipes.voxpopuli import (
    LANGUAGES,
    LANGUAGES_V2,
    S2S_SRC_LANGUAGES,
    S2S_TGT_LANGUAGES,
)
from lhotse.utils import Pathlike

__all__ = ["voxpopuli"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--task",
    type=click.Choice(["asr", "s2s", "lm"]),
    default="asr",
    help="The task for which to prepare the VoxPopuli data.",
    show_default=True,
)
@click.option(
    "--lang",
    type=click.Choice(LANGUAGES + LANGUAGES_V2),
    default="en",
    help="The language to prepare (only used if task is asr or lm).",
    show_default=True,
)
@click.option(
    "--src-lang",
    type=click.Choice(S2S_SRC_LANGUAGES),
    default=None,
    help="The source language (only used if task is s2s).",
    show_default=True,
)
@click.option(
    "--tgt-lang",
    type=click.Choice(S2S_TGT_LANGUAGES),
    default=None,
    help="The target language (only used if task is s2s).",
    show_default=True,
)
@click.option(
    "--num-jobs",
    "-j",
    type=int,
    default=1,
    help="Number of parallel jobs (can provide small speed-ups).",
    show_default=True,
)
def voxpopuli(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    task: str,
    lang: str,
    src_lang: str,
    tgt_lang: str,
    num_jobs: int,
):
    """voxpopuli data preparation."""
    prepare_voxpopuli(
        corpus_dir,
        output_dir=output_dir,
        task=task,
        lang=lang,
        source_lang=src_lang,
        target_lang=tgt_lang,
    )


@download.command()
@click.argument("target_dir", type=click.Path())
@click.option(
    "--subset",
    type=click.Choice(["asr", "10k", "100k", "400k"] + LANGUAGES + LANGUAGES_V2),
    default="asr",
)
def voxpopuli(target_dir: Pathlike, subset: str):
    """voxpopuli download."""
    download_voxpopuli(target_dir, subset)
