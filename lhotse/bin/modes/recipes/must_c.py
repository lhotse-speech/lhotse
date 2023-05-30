import logging

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.must_c import prepare_must_c
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--tgt-lang", type=str, help="The target language, e.g., zh, de, fr.")
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def must_c(corpus_dir: Pathlike, output_dir: Pathlike, tgt_lang, num_jobs: int):
    """MUST-C speech translation data preparation."""
    logging.basicConfig(level=logging.INFO)
    prepare_must_c(
        corpus_dir,
        output_dir=output_dir,
        tgt_lang=tgt_lang,
        num_jobs=num_jobs,
    )
