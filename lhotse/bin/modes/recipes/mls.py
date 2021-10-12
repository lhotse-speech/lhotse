import click

from lhotse.bin.modes import prepare
from lhotse.recipes.mls import prepare_mls
from lhotse.utils import Pathlike

__all__ = ["mls"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--opus/--flac",
    type=bool,
    default=True,
    help="Which codec should be used (OPUS or FLAC)",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def mls(corpus_dir: Pathlike, output_dir: Pathlike, opus: bool, num_jobs: int):
    """
    Multilingual Librispeech (MLS) data preparation.

    Multilingual LibriSpeech (MLS) dataset is a large multilingual corpus suitable for speech research.
    The dataset is derived from read audiobooks from LibriVox and consists of 8 languages -
    English, German, Dutch, Spanish, French, Italian, Portuguese, Polish.
    It is available at OpenSLR: http://openslr.org/94
    """
    prepare_mls(corpus_dir, opus=opus, output_dir=output_dir, num_jobs=num_jobs)
