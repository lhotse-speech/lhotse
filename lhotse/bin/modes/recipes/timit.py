import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.timit import download_timit, prepare_timit
from lhotse.utils import Pathlike

__all__ = ["timit"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--num-phones",
    type=int,
    default=48,
    help="The number of phones (60, 48 or 39) for modeling. "
    "And 48 is regarded as the default value.",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def timit(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    num_phones: int,
    num_jobs: int = 1,
):
    """TIMIT data preparation.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write and save the manifests.
    """
    prepare_timit(
        corpus_dir,
        output_dir=output_dir,
        num_phones=num_phones,
        num_jobs=num_jobs,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def timit(target_dir: Pathlike):
    """TIMIT download."""
    download_timit(target_dir)
