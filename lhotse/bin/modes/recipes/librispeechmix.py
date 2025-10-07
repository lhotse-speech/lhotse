import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.librispeechmix import (
    download_librispeechmix,
    prepare_librispeechmix,
)
from lhotse.utils import Pathlike

__all__ = ["librispeechmix"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("librispeech_root_path", type=click.Path(exists=True, dir_okay=True))
@click.argument(
    "librispeechmix_metadata_path", type=click.Path(exists=True, dir_okay=True)
)
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use.",
)
def librispeechmix(
    librispeech_root_path: Pathlike,
    librispeechmix_metadata_path: Pathlike,
    output_dir: Pathlike,
    num_jobs: int,
):
    prepare_librispeechmix(
        librispeech_root_path=librispeech_root_path,
        librispeechmix_metadata_path=librispeechmix_metadata_path,
        output_dir=output_dir,
        num_jobs=num_jobs,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def librispeechmix(target_dir: Pathlike):
    download_librispeechmix(target_dir)
