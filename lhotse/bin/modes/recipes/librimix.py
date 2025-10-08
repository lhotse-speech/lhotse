import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.librimix import download_librimix, prepare_librimix
from lhotse.utils import Pathlike

__all__ = ["librimix"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("librispeech_root_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("wham_recset_root_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("librimix_metadata_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("workdir", type=click.Path(exists=False, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-n",
    "--n_src",
    type=click.Choice(["2", "3"], case_sensitive=False),
    default="2",
    help="Number of sources used to create mixtures.",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def librimix(
    librispeech_root_path: Pathlike,
    wham_recset_root_path: Pathlike,
    librimix_metadata_path: Pathlike,
    workdir: Pathlike,
    output_dir: Pathlike,
    n_src: str,
    num_jobs: int,
):
    """LibrMix source separation data preparation."""
    prepare_librimix(
        librispeech_root_path=librispeech_root_path,
        wham_recset_root_path=wham_recset_root_path,
        librimix_metadata_path=librimix_metadata_path,
        workdir=workdir,
        output_dir=output_dir,
        n_src=int(n_src),
        num_jobs=num_jobs,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def librimix(target_dir: Pathlike):
    """Mini LibriMix download."""
    download_librimix(target_dir)
