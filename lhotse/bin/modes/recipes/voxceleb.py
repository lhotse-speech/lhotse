import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_voxceleb1, download_voxceleb2, prepare_voxceleb
from lhotse.utils import Pathlike

from typing import Optional


@download.command()
@click.argument("target_dir", type=click.Path())
@click.option("--force-download", is_flag=True, default=False, help="Force download")
def voxceleb1(target_dir: Pathlike, force_download: Optional[bool] = False):
    """VoxCeleb1 download."""
    download_voxceleb1(target_dir, force_download=force_download)


@download.command()
@click.argument("target_dir", type=click.Path())
@click.option("--force-download", is_flag=True, default=False, help="Force download")
def voxceleb2(target_dir: Pathlike, force_download: Optional[bool] = False):
    """VoxCeleb2 download."""
    download_voxceleb2(target_dir, force_download=force_download)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("output-dir", type=click.Path())
@click.option(
    "--voxceleb1",
    "-v1",
    type=click.Path(),
    default=None,
    help="Path to VoxCeleb1 dataset.",
)
@click.option(
    "--voxceleb2",
    "-v2",
    type=click.Path(),
    default=None,
    help="Path to VoxCeleb2 dataset.",
)
@click.option("--num-jobs", "-j", type=int, default=1, help="Number of parallel jobs.")
def voxceleb(
    output_dir: Pathlike,
    voxceleb1: Optional[Pathlike],
    voxceleb2: Optional[Pathlike],
    num_jobs: int = 1,
):
    """
    The VoxCeleb corpus preparation.

    VoxCeleb is an audio-visual dataset consisting of short clips of human speech, extracted
    from interview videos uploaded to YouTube. VoxCeleb contains speech from speakers spanning
    a wide range of different ethnicities, accents, professions and ages. There are a total of
    7000+ speakers and 1 million utterances.
    """
    prepare_voxceleb(
        voxceleb1_root=voxceleb1,
        voxceleb2_root=voxceleb2,
        output_dir=output_dir,
        num_jobs=num_jobs,
    )
