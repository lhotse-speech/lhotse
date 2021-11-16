import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_voxceleb
from lhotse.utils import Pathlike

from typing import Optional


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
    voxceleb1_root: Optional[Pathlike],
    voxceleb2_root: Optional[Pathlike],
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
        voxceleb1_root=voxceleb1_root,
        voxceleb2_root=voxceleb2_root,
        output_dir=output_dir,
        num_jobs=num_jobs,
    )
