import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_rir_noise, prepare_rir_noise
from lhotse.utils import Pathlike

from typing import Sequence, Union

__all__ = ["rir_noise"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--parts",
    "-p",
    type=str,
    multiple=True,
    default=["point_noise", "iso_noise", "real_rir", "sim_rir"],
    show_default=True,
    help="Parts to prepare.",
)
def rir_noise(
    corpus_dir: Pathlike, output_dir: Pathlike, parts: Union[str, Sequence[str]]
):
    """RIRS and noises data preparation."""
    prepare_rir_noise(corpus_dir, output_dir=output_dir, parts=parts)


@download.command()
@click.argument("target_dir", type=click.Path())
def rir_noise(target_dir: Pathlike):
    """RIRS and noises download."""
    download_rir_noise(target_dir)
