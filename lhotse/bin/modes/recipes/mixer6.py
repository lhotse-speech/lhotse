import click

from lhotse.bin.modes import prepare
from lhotse.recipes.mixer6 import prepare_mixer6
from lhotse.utils import Pathlike

__all__ = ["mixer6"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("transcript_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--part", type=click.Choice(["call", "intv"]), default="intv")
def mixer6(
    corpus_dir: Pathlike, transcript_dir: Pathlike, output_dir: Pathlike, part: str
):
    """Mixer 6 data preparation."""
    prepare_mixer6(
        corpus_dir, transcript_dir=transcript_dir, output_dir=output_dir, part=part
    )
