import click

from lhotse.bin.modes import prepare
from lhotse.recipes.ali_meeting import prepare_ali_meeting
from lhotse.utils import Pathlike

__all__ = ["ali_meeting"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--mic", type=click.Choice(["near", "far"]), default="far")
def ali_meeting(corpus_dir: Pathlike, output_dir: Pathlike, mic: str):
    """AliMeeting data preparation."""
    prepare_ali_meeting(corpus_dir, output_dir=output_dir, mic=mic)
