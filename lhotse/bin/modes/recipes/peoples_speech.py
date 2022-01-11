import click

from lhotse.bin.modes import prepare
from lhotse.recipes.peoples_speech import prepare_peoples_speech
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def peoples_speech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """Prepare The People's Speech corpus manifests."""
    prepare_peoples_speech(
        corpus_dir,
        output_dir=output_dir,
    )
