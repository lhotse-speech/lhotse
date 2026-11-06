import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.mobvoihotwords import (
    download_mobvoihotwords,
    prepare_mobvoihotwords,
)
from lhotse.utils import Pathlike

__all__ = ["mobvoihotwords"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def mobvoihotwords(corpus_dir: Pathlike, output_dir: Pathlike):
    """MobvoiHotwords data preparation."""
    prepare_mobvoihotwords(corpus_dir, output_dir=output_dir)


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    type=bool,
    is_flag=True,
    default=False,
    help="Force download.",
)
def mobvoihotwords(target_dir: Pathlike, force_download: bool):
    """MobvoiHotwords download."""
    download_mobvoihotwords(target_dir, force_download=force_download)
