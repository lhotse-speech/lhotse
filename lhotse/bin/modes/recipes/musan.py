import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.musan import download_musan, prepare_musan
from lhotse.utils import Pathlike

__all__ = ["musan"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--use-vocals/--no-vocals",
    default=True,
    help='Whether to include vocal music in "music" part.',
)
def musan(corpus_dir: Pathlike, output_dir: Pathlike, use_vocals: bool):
    """MUSAN data preparation."""
    prepare_musan(corpus_dir, output_dir=output_dir, use_vocals=use_vocals)


@download.command()
@click.argument("target_dir", type=click.Path())
def musan(target_dir: Pathlike):
    """MUSAN download."""
    download_musan(target_dir)
