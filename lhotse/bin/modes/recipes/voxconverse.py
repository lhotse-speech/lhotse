import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.voxconverse import download_voxconverse, prepare_voxconverse
from lhotse.utils import Pathlike


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option("--force-download", is_flag=True, default=False, help="Force download")
def voxconverse(target_dir: Pathlike, force_download=False):
    """VoxConverse dataset download."""
    download_voxconverse(target_dir, force_download=force_download)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--split-test",
    is_flag=True,
    default=False,
    help="Split test part into dev and test parts",
)
def voxconverse(corpus_dir: Pathlike, output_dir: Pathlike, split_test: bool = False):
    """VoxConverse data preparation."""
    prepare_voxconverse(corpus_dir, output_dir=output_dir, split_test=split_test)
