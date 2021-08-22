import click
from lhotse.bin.modes import download, prepare
from lhotse.recipes.yesno import download_yesno, prepare_yesno
from lhotse.utils import Pathlike


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def yesno(target_dir: Pathlike):
    """yes_no dataset download."""
    download_yesno(target_dir)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def yesno(corpus_dir: Pathlike, output_dir: Pathlike):
    """yes_no data preparation."""
    prepare_yesno(corpus_dir, output_dir=output_dir)
