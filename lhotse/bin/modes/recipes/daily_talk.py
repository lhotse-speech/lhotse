import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.daily_talk import download_daily_talk, prepare_daily_talk
from lhotse.utils import Pathlike


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def daily_talk(corpus_dir: Pathlike, output_dir: Pathlike):
    """
    DailyTalk recording and supervision manifest preparation.
    """
    prepare_daily_talk(corpus_dir, output_dir)


@download.command()
@click.argument("target_dir", type=click.Path(exists=True, dir_okay=True))
@click.option("--force-download", is_flag=True, help="Force download")
def daily_talk(target_dir: Pathlike, force_download: bool = False):
    """
    Download DailyTalk dataset.
    """
    download_daily_talk(target_dir, force_download)
