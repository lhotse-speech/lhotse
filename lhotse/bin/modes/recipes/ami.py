import click

from lhotse.bin.modes import obtain, prepare
from lhotse.recipes.ami import download, prepare_ami
from lhotse.utils import Pathlike

__all__ = ['ami']


@prepare.command(context_settings=dict(show_default=True))
@click.argument('corpus_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
@click.option('-mic', type=click.Choice(['ihm','ihm-mix','sdm','mdm'], case_sensitive=False),
              default='ihm', help='AMI microphone setting.')
@click.option('-partition', type=click.Choice(['scenario-only','full-corpus','full-corpus-asr'],case_sensitive=False),
              default='full-corpus-asr',
              help='Data partition to use (see http://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml).')
@click.option('-max-pause', type=float, default=0.0, help='Max pause allowed between word segments to combine segments.')
def ami(
        corpus_dir: Pathlike,
        output_dir: Pathlike,
        mic: str,
        partition: str,
        max_pause: float
):
    """AMI data preparation."""
    prepare_ami(corpus_dir, output_dir=output_dir, mic=mic, partition=partition, max_pause=max_pause)


@obtain.command(context_settings=dict(show_default=True))
@click.argument('target_dir', type=click.Path())
@click.option('-mic', type=click.Choice(['ihm','ihm-mix','sdm','mdm'], case_sensitive=False),
              default='ihm', help='AMI microphone setting.')
@click.option('-url', type=str, default='http://groups.inf.ed.ac.uk/ami',
              help='AMI data downloading URL.')
@click.option('-force-download', type=bool, default=False,
              help='If True, download even if file is present.')
def ami(
        target_dir: Pathlike,
        mic: str,
        url: str,
        force_download: bool
):
    """AMI download."""
    download(target_dir, mic=mic, url=url, force_download=force_download)
