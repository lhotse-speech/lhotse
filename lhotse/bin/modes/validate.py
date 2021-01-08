import click

from lhotse import load_manifest, validate
from lhotse.bin.modes.cli_base import cli
from lhotse.utils import Pathlike


@cli.command(name='validate')
@click.argument('manifest', type=click.Path(exists=True, dir_okay=False))
@click.option('--read-data/--dont-read-data', default=False,
              help='Should the audio/features data be read from disk to perform additional checks '
                   '(could be extremely slow for large manifests).')
def validate_(manifest: Pathlike, read_data: bool):
    """Validate a Lhotse manifest file."""
    data = load_manifest(manifest)
    validate(data, read_data=read_data)
