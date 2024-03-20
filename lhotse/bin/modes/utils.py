import click

from .cli_base import cli


@cli.command()
def list_audio_backends():
    """
    List the names of all available audio backends.
    """
    from lhotse import available_audio_backends

    click.echo(available_audio_backends())
