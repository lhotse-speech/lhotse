import click

from .cli_base import cli


@cli.command()
def list_audio_backends():
    """
    List the names of all available audio backends.
    """
    from lhotse import available_audio_backends

    click.echo(available_audio_backends())


@cli.command()
def list_io_backends():
    """
    List the names of all available IO backends.
    """
    from lhotse import available_io_backends

    click.echo(available_io_backends())


@cli.command()
def list_resampling_backends():
    """
    List the names of all available resampling backends.
    """
    from lhotse import available_resampling_backends

    click.echo(available_resampling_backends())


@cli.command()
def list_storage_backends():
    """
    List all known feature/array storage backends and mark unavailable ones.
    """
    from lhotse import storage_backend_statuses

    for backend in storage_backend_statuses():
        line = backend.name
        if not backend.available:
            line += f" (unavailable, requires: {backend.install_hint})"
        click.echo(line)
