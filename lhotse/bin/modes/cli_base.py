import click

__all__ = ['cli', 'prepare', 'obtain']


@click.group()
def cli():
    """
    The shell entry point to Lhotse, a tool and a library for audio data manipulation in high altitudes.
    """
    pass


@cli.group()
def prepare():
    """Command group with data preparation recipes."""
    pass


@cli.group()
def obtain():
    """Command group for download and extract data."""
    pass
