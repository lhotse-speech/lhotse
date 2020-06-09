import click

__all__ = ['cli', 'recipe']


@click.group()
def cli():
    """
    The shell entry point to Lhotse, a tool and a library for audio data manipulation in high altitudes.
    """
    pass


@cli.group()
def recipe():
    """Command group with data preparation recipes."""
    pass
