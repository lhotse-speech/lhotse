import click

from lhotse.utils import fix_random_seed


@click.group()
@click.option('-s', '--seed', type=int, help='Random seed.')
def cli(seed):
    """
    The shell entry point to Lhotse, a tool and a library for audio data manipulation in high altitudes.
    """
    if seed is not None:
        fix_random_seed(seed)


@cli.group()
def prepare():
    """Command group with data preparation recipes."""
    pass


@cli.group()
def obtain():
    """Command group for download and extract data."""
    pass
