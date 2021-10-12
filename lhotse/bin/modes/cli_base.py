import logging

import click


@click.group()
@click.option("-s", "--seed", type=int, help="Random seed.")
def cli(seed):
    """
    The shell entry point to Lhotse, a tool and a library for audio data manipulation in high altitudes.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    if seed is not None:
        from lhotse.utils import fix_random_seed

        fix_random_seed(seed)


@cli.group()
def prepare():
    """Command group with data preparation recipes."""
    pass


@cli.group()
def download():
    """Command group for download and extract data."""
    pass
