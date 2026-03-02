from pathlib import Path

import click

from lhotse.bin.modes.cli_base import cli


@cli.group()
def index():
    """Create binary index files for O(1) random-access reads."""
    pass


@index.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def jsonl(path: str):
    """
    Create a binary index for an uncompressed JSONL file.

    The index file is written next to the input as ``<path>.idx``.
    """
    from lhotse.indexing import create_jsonl_index

    idx_path = create_jsonl_index(path)
    click.echo(f"Created index: {idx_path}")


@index.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def tar(path: str):
    """
    Create a binary index for an uncompressed tar archive.

    The index file is written next to the input as ``<path>.idx``.
    """
    from lhotse.indexing import create_tar_index

    idx_path = create_tar_index(path)
    click.echo(f"Created index: {idx_path}")


@index.command()
@click.argument("shar_dir", type=click.Path(exists=True, file_okay=False))
def shar(shar_dir: str):
    """
    Create binary indexes for all JSONL and tar files in a Shar directory.

    Indexes are written next to each data file as ``<file>.idx``.
    Compressed files (``.jsonl.gz``, ``.tar.gz``) are skipped.
    """
    from lhotse.indexing import create_shar_index

    create_shar_index(shar_dir)
    click.echo(f"Created indexes for Shar directory: {shar_dir}")
