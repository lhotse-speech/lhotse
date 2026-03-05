from pathlib import Path

import click

from lhotse.bin.modes.cli_base import cli


@cli.group()
def index():
    """Create binary index files for O(1) random-access reads."""
    pass


@index.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Write the .idx file into this directory instead of next to the input.",
)
def jsonl(path: str, output_dir: str):
    """
    Create a binary index for an uncompressed JSONL file.

    The index file is written next to the input as ``<path>.idx``,
    unless ``--output-dir`` is specified.
    """
    from lhotse.indexing import create_jsonl_index

    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir) / (Path(path).name + ".idx")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    idx_path = create_jsonl_index(path, output_path=output_path)
    click.echo(f"Created index: {idx_path}")


@index.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Write the .idx file into this directory instead of next to the input.",
)
def tar(path: str, output_dir: str):
    """
    Create a binary index for an uncompressed tar archive.

    The index file is written next to the input as ``<path>.idx``,
    unless ``--output-dir`` is specified.
    """
    from lhotse.indexing import create_tar_index

    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir) / (Path(path).name + ".idx")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    idx_path = create_tar_index(path, output_path=output_path)
    click.echo(f"Created index: {idx_path}")


@index.command()
@click.argument("shar_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Write .idx files into this directory instead of next to the data files.",
)
def shar(shar_dir: str, output_dir: str):
    """
    Create binary indexes for all JSONL and tar files in a Shar directory.

    Indexes are written next to each data file as ``<file>.idx``,
    unless ``--output-dir`` is specified.
    Compressed files (``.jsonl.gz``, ``.tar.gz``) are skipped.
    """
    from lhotse.indexing import create_shar_index

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    create_shar_index(shar_dir, output_dir=output_dir)
    click.echo(f"Created indexes for Shar directory: {shar_dir}")
