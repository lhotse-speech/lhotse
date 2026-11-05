from pathlib import Path

import click

from lhotse.bin.modes.cli_base import cli


@cli.group()
def index():
    """Create binary index files for O(1) random-access reads."""
    pass


def _output_index_path(path: str, output_dir: str):
    if output_dir is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (Path(path).name + ".idx")


def _create_single_index(path: str, output_dir: str, create_index_fn):
    idx_path = create_index_fn(path, output_path=_output_index_path(path, output_dir))
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
def jsonl(path: str, output_dir: str):
    """
    Create a binary index for a plain or gzip JSONL file.

    The index file is written next to the input as ``<path>.idx``,
    unless ``--output-dir`` is specified. Gzip inputs also create a
    ``<path>.gzidx`` seek index in the same location.
    """
    from lhotse.indexing import create_jsonl_index

    _create_single_index(path, output_dir, create_jsonl_index)


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

    _create_single_index(path, output_dir, create_tar_index)


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
    Gzip JSONL is indexed when ``indexed_gzip`` is installed; compressed tar
    files are skipped.
    """
    from lhotse.indexing import create_shar_index

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    create_shar_index(shar_dir, output_dir=output_dir)
    click.echo(f"Created indexes for Shar directory: {shar_dir}")


@index.command(name="shar-pack")
@click.argument("output", type=click.Path(dir_okay=False))
@click.argument(
    "shar_dirs", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option(
    "--indexes-root",
    type=click.Path(file_okay=False),
    default=None,
    help="Look up the .idx files under this root, which mirrors the layout of the data, instead of next to the data.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Replace OUTPUT if it already exists.",
)
def shar_pack(output: str, shar_dirs, indexes_root: str, overwrite: bool):
    """
    Pack the .idx files of one or more Shar directories into a single .idxpack.

    The indexes must exist already (see ``lhotse index shar``). A Shar reader is pointed
    at the pack with ``CutSet.from_shar(in_dir=SHAR_DIR, index_pack=OUTPUT)``; each reader
    finds its directory in the pack by the same path string it was packed with.
    Tar shards must be uncompressed; their sidecars are validated while packing.
    """
    from lhotse.shar import write_shar_index_pack

    write_shar_index_pack(
        output, list(shar_dirs), indexes_root=indexes_root, overwrite=overwrite
    )
    click.echo(
        f"Created index pack for {len(shar_dirs)} Shar director{'y' if len(shar_dirs) == 1 else 'ies'}: {output}"
    )
