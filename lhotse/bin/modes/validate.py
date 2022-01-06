from pathlib import Path

import click

from lhotse.bin.modes.cli_base import cli
from lhotse.utils import Pathlike


@cli.command(name="validate")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--read-data/--dont-read-data",
    default=False,
    help="Should the audio/features data be read from disk to perform additional checks "
    "(could be extremely slow for large manifests).",
)
def validate_(manifest: Pathlike, read_data: bool):
    """Validate a Lhotse manifest file."""
    from lhotse import load_manifest, validate

    data = load_manifest(manifest)
    validate(data, read_data=read_data)


@cli.command(name="validate-pair")
@click.argument("recordings", type=click.Path(exists=True, dir_okay=False))
@click.argument("supervisions", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--read-data/--dont-read-data",
    default=False,
    help="Should the audio/features data be read from disk to perform additional checks "
    "(could be extremely slow for large manifests).",
)
def validate_(recordings: Pathlike, supervisions: Pathlike, read_data: bool):
    """
    Validate a pair of Lhotse RECORDINGS and SUPERVISIONS manifest files.
    Checks whether the two manifests are consistent with each other.
    """
    from lhotse import load_manifest, validate_recordings_and_supervisions

    recs = load_manifest(recordings)
    sups = load_manifest(supervisions)
    validate_recordings_and_supervisions(
        recordings=recs, supervisions=sups, read_data=read_data
    )


@cli.command(name="fix")
@click.argument("recordings", type=click.Path(exists=True, dir_okay=False))
@click.argument("supervisions", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path())
def fix_(recordings: Pathlike, supervisions: Pathlike, output_dir: Pathlike):
    """
    Fix a pair of Lhotse RECORDINGS and SUPERVISIONS manifests.
    It removes supervisions without corresponding recordings and vice versa,
    trims the supervisions that exceed the recording, etc.
    Stores the output files in OUTPUT_DIR under the same names as the input
    files.
    """
    from lhotse import RecordingSet, SupervisionSet, fix_manifests

    output_dir = Path(output_dir)
    recordings = Path(recordings)
    supervisions = Path(supervisions)
    output_dir.mkdir(parents=True, exist_ok=True)
    recs = RecordingSet.from_file(recordings)
    sups = SupervisionSet.from_file(supervisions)
    recs, sups = fix_manifests(recordings=recs, supervisions=sups)
    recs.to_file(output_dir / recordings.name)
    sups.to_file(output_dir / supervisions.name)
