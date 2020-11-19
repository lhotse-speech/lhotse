from pathlib import Path

import click

from lhotse import load_manifest
from lhotse.bin.modes.cli_base import cli
from lhotse.kaldi import export_to_kaldi, load_kaldi_data_dir
from lhotse.utils import Pathlike


@cli.group()
def kaldi():
    """Kaldi import/export related commands."""
    pass


@kaldi.command(name='import')
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('sampling_rate', type=int)
@click.argument('manifest_dir', type=click.Path())
def import_(data_dir: Pathlike, sampling_rate: int, manifest_dir: Pathlike):
    """
    Convert a Kaldi data dir DATA_DIR into a directory MANIFEST_DIR of lhotse manifests. Ignores feats.scp.
    The SAMPLING_RATE has to be explicitly specified as it is not available to read from DATA_DIR.
    """
    recording_set, maybe_supervision_set = load_kaldi_data_dir(path=data_dir, sampling_rate=sampling_rate)
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    recording_set.to_json(manifest_dir / 'audio.json')
    if maybe_supervision_set is not None:
        maybe_supervision_set.to_json(manifest_dir / 'supervision.json')


@kaldi.command()
@click.argument('recordings', type=click.Path(exists=True, dir_okay=False))
@click.argument('supervisions', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path())
def export(recordings: Pathlike, supervisions: Pathlike, output_dir: Pathlike):
    """
    Convert a pair of ``RecordingSet`` and ``SupervisionSet`` manifests into a Kaldi-style data directory.
    """
    export_to_kaldi(
        recordings=load_manifest(recordings),
        supervisions=load_manifest(supervisions),
        output_dir=output_dir
    )
