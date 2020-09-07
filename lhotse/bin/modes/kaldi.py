from pathlib import Path

import click

from lhotse.bin.modes.cli_base import cli
from lhotse.kaldi import load_kaldi_data_dir
from lhotse.utils import Pathlike

__all__ = ['convert_kaldi']


@cli.command()
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('sampling_rate', type=int)
@click.argument('manifest_dir', type=click.Path())
def convert_kaldi(data_dir: Pathlike, sampling_rate: int, manifest_dir: Pathlike):
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
