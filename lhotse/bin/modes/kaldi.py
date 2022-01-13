from pathlib import Path
from typing import Optional

import click

from lhotse.bin.modes.cli_base import cli
from lhotse.utils import Pathlike


@cli.group()
def kaldi():
    """Kaldi import/export related commands."""
    pass


@kaldi.command(name="import")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("sampling_rate", type=int)
@click.argument("manifest_dir", type=click.Path())
@click.option(
    "-f",
    "--frame-shift",
    type=float,
    help="Frame shift (in seconds) is required to support reading feats.scp.",
)
@click.option(
    "-u",
    "--map-string-to-underscores",
    default=None,
    type=str,
    help="""When specified, we will replace all instances of this string 
    in SupervisonSegment IDs to underscores. This is to help with handling 
    underscores in Kaldi (see 'export_to_kaldi').""",
)
@click.option(
    "-j",
    "--num-jobs",
    default=1,
    type=int,
    help="Number of jobs for computing recording durations.",
)
def import_(
    data_dir: Pathlike,
    sampling_rate: int,
    manifest_dir: Pathlike,
    frame_shift: float,
    map_string_to_underscores: Optional[str],
    num_jobs: int,
):
    """
    Convert a Kaldi data dir DATA_DIR into a directory MANIFEST_DIR of lhotse manifests. Ignores feats.scp.
    The SAMPLING_RATE has to be explicitly specified as it is not available to read from DATA_DIR.
    """
    from lhotse.kaldi import load_kaldi_data_dir

    recording_set, maybe_supervision_set, maybe_feature_set = load_kaldi_data_dir(
        path=data_dir,
        sampling_rate=sampling_rate,
        frame_shift=frame_shift,
        map_string_to_underscores=map_string_to_underscores,
        num_jobs=num_jobs,
    )
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    recording_set.to_file(manifest_dir / "recordings.jsonl.gz")
    if maybe_supervision_set is not None:
        maybe_supervision_set.to_file(manifest_dir / "supervisions.jsonl.gz")
    if maybe_feature_set is not None:
        maybe_feature_set.to_file(manifest_dir / "features.jsonl.gz")


@kaldi.command()
@click.argument("recordings", type=click.Path(exists=True, dir_okay=False))
@click.argument("supervisions", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-u",
    "--map-underscores-to",
    type=str,
    default=None,
    help=(
        "Optional string with which we will replace all underscores."
        "This helps avoid issues with Kaldi data dir sorting."
    ),
)
@click.option(
    "-p",
    "--prefix-spk-id",
    is_flag=True,
    help=(
        "Prefix utterance ids with speaker ids."
        "This helps avoid issues with Kaldi data dir sorting."
    ),
)
def export(
    recordings: Pathlike,
    supervisions: Pathlike,
    output_dir: Pathlike,
    map_underscores_to: Optional[str],
    prefix_spk_id: Optional[bool] = False,
):
    """
    Convert a pair of ``RecordingSet`` and ``SupervisionSet`` manifests into a Kaldi-style data directory.
    """
    from lhotse import load_manifest
    from lhotse.kaldi import export_to_kaldi

    output_dir = Path(output_dir)
    export_to_kaldi(
        recordings=load_manifest(recordings),
        supervisions=load_manifest(supervisions),
        output_dir=output_dir,
        map_underscores_to=map_underscores_to,
        prefix_spk_id=prefix_spk_id,
    )
    click.secho(
        "Export completed! You likely need to run the following Kaldi commands:",
        bold=True,
        fg="yellow",
    )
    click.secho(
        f"  utils/utt2spk_to_spk2utt.pl {output_dir}/utt2spk > {output_dir}/spk2utt",
        fg="yellow",
    )
    click.secho(f"  utils/fix_data_dir.sh {output_dir}", fg="yellow")
