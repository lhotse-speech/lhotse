from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from lhotse import CutSet, RecordingSet
from lhotse.bin.modes.cli_base import cli
from lhotse.utils import exactly_one_not_null


@cli.group()
def workflows():
    """Workflows using corpus creation tools."""
    pass


@workflows.command()
@click.argument("out_cuts", type=click.Path())
@click.option(
    "-m",
    "--recordings-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an existing recording manifest.",
)
@click.option(
    "-r",
    "--recordings-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Directory with recordings. We will create a RecordingSet for it automatically.",
)
@click.option(
    "-e",
    "--extension",
    default="wav",
    help="Audio file extension to search for. Used with RECORDINGS_DIR.",
)
@click.option(
    "-n",
    "--model-name",
    default="base",
    help="One of Whisper variants (base, medium, large, etc.)",
)
@click.option(
    "-l",
    "--language",
    help="Language spoken in the audio. Inferred by default.",
)
@click.option(
    "-d", "--device", default="cpu", help="Device on which to run the inference."
)
@click.option("-j", "--jobs", default=1, help="Number of jobs for audio scanning.")
def annotate_with_whisper(
    out_cuts: str,
    recordings_manifest: Optional[str],
    recordings_dir: Optional[str],
    extension: str,
    model_name: str,
    language: Optional[str],
    device: str,
    jobs: int,
):
    """
    Use OpenAI Whisper model to annotate either RECORDINGS_MANIFEST or RECORDINGS_DIR.
    It will perform automatic segmentation, transcription, and language identification.

    RECORDINGS_MANIFEST and RECORDINGS_DIR are mutually exclusive.

    Note: this is an experimental feature of Lhotse, and is not guaranteed to yield
    high quality of data.
    """
    from lhotse import annotate_with_whisper as annotate_with_whisper_

    assert exactly_one_not_null(recordings_manifest, recordings_dir), (
        "Options RECORDINGS_MANIFEST and RECORDINGS_DIR are mutually exclusive "
        "and at least one is required."
    )

    if recordings_manifest is not None:
        recordings = RecordingSet.from_file(recordings_manifest)
    else:
        recordings = RecordingSet.from_dir(
            recordings_dir, pattern=f"*.{extension}", num_jobs=jobs
        )

    with CutSet.open_writer(out_cuts) as writer:
        for cut in tqdm(
            annotate_with_whisper_(
                recordings, language=language, model_name=model_name, device=device
            ),
            total=len(recordings),
            desc="Recordings",
        ):
            writer.write(cut, flush=True)
