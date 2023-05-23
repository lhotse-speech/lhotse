from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.chime6 import download_chime6, prepare_chime6
from lhotse.utils import Pathlike

__all__ = ["chime6"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=["all"],
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p dev -p eval`. By default, all parts are prepared.",
)
@click.option(
    "--mic",
    type=click.Choice(["ihm", "mdm"], case_sensitive=False),
    default="mdm",
    help="CHiME-6 microphone setting.",
)
@click.option(
    "--use-reference-array",
    is_flag=True,
    default=False,
    help="If True, use the reference array for the MDM setting. Only the supervision "
    "segments have the reference array information in the `channel` field. The "
    "recordings will still have all the channels in the array. Note that reference "
    "array is not available for the train set.",
)
@click.option(
    "--perform-array-sync",
    is_flag=True,
    default=False,
    help="If True, perform array synchronization for the MDM setting.",
)
@click.option(
    "--verify-md5-checksums",
    is_flag=True,
    default=False,
    help="If True, verify the MD5 checksums of the audio files. This is useful to "
    "ensure correct array synchronization. Note that this step is slow, so we recommend "
    "only doing it once. You can speed it up by using more jobs.",
)
@click.option(
    "--num-jobs",
    "-j",
    type=int,
    default=1,
    help="Number of parallel jobs to run for array synchronization.",
)
@click.option(
    "--num-threads-per-job",
    "-t",
    type=int,
    default=1,
    help="Number of threads to use per job for clock drift correction. Large values "
    "may require more memory, so we recommend using a job scheduler.",
)
@click.option(
    "--sox-path",
    type=click.Path(exists=True, dir_okay=False),
    default="/usr/bin/sox",
    help="Path to the sox binary. This must be v14.4.2.",
    show_default=True,
)
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "upper", "kaldi"], case_sensitive=False),
    default="kaldi",
    help="Text normalization method.",
    show_default=True,
)
@click.option(
    "--use-chime7-split",
    is_flag=True,
    default=False,
    help="If True, use the new splits from CHiME-7 challenge.",
)
def chime6(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    mic: str,
    use_reference_array: bool,
    perform_array_sync: bool,
    verify_md5_checksums: bool,
    num_jobs: int,
    num_threads_per_job: int,
    sox_path: Pathlike,
    normalize_text: str,
    use_chime7_split: bool,
):
    """CHiME-6 data preparation."""
    prepare_chime6(
        corpus_dir,
        output_dir=output_dir,
        dataset_parts=dataset_parts,
        mic=mic,
        use_reference_array=use_reference_array,
        perform_array_sync=perform_array_sync,
        verify_md5_checksums=verify_md5_checksums,
        num_jobs=num_jobs,
        num_threads_per_job=num_threads_per_job,
        sox_path=sox_path,
        normalize_text=normalize_text,
        use_chime7_split=use_chime7_split,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def chime6(target_dir: Pathlike):
    """CHiME-6 download."""
    download_chime6(target_dir)
