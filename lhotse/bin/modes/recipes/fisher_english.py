from typing import List

import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_fisher_english
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output-dir", type=click.Path())
@click.option(
    "-ad",
    "--audio-dirs",
    type=str,
    multiple=True,
    default=["LDC2004S13", "LDC2005S13"],
    help="Audio dirs, e.g., `LDC2004S13 LDC2005S13`. Multiple corpora can be provided by repeating `-ad`.",
)
@click.option(
    "-td",
    "--transcript-dirs",
    type=str,
    multiple=True,
    default=["LDC2004T19", "LDC2005T19"],
    help="Transcript dirs, e.g., `LDC2004T19 LDC2005T19`. Multiple corpora can be provided by repeating `-ad`.",
)
@click.option(
    "--absolute-paths",
    default=False,
    help="Whether to return absolute or relative (to the corpus dir) paths for recordings.",
)
@click.option(
    "-j",
    "--num-jobs",
    default=1,
    type=int,
    help="Number of concurrent processes scanning the audio files.",
)
def fisher_english(
    corpus_dir: Pathlike,
    audio_dirs: List[str],
    transcript_dirs: List[str],
    output_dir: Pathlike,
    absolute_paths: bool,
    num_jobs: int,
):
    """
    The Fisher English Part 1, 2 corpus preparation.

    \b
    This is conversational telephone speech collected as 2-channel, 8kHz-sampled data.
    The catalog number LDC2004S13 and LDC2005S13 for audio corpora and LDC2004T19 LDC2005T19 for transcripts.

    This data is not available for free - your institution needs to have an LDC subscription.
    """
    prepare_fisher_english(
        corpus_dir=corpus_dir,
        audio_dirs=audio_dirs,
        transcript_dirs=transcript_dirs,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
        num_jobs=num_jobs,
    )
