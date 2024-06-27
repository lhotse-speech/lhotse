from typing import Optional, Sequence, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.callhome_spanish_st import prepare_callhome_spanish
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("audio_dir_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("transcript_dir_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("split_dir", type=click.Path())
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
@click.option(
    "--absolute_paths",
    default=False,
    help=" Whether to return absolute or relative (to the corpus dir) paths for recordings.",
)
@click.option(
    "--remove_punc",
    default=False,
    help=" Remove punctuations from the text",
)
@click.option(
    "--lowercase",
    default=False,
    help="Lower case the text",
)
def callhome_spanish_st(
    audio_dir_path: Pathlike,
    transcript_dir_path: Pathlike,
    split_dir: Pathlike,
    output_dir: Pathlike,
    absolute_paths: bool,
    remove_punc: bool,
    lowercase: bool,
    num_jobs: int,
):
    """
    Callhome Spanish data preparation.
    \b
    This is conversational telephone speech collected as 2-channel μ-law, 8kHz-sampled data.
    The catalog number LDC96S35 for audio corpus and LDC96T17 for transcripts.

    This data is not available for free - your institution needs to have an LDC subscription.
    You should also download and prepare the pre-defined splits with:
        git clone https://github.com/joshua-decoder/fisher-callhome-corpus.git
        cd fisher-callhome-corpus
        make
        cd ../
    """
    prepare_callhome_spanish(
        audio_dir_path=audio_dir_path,
        transcript_dir_path=transcript_dir_path,
        split_dir=split_dir,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
        remove_punc=remove_punc,
        lowercase=lowercase,
        num_jobs=num_jobs,
    )
