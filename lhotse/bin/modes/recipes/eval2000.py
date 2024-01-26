from typing import Optional

import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_eval2000
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output-dir", type=click.Path())
@click.option(
    "--transcript-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    required=False,
)
@click.option(
    "--absolute-paths",
    default=False,
    help="Whether to return absolute or relative (to the corpus dir) paths for recordings.",
)
def eval2000(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    absolute_paths: bool,
    transcript_dir: Optional[Pathlike] = None,
):
    """
    The Eval2000 corpus preparation.

    \b
    This is conversational telephone speech collected as 2-channel, 8kHz-sampled data.
    The catalog number LDC2002S09 for audio corpora and LDC2002T43 for transcripts.

    This data is not available for free - your institution needs to have an LDC subscription.
    """

    prepare_eval2000(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
        transcript_path=transcript_dir,
    )
