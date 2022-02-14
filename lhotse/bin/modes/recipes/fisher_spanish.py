import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_fisher_spanish
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("audio-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("transcript-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output-dir", type=click.Path())
@click.option(
    "--absolute-paths",
    default=False,
    help="Whether to return absolute or relative (to the corpus dir) paths for recordings.",
)
def fisher_spanish(
    audio_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Pathlike,
    absolute_paths: bool,
):
    """
    The Fisher Spanish corpus preparation.

    \b
    This is conversational telephone speech collected as 2-channel μ-law, 8kHz-sampled data.
    The catalog number LDC2010S01 for audio corpus and LDC2010T04 for transcripts.

    This data is not available for free - your institution needs to have an LDC subscription.
    """
    prepare_fisher_spanish(
        audio_dir_path=audio_dir,
        transcript_dir_path=transcript_dir,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
    )
