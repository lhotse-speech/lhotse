import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_switchboard
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("audio-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output-dir", type=click.Path())
@click.option("--transcript-dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--sentiment-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Optional path to LDC2020T14 package with sentiment annotations for SWBD.",
)
@click.option(
    "--omit-silence/--retain-silence",
    default=True,
    help="Should the [silence] segments be kept.",
)
@click.option(
    "--absolute-paths",
    default=False,
    help="Whether to return absolute or relative (to the corpus dir) paths for recordings.",
)
def switchboard(
    audio_dir: Pathlike,
    output_dir: Pathlike,
    transcript_dir: Pathlike,
    sentiment_dir: Pathlike,
    omit_silence: bool,
    absolute_paths: bool,
):
    """
    The Switchboard corpus preparation.

    \b
    This is conversational telephone speech collected as 2-channel, 8kHz-sampled
    data.  We are using just the Switchboard-1 Phase 1 training data.
    The catalog number LDC97S62 (Switchboard-1 Release 2) corresponds, we believe,
    to what we have.  We also use the Mississippi State transcriptions, which
    we download separately from
    http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz

    This data is not available for free - your institution needs to have an LDC subscription.
    """
    prepare_switchboard(
        audio_dir=audio_dir,
        transcripts_dir=transcript_dir,
        sentiment_dir=sentiment_dir,
        output_dir=output_dir,
        omit_silence=omit_silence,
        absolute_paths=absolute_paths,
    )
