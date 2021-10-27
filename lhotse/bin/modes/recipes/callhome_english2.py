import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_callhome_english2
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
def callhome_english2(
    audio_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Pathlike,
    absolute_paths: bool,
):
    """
    About the Callhome American English Corpus

    CALLHOME American English Speech was developed by the Linguistic Data Consortium (LDC)
    and consists of 120 unscripted 30-minute telephone conversations between native
    speakers of English.

    All calls originated in North America; 90 of the 120 calls were placed to various
    locations outisde of North America, while the remaining 30 calls were made within
    North America. Most participants called family members or close friends.

    This recipe uses the speech and transcripts available through LDC. In addition,
    an CALLHOME American English Lexicon (PRONLEX) (available via LDC) was
    provided  to get word to phoneme mappings for the vocabulary.

    The datasets are:

    Speech : LDC97S42
    Transcripts : LDC97T14
    Lexicon : LDC97L20 (unused here)

    To actually read the audio, you will need the SPH2PIPE binary: you can provide its
    path, so that we will add it in the manifests (otherwise you might need to modify
    your PATH environment variable to find sph2pipe).
    """
    prepare_callhome_english2(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
    )
