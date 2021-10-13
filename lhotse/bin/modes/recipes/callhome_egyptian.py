import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_callhome_egyptian
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
def callhome_egyptian(
    audio_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Pathlike,
    absolute_paths: bool,
):
    """
    About the Callhome Egyptian Arabic Corpus

    The CALLHOME Egyptian Arabic corpus of telephone speech consists of 120 unscripted
    telephone conversations between native speakers of Egyptian Colloquial Arabic (ECA),
    the spoken variety of Arabic found in Egypt. The dialect of ECA that this
    dictionary represents is Cairene Arabic.

    This recipe uses the speech and transcripts available through LDC. In addition,
    an Egyptian arabic phonetic lexicon (available via LDC) is used to get word to
    phoneme mappings for the vocabulary. This datasets are:

    Speech : LDC97S45
    Transcripts : LDC97T19
    Lexicon : LDC99L22 (unused here)

    To actually read the audio, you will need the SPH2PIPE binary: you can provide its path,
    so that we will add it in the manifests (otherwise you might need to modify your PATH
    environment variable to find sph2pipe).
    """
    prepare_callhome_egyptian(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
    )
