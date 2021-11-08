import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_callhome_english
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("audio-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output-dir", type=click.Path())
@click.option("--rttm-dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--absolute-paths",
    default=False,
    help="Whether to return absolute or relative (to the corpus dir) paths for recordings.",
)
@click.option(
    "--transcript-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the LDC97T14 corpus. Please note that providing this path, "
    "the ASR corpus will be prepared, not the SRE corpus!",
)
def callhome_english(
    audio_dir: Pathlike,
    output_dir: Pathlike,
    rttm_dir: Pathlike,
    absolute_paths: bool,
    transcript_dir: Pathlike,
):
    """
    CallHome American English corpus preparation.

    \b
    Depending on the value of transcript_dir, will prepare either
        * if transcript_dir = None, the SRE task (expected corpus ``LDC2001S97``).
        The setup will reflect speaker diarization on a portion of CALLHOME used in
        the 2000 NIST speaker recognition evaluation. The 2000 NIST SRE is
        required, and has an LDC catalog number LDC2001S97. The data is not
        available for free, but can be licensed from the LDC (Linguistic Data
        Consortium)
        * otherwise data for ASR task (expected LDC corpora ``LDC97S42`` and
        ``LDC97T14``) will be prepared. The data is not available for free, but can
        be licensed from the LDC (Linguistic Data Consortium)


    The data should be located at AUDIO_DIR.
    Optionally, for the SRE task, RTTM_DIR can be provided that has the contents
    of http://www.openslr.org/resources/10/; otherwise, we will download it.

    To actually read the audio, you will need the SPH2PIPE binary: you can provide
    its path, so that we will add it in the manifests (otherwise you might need to
    modify your PATH environment variable to find sph2pipe).

    Example:

        lhotse prepare  callhome-english /export/corpora5/LDC/LDC97S42 --transcript-dir /export/corpora5/LDC/LDC97T14 ./callhome_asr

      or

        lhotse prepare  callhome-english /export/corpora5/LDC/LDC2001S97 ./callhome_sre
    """
    prepare_callhome_english(
        audio_dir=audio_dir,
        rttm_dir=rttm_dir,
        output_dir=output_dir,
        absolute_paths=absolute_paths,
        transcript_dir=transcript_dir,
    )
