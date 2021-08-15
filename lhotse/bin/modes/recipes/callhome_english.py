import click

from lhotse.bin.modes import prepare
from lhotse.recipes import prepare_callhome_english
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument('audio-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output-dir', type=click.Path())
@click.option('--rttm-dir', type=click.Path(exists=True, file_okay=False))
@click.option('--absolute-paths', default=False,
              help='Whether to return absolute or relative (to the corpus dir) paths for recordings.')
def callhome_english(
        audio_dir: Pathlike,
        output_dir: Pathlike,
        rttm_dir: Pathlike,
        absolute_paths: bool
):
    """
    CallHome English (LDC2001S97) corpus preparation.

    \b
     This script prepares data for speaker diarization on
     a portion of CALLHOME used in the 2000 NIST speaker recognition evaluation.
     The 2000 NIST SRE is required, and has an LDC catalog number LDC2001S97.

    This data is not available for free - your institution needs to have an LDC subscription.

    The data should be located at AUDIO_DIR.
    Optionally, RTTM_DIR can be provided that has the contents of http://www.openslr.org/resources/10/;
    otherwise, we will download it.

    To actually read the audio, you will need the SPH2PIPE binary: you can provide its path,
    so that we will add it in the manifests (otherwise you might need to modify your PATH
    environment variable to find sph2pipe).
    """
    prepare_callhome_english(
        audio_dir=audio_dir,
        rttm_dir=rttm_dir,
        output_dir=output_dir,
        absolute_paths=absolute_paths
    )
