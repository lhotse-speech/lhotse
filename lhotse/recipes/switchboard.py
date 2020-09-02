"""
About the Switchboard corpus

    This is conversational telephone speech collected as 2-channel, 8kHz-sampled
    data.  We are using just the Switchboard-1 Phase 1 training data.
    The catalog number LDC97S62 (Switchboard-1 Release 2) corresponds, we believe,
    to what we have.  We also use the Mississippi State transcriptions, which
    we download separately from
    http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
"""
import tarfile
import urllib
from pathlib import Path

from lhotse.utils import Pathlike


SWBD_TEXT_URL = 'http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz'


def download_and_untar(
        target_dir: Pathlike = '.',
        force_download: bool = False,
        url: str = SWBD_TEXT_URL
) -> None:
    target_dir = Path(target_dir)
    if (target_dir / 'swb_ms98_transcriptions').is_dir():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_name = 'switchboard_word_alignments.tar.gz'
    tar_path = target_dir / tar_name
    if force_download or not tar_path.is_file():
        urllib.request.urlretrieve(url, filename=tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
