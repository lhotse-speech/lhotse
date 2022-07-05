"""
DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech

Abstract: The majority of current TTS datasets, which are collections of individual utterances, contain few conversational aspects in terms of both style and metadata. In this paper, we introduce DailyTalk, a high-quality conversational speech dataset designed for Text-to-Speech. We sampled, modified, and recorded 2,541 dialogues from the open-domain dialogue dataset DailyDialog which are adequately long to represent context of each dialogue. During the data construction step, we maintained attributes distribution originally annotated in DailyDialog to support diverse dialogue in DailyTalk. On top of our dataset, we extend prior work as our baseline, where a non-autoregressive TTS is conditioned on historical information in a dialog. We gather metadata so that a TTS model can learn historical dialog information, the key to generating context-aware speech. From the baseline experiment results, we show that DailyTalk can be used to train neural text-to-speech models, and our baseline can represent contextual information. The DailyTalk dataset and baseline code are freely available for academic use with CC-BY-SA 4.0 license.

Paper: https://arxiv.org/abs/2207.01063
GitHub: https://github.com/keonlee9420/DailyTalk
"""
import logging
import subprocess
from pathlib import Path
from typing import Dict, Union

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.audio import Recording
from lhotse.utils import Pathlike

DAILY_TALK_URL = "https://drive.google.com/file/d/1nPrfJn3TcIVPc0Uf5tiAXUYLJceb_5k-"


def download_daily_talk(target_dir: Pathlike, force_download: bool = False) -> Path:
    """
    Downloads the DailyTalk data from the Google Drive and extracts it.
    :param target_dir: the directory where DailyTalk data will be saved.
    :param force_download: if True, it will download the DailyTalk data even if it is already present.
    :return: the path to downloaded and extracted directory with data.
    """
    command = f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate '{DAILY_TALK_URL}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nPrfJn3TcIVPc0Uf5tiAXUYLJceb_5k-" -O daily_talk.zip && rm -rf /tmp/cookies.txt"""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "daily_talk"
    corpus_zip = corpus_dir.with_suffix(".zip")

    if not force_download and corpus_zip.exists():
        logging.info(f"{corpus_zip} already exists. Skipping download.")
    else:
        subprocess.run(command, shell=True, cwd=target_dir)

    # Extract the zipped file
    if not corpus_dir.exists() or force_download:
        logging.info(f"Extracting {corpus_zip} to {target_dir}")
        corpus_zip.unzip(target_dir)

    return target_dir


def prepare_daily_talk(
    corpus_dir: Pathlike,
    output_dir: Pathlike = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    NOTE: The recordings contain all 7 channels. If you want to use only one channel, you can
    use either ``recording.load_audio(channel=0)`` or ``MonoCut(id=...,recording=recording,channel=0)``
    while creating the CutSet.

    :param corpus_dir: Pathlike, the path to the extracted corpus.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param type: str, the type of data to prepare ('mdm', 'sdm', 'ihm-mix', or 'ihm'). These settings
        are similar to the ones in AMI and ICSI recipes.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.

    """
    raise NotImplementedError()
