"""
DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech

Abstract: The majority of current TTS datasets, which are collections of individual utterances, contain few conversational aspects in terms of both style and metadata. In this paper, we introduce DailyTalk, a high-quality conversational speech dataset designed for Text-to-Speech. We sampled, modified, and recorded 2,541 dialogues from the open-domain dialogue dataset DailyDialog which are adequately long to represent context of each dialogue. During the data construction step, we maintained attributes distribution originally annotated in DailyDialog to support diverse dialogue in DailyTalk. On top of our dataset, we extend prior work as our baseline, where a non-autoregressive TTS is conditioned on historical information in a dialog. We gather metadata so that a TTS model can learn historical dialog information, the key to generating context-aware speech. From the baseline experiment results, we show that DailyTalk can be used to train neural text-to-speech models, and our baseline can represent contextual information. The DailyTalk dataset and baseline code are freely available for academic use with CC-BY-SA 4.0 license.

Paper: https://arxiv.org/abs/2207.01063
GitHub: https://github.com/keonlee9420/DailyTalk
"""
import logging
import zipfile
from pathlib import Path
from typing import Tuple

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.qa import fix_manifests
from lhotse.serialization import load_json
from lhotse.utils import Pathlike, is_module_available

DAILY_TALK_URL = (
    "https://drive.google.com/file/d/1nPrfJn3TcIVPc0Uf5tiAXUYLJceb_5k-/view?usp=sharing"
)


def download_daily_talk(target_dir: Pathlike, force_download: bool = False) -> Path:
    """
    Downloads the DailyTalk data from the Google Drive and extracts it.
    :param target_dir: the directory where DailyTalk data will be saved.
    :param force_download: if True, it will download the DailyTalk data even if it is already present.
    :return: the path to downloaded and extracted directory with data.
    """
    if not is_module_available("gdown"):
        raise ValueError("Please run 'pip install gdown' to download DailyTalk.")

    import gdown

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "dailytalk"
    corpus_zip = corpus_dir.with_suffix(".zip")

    if not force_download and corpus_zip.exists():
        logging.info(f"{corpus_zip} already exists. Skipping download.")
    else:
        logging.info(f"Running: gdown --fuzzy {DAILY_TALK_URL}")
        gdown.download(DAILY_TALK_URL, str(corpus_zip), fuzzy=True, quiet=False)

    # Extract the zipped file
    if not corpus_dir.exists() or force_download:
        logging.info(f"Extracting {corpus_zip} to {target_dir}")
        with zipfile.ZipFile(corpus_zip) as zf:
            zf.extractall(path=target_dir)

    return corpus_dir


def prepare_daily_talk(
    corpus_dir: Pathlike,
    output_dir: Pathlike = None,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Create RecordingSet and SupervisionSet manifests for DailyTalk from a raw corpus distribution.

    :param corpus_dir: Pathlike, the path to the extracted corpus.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    meta = corpus_dir / "metadata.json"
    data = corpus_dir / "data"

    recordings = RecordingSet.from_dir(
        data, "*.wav", num_jobs=num_jobs, recording_id=make_recording_id
    )
    supervisions = []
    for dialog_idx, dialog in load_json(meta).items():
        for utt_idx, utt in dialog.items():
            recording_id = f"dailytalk_{utt_idx}_{utt['speaker']}_d{dialog_idx}"
            assert utt["speaker"] in [
                0,
                1,
            ], f"Unknown speaker index: {utt['speaker']}"
            supervisions.append(
                SupervisionSegment(
                    id=f"dailytalk_{utt['index']}",
                    recording_id=recording_id,
                    start=0.0,
                    duration=recordings[recording_id].duration,
                    channel=0,
                    text=utt["text"],
                    language="English",
                    speaker=f"dailytalk_spk{utt['speaker']}",
                    gender="F" if utt["speaker"] == 1 else "M",
                    custom={
                        "turn": utt["turn"],
                        "topic": utt["topic"],
                        "emotion": utt["emotion"],
                        "act": utt["act"],
                    },
                )
            )
    supervisions = SupervisionSet.from_segments(supervisions)
    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(
        recordings=recordings, supervisions=supervisions
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        recordings.to_file(output_dir / "dailytalk_recordings_all.jsonl.gz")
        supervisions.to_file(output_dir / "dailytalk_supervisions_all.jsonl.gz")

    return recordings, supervisions


def make_recording_id(path: Path) -> str:
    return f"dailytalk_{path.stem}"
