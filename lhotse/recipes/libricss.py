"""
LibriCSS is a multi-talker meeting corpus formed from mixing together LibriSpeech utterances
and replaying in a real meeting room. It consists of 10 1-hour sessions of audio, each
recorded on a 7-channel microphone. The sessions are recorded at a sampling rate of 16 kHz.
For more information, refer to the paper:
Z. Chen et al., "Continuous speech separation: dataset and analysis," 
ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
Barcelona, Spain, 2020
"""
import logging
import subprocess

from pathlib import Path
from typing import Dict, Union
from lhotse.audio import Recording

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike


# fmt: off
# The following mapping is courtesy Zhuo Chen (Microsoft). It is not available in the original
# LibriCSS dataset. It is useful for preparation of the data in the IHM setting, which consists
# of 8 channels, each belonging to a different speaker. This mapping provides the correspondence
# between the IHM channel and the speaker, which is then used in preparing the supervisions.
SPK_TO_CHANNEL_MAP = {
    "overlap_ratio_0.0_sil0.1_0.5_session0_actual0.0": {"1089": 5, "1320": 3, "1580": 0, "4077": 6, "4992": 1, "6829": 2, "6930": 7, "7176": 4},
    "overlap_ratio_0.0_sil0.1_0.5_session1_actual0.0": {"1089": 4, "121": 3, "2961": 0, "3575": 2, "5105": 6, "6829": 5, "8463": 7, "8555": 1},
    "overlap_ratio_0.0_sil0.1_0.5_session2_actual0.0": {"2961": 3, "4970": 0, "5105": 7, "5639": 5, "61": 2, "7176": 6, "7729": 4, "8224": 1}, 
    "overlap_ratio_0.0_sil0.1_0.5_session3_actual0.0": {"1089": 5, "1320": 4, "260": 3, "5105": 7, "61": 2, "672": 1, "6829": 6, "908": 0}, 
    "overlap_ratio_0.0_sil0.1_0.5_session4_actual0.0": {"1188": 2, "1221": 6, "1995": 1, "2961": 5, "4507": 7, "4970": 0, "5683": 3, "672": 4}, 
    "overlap_ratio_0.0_sil0.1_0.5_session5_actual0.0": {"121": 5, "1221": 4, "2300": 2, "237": 3, "4507": 6, "4970": 0, "7021": 1, "8463": 7}, 
    "overlap_ratio_0.0_sil0.1_0.5_session6_actual0.0": {"260": 5, "3575": 4, "3729": 2, "4507": 6, "4970": 0, "5683": 1, "6829": 7, "7729": 3}, 
    "overlap_ratio_0.0_sil0.1_0.5_session7_actual0.0": {"121": 4, "2300": 1, "260": 3, "3729": 2, "4077": 5, "8224": 0, "8230": 6, "8463": 7}, 
    "overlap_ratio_0.0_sil0.1_0.5_session8_actual0.0": {"1188": 2, "1995": 1, "237": 5, "3570": 0, "5639": 6, "5683": 3, "61": 4, "7127": 7}, 
    "overlap_ratio_0.0_sil0.1_0.5_session9_actual0.0": {"61": 2, "672": 3, "6930": 1, "7021": 0, "7127": 5, "7729": 4, "8230": 6, "8463": 7}, 
    "overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0": {"121": 7, "260": 5, "3575": 0, "5105": 2, "5683": 1, "6930": 6, "8224": 4, "8230": 3}, 
    "overlap_ratio_0.0_sil2.9_3.0_session1_actual0.0": {"1284": 6, "1580": 5, "237": 7, "2961": 3, "3575": 0, "4446": 1, "4507": 4, "7127": 2}, 
    "overlap_ratio_0.0_sil2.9_3.0_session2_actual0.0": {"1188": 4, "121": 6, "1995": 7, "4446": 1, "7021": 5, "7729": 0, "8463": 3, "8555": 2}, 
    "overlap_ratio_0.0_sil2.9_3.0_session3_actual0.0": {"1995": 7, "2094": 3, "2830": 4, "2961": 2, "3729": 6, "4992": 1, "5105": 0, "7021": 5}, 
    "overlap_ratio_0.0_sil2.9_3.0_session4_actual0.0": {"1089": 6, "1188": 5, "2961": 2, "7021": 7, "7729": 0, "8230": 1, "8463": 4, "8555": 3}, 
    "overlap_ratio_0.0_sil2.9_3.0_session5_actual0.0": {"1580": 3, "2094": 2, "260": 6, "3729": 5, "4992": 0, "672": 1, "7021": 4, "8455": 7}, 
    "overlap_ratio_0.0_sil2.9_3.0_session6_actual0.0": {"1188": 4, "1320": 7, "1995": 6, "2300": 0, "3729": 5, "4507": 2, "7127": 1, "8455": 3}, 
    "overlap_ratio_0.0_sil2.9_3.0_session7_actual0.0": {"1089": 5, "1320": 7, "2830": 4, "4077": 3, "4992": 2, "7127": 1, "8230": 0, "908": 6}, 
    "overlap_ratio_0.0_sil2.9_3.0_session8_actual0.0": {"2961": 2, "4992": 1, "5142": 7, "672": 0, "6930": 6, "7176": 5, "8463": 3, "908": 4}, 
    "overlap_ratio_0.0_sil2.9_3.0_session9_actual0.0": {"1089": 5, "1188": 4, "2300": 0, "260": 7, "4077": 3, "672": 1, "8555": 2, "908": 6}, 
    "overlap_ratio_10.0_sil0.1_1.0_session0_actual10.1": {"1320": 6, "1995": 1, "260": 7, "4992": 0, "672": 4, "6930": 5, "8455": 2, "8463": 3}, 
    "overlap_ratio_10.0_sil0.1_1.0_session1_actual10.2": {"1188": 7, "1580": 1, "2094": 5, "3570": 3, "8224": 6, "8463": 4, "8555": 0, "908": 2}, 
    "overlap_ratio_10.0_sil0.1_1.0_session2_actual10.0": {"1188": 7, "3570": 3, "3729": 1, "5683": 4, "61": 6, "7127": 0, "7729": 5, "8463": 2}, 
    "overlap_ratio_10.0_sil0.1_1.0_session3_actual10.1": {"1580": 2, "1995": 3, "2300": 1, "3575": 7, "672": 4, "6829": 0, "7729": 6, "8224": 5}, 
    "overlap_ratio_10.0_sil0.1_1.0_session4_actual10.0": {"1188": 7, "121": 6, "2300": 1, "260": 5, "672": 2, "6829": 0, "7021": 3, "8224": 4},
    "overlap_ratio_10.0_sil0.1_1.0_session5_actual9.9": {"237": 0, "3575": 7, "3729": 3, "4507": 1, "4970": 6, "672": 4, "6930": 5, "8230": 2}, 
    "overlap_ratio_10.0_sil0.1_1.0_session6_actual9.9": {"1089": 5, "121": 3, "1320": 6, "3575": 7, "4446": 4, "4992": 2, "6829": 0, "8555": 1}, 
    "overlap_ratio_10.0_sil0.1_1.0_session7_actual10.1": {"121": 7, "1221": 5, "1995": 3, "4077": 0, "61": 1, "7729": 6, "8463": 4, "908": 2}, 
    "overlap_ratio_10.0_sil0.1_1.0_session8_actual10.0": {"1320": 5, "1580": 1, "2300": 0, "4077": 7, "4446": 3, "672": 4, "7176": 2, "7729": 6},
    "overlap_ratio_10.0_sil0.1_1.0_session9_actual10.0": {"1320": 5, "2830": 3, "3570": 7, "5639": 1, "6930": 4, "8224": 6, "8455": 2, "8555": 0},
    "overlap_ratio_20.0_sil0.1_1.0_session0_actual20.8": {"1089": 2, "121": 1, "1284": 0, "4507": 4, "4970": 7, "6930": 5, "7127": 6, "8555": 3}, 
    "overlap_ratio_20.0_sil0.1_1.0_session1_actual20.5": {"1089": 0, "1320": 6, "1580": 2, "260": 1, "4446": 4, "5105": 7, "5142": 3, "8224": 5}, 
    "overlap_ratio_20.0_sil0.1_1.0_session2_actual21.1": {"1580": 2, "2830": 6, "2961": 5, "3570": 7, "4507": 4, "5639": 1, "6829": 3, "8230": 0},
    "overlap_ratio_20.0_sil0.1_1.0_session3_actual20.0": {"1320": 5, "260": 1, "4992": 7, "5105": 6, "5142": 3, "7729": 2, "8455": 4, "908": 0}, 
    "overlap_ratio_20.0_sil0.1_1.0_session4_actual20.0": {"1089": 1, "1580": 2, "2830": 3, "3570": 7, "3729": 5, "5105": 4, "7127": 6, "8230": 0},
    "overlap_ratio_20.0_sil0.1_1.0_session5_actual19.6": {"1089": 1, "1188": 5, "1284": 0, "2961": 3, "3570": 7, "3575": 6, "61": 2, "8455": 4}, 
    "overlap_ratio_20.0_sil0.1_1.0_session6_actual20.0": {"121": 0, "4446": 5, "4507": 4, "5105": 7, "6829": 2, "7176": 6, "8224": 3, "8463": 1}, 
    "overlap_ratio_20.0_sil0.1_1.0_session7_actual20.1": {"2300": 0, "237": 5, "2830": 2, "2961": 1, "4970": 7, "4992": 6, "672": 4, "6930": 3}, 
    "overlap_ratio_20.0_sil0.1_1.0_session8_actual19.8": {"1221": 2, "1995": 4, "2300": 1, "672": 6, "7127": 7, "8224": 5, "8230": 0, "908": 3}, 
    "overlap_ratio_20.0_sil0.1_1.0_session9_actual20.7": {"1089": 2, "1284": 0, "4077": 3, "4446": 5, "5105": 7, "5639": 1, "7176": 6, "7729": 4},
    "overlap_ratio_30.0_sil0.1_1.0_session0_actual29.7": {"1089": 1, "1995": 6, "237": 7, "2830": 0, "2961": 4, "3575": 2, "672": 3, "7021": 5},
    "overlap_ratio_30.0_sil0.1_1.0_session1_actual30.4": {"1580": 6, "3575": 4, "4970": 2, "4992": 1, "5142": 5, "7729": 0, "8230": 7, "8455": 3},
    "overlap_ratio_30.0_sil0.1_1.0_session2_actual29.6": {"1284": 6, "1995": 7, "3575": 4, "4507": 2, "5639": 3, "61": 5, "8224": 0, "8463": 1}, 
    "overlap_ratio_30.0_sil0.1_1.0_session3_actual30.2": {"1320": 7, "2094": 3, "260": 4, "3575": 5, "4446": 6, "5105": 0, "6930": 1, "7729": 2},
    "overlap_ratio_30.0_sil0.1_1.0_session4_actual29.8": {"121": 1, "1320": 7, "260": 6, "2830": 5, "5683": 4, "6829": 0, "8463": 3, "8555": 2},
    "overlap_ratio_30.0_sil0.1_1.0_session5_actual29.7": {"1089": 5, "260": 4, "2830": 3, "3729": 2, "4077": 1, "4446": 6, "8224": 0, "908": 7},
    "overlap_ratio_30.0_sil0.1_1.0_session6_actual30.1": {"2094": 2, "237": 6, "4992": 4, "5683": 3, "61": 5, "6829": 1, "8555": 0, "908": 7},
    "overlap_ratio_30.0_sil0.1_1.0_session7_actual30.2": {"1089": 5, "1188": 1, "1284": 6, "2300": 4, "2830": 2, "3570": 0, "4446": 7, "4970": 3},
    "overlap_ratio_30.0_sil0.1_1.0_session8_actual29.7": {"1188": 1, "1284": 4, "3570": 0, "3575": 3, "4970": 2, "7021": 5, "8230": 6, "908": 7},
    "overlap_ratio_30.0_sil0.1_1.0_session9_actual29.8": {"1188": 2, "1320": 7, "61": 3, "6930": 1, "7021": 4, "7127": 5, "7176": 6, "7729": 0},
    "overlap_ratio_40.0_sil0.1_1.0_session0_actual39.5": {"121": 0, "1284": 7, "1320": 1, "2830": 5, "3729": 6, "4446": 3, "7127": 2, "7729": 4},
    "overlap_ratio_40.0_sil0.1_1.0_session1_actual39.7": {"121": 0, "1580": 2, "237": 3, "260": 1, "4446": 4, "7021": 7, "7729": 5, "8455": 6},
    "overlap_ratio_40.0_sil0.1_1.0_session2_actual41.2": {"1188": 3, "1284": 7, "1320": 4, "260": 1, "4507": 0, "6930": 5, "8224": 6, "8230": 2},
    "overlap_ratio_40.0_sil0.1_1.0_session3_actual40.2": {"1320": 4, "1580": 1, "3575": 2, "4077": 5, "4970": 0, "5105": 7, "7127": 6, "8463": 3},
    "overlap_ratio_40.0_sil0.1_1.0_session4_actual39.0": {"1188": 1, "121": 0, "1995": 2, "3729": 7, "4077": 3, "7729": 5, "8555": 4, "908": 6},
    "overlap_ratio_40.0_sil0.1_1.0_session5_actual42.0": {"1089": 4, "1284": 7, "237": 2, "2961": 6, "4077": 1, "4446": 3, "4507": 0, "8224": 5},
    "overlap_ratio_40.0_sil0.1_1.0_session6_actual39.9": {"1188": 1, "2094": 4, "3575": 5, "4970": 0, "5105": 7, "672": 2, "7021": 6, "8230": 3},
    "overlap_ratio_40.0_sil0.1_1.0_session7_actual40.5": {"1221": 2, "1580": 1, "2830": 6, "5142": 4, "7021": 7, "8230": 0, "8455": 5, "8463": 3},
    "overlap_ratio_40.0_sil0.1_1.0_session8_actual40.5": {"1580": 4, "260": 3, "3729": 7, "4970": 2, "5639": 6, "61": 0, "6930": 5, "8230": 1},
    "overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9": {"1284": 7, "1995": 0, "2961": 6, "3575": 2, "4077": 4, "7176": 1, "8224": 5, "8463": 3}
}

OVERLAP_RATIOS = ["0L", "0S", "OV10", "OV20", "OV30", "OV40"]
# fmt: on


def download_libricss(target_dir: Pathlike, force_download: bool = False) -> Path:
    """
    Downloads the LibriCSS data from the Google Drive and extracts it.
    :param target_dir: the directory where the LibriCSS data will be saved.
    :param force_download: if True, it will download the LibriCSS data even if it is already present.
    :return: the path to downloaded and extracted directory with data.
    """
    # Download command (taken from https://github.com/chenzhuo1011/libri_css/blob/9e3b7b0c9bffd8ef6da19f7056f3a2f2c2484ffa/dataprep/scripts/dataprep.sh#L27)
    command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l" -O for_release.zip && rm -rf /tmp/cookies.txt"""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_zip = target_dir / "for_release.zip"
    corpus_dir = target_dir / "for_release"

    if not force_download and corpus_zip.exists():
        logging.info(f"{corpus_zip} already exists. Skipping download.")
    else:
        subprocess.run(command, shell=True, cwd=target_dir)

    # Extract the zipped file
    if not corpus_dir.exists() or force_download:
        logging.info(f"Extracting {corpus_zip} to {target_dir}")
        corpus_zip.unzip(target_dir)

    return target_dir


def prepare_libricss(
    corpus_dir: Pathlike,
    output_dir: Pathlike = None,
    type: str = "mdm",
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
    assert type in ["mdm", "ihm-mix", "ihm"]

    manifests = {}

    corpus_dir = Path(corpus_dir)
    corpus_dir = (
        corpus_dir / "for_release" if corpus_dir.stem != "for_release" else corpus_dir
    )

    recordings = []
    segments = []

    for ov in OVERLAP_RATIOS:
        for session in (corpus_dir / ov).iterdir():
            _, _, _, _, _, name, actual_ov = session.name.split("_")
            actual_ov = float(actual_ov.split("actual")[1])
            recording_id = f"{ov}_{name}"
            audio_path = (
                session / "clean" / "mix.wav"
                if type == "ihm-mix"
                else session / "clean" / "each_spk.wav"
                if type == "ihm"
                else session / "record" / "raw_recording.wav"
            )
            recordings.append(
                Recording.from_file(audio_path, recording_id=recording_id)
            )
            for idx, seg in enumerate(
                parse_transcript(session / "transcription" / "meeting_info.txt")
            ):
                segments.append(
                    SupervisionSegment(
                        id=f"{recording_id}-{idx}",
                        recording_id=recording_id,
                        start=seg[0],
                        duration=seg[1] - seg[0],
                        text=seg[4],
                        language="English",
                        speaker=seg[2],
                        channel=SPK_TO_CHANNEL_MAP[session.name][seg[2]]
                        if type == "ihm"
                        else 0,
                    )
                )

    supervisions = SupervisionSet.from_segments(segments)
    recordings = RecordingSet.from_recordings(recordings)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        recordings.to_jsonl(output_dir / "recordings.jsonl")
        supervisions.to_jsonl(output_dir / "supervisions.jsonl")

    return {"recordings": recordings, "supervisions": supervisions}


def parse_transcript(file_name):
    """
    Parses the transcript file and returns a list of SupervisionSegment objects.
    """
    segments = []
    with open(file_name, "r") as f:
        next(f)  # skip the first line
        for line in f:
            start, end, speaker, utt_id, text = line.split("\t")
            segments.append((float(start), float(end), speaker, utt_id, text))
    return segments
