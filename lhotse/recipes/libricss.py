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
import os

from pathlib import Path
from typing import Dict, Union
from lhotse.audio import Recording

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike, check_and_rglob


OVERLAP_RATIOS = ["0L", "0S", "OV10", "OV20", "OV30", "OV40"]


def download_libricss(target_dir: Pathlike):
    """
    Downloads the LibriCSS data from the Google Drive.
    :param target_dir: the directory where the LibriCSS data will be saved.
    """
    # Download command (taken from https://github.com/chenzhuo1011/libri_css/blob/9e3b7b0c9bffd8ef6da19f7056f3a2f2c2484ffa/dataprep/scripts/dataprep.sh#L27)
    command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l" -O for_release.zip && rm -rf /tmp/cookies.txt"""

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, shell=True, cwd=target_dir)


def prepare_libricss(
    corpus_zip: Pathlike,
    output_dir: Pathlike = None,
    type: str = "replay",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    NOTE: To run this function, you need to pass the `for_release.zip` file, which contains
    the LibriCSS data. You can download it from the Google Drive by running ``download_libricss(target_dir)``.

    NOTE: The recordings contain all 7 channels. If you want to use only one channel, you can
    use either ``recording.load_audio(channel=0)`` or ``MonoCut(id=...,recording=recording,channel=0)``
    while creating the CutSet.

    :param corpus_zip: Pathlike, the path to the for_release.zip file.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param type: str, the type of the LibriCSS data. It should be one of "replay" or "mix" for the replayed audio
        and the mixed audio respectively.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.

    """
    corpus_zip = Path(corpus_zip)
    assert (
        corpus_zip.is_file()
    ), f"""No such file: {corpus_zip}. Please run `help(prepare_libricss)` for details"""
    assert type in ["replay", "mix"], f"type should be one of 'replay' or 'mix'"

    manifests = {}

    if not (corpus_zip.parent / "for_release").exists():
        logging.info(f"Extracting {corpus_zip} to {corpus_zip.parent}")
        corpus_zip.unzip(corpus_zip.parent)

    corpus_dir = corpus_zip.parent / "for_release"

    recordings = []
    segments = []

    for ov in OVERLAP_RATIOS:
        for session in (corpus_dir / ov).iterdir():
            _, _, _, _, _, name, actual_ov = session.name.split("_")
            actual_ov = float(actual_ov.split("actual")[1])
            recording_id = f"{ov}_{name}"
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
                    )
                )

            audio_path = (
                session / "clean" / "mix.wav"
                if type == "mix"
                else session / "record" / "raw_recording.wav"
            )
            recordings.append(
                Recording.from_file(audio_path, recording_id=recording_id)
            )

    supervisions = SupervisionSet.from_segments(segments)
    recordings = RecordingSet.from_recordings(recordings)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        recordings.to_jsonl(output_dir / "recordings.jsonl")
        supervisions.to_jsonl(output_dir / "supervisions.jsonl")

    return {'recordings': recordings, 'supervisions': supervisions}


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
