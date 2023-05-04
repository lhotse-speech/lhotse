"""
DiPCo is a speech data corpus that simulates a “dinner party” scenario taking place in
an everyday home environment. The corpus was created by recording multiple groups of
four Amazon employee volunteers having a natural conversation in English around a dining
table. The participants were recorded by a single-channel close-talk microphone and
by five far-field 7-microphone array devices positioned at different locations in the
recording room. The dataset contains the audio recordings and human labeled transcripts
of a total of 10 sessions with a duration between 15 and 45 minutes. The corpus was
created to advance in the field of noise robust and distant speech processing and is
intended to serve as a public research and benchmarking data set.

The corpus is made availabe under the CDLA-Permissive license.

More details and download link: https://www.amazon.science/publications/dipco-dinner-party-corpus
"""

import tarfile
from collections import defaultdict
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.recipes.utils import normalize_text_chime6
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations, resumable_download, safe_extract

CORPUS_URL = "https://s3.amazonaws.com/dipco/DiPCo.tgz"

DATASET_PARTS = {
    "dev": ["S02", "S04", "S05", "S09", "S10"],
    "eval": ["S01", "S03", "S06", "S07", "S08"],
}


def download_dipco(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and untar the dataset
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_path = target_dir / "DiPCo.tgz"
    resumable_download(CORPUS_URL, filename=tar_path, force_download=force_download)
    with tarfile.open(tar_path) as tar:
        safe_extract(tar, path=target_dir)

    return target_dir


def get_session_id(session: str, use_chime7_offset: bool = False) -> str:
    """
    Returns the session ID
    :param session: str, the session ID.
    :param use_chime7_offset: bool, if True, offset session IDs (from CHiME-7 challenge).
    :return: str, the session ID.
    """
    # CHiME-7 challenge offset DiPCo sessions by 24 since the first 24 sessions are
    # used for CHiME-6 sessions.
    if use_chime7_offset:
        session_number = int(session[1:])
        return f"S{24+session_number:02d}"
    else:
        return session


def prepare_dipco(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    mic: Optional[str] = "mdm",
    normalize_text: Optional[str] = "kaldi",
    use_chime7_offset: Optional[bool] = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use, choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings. For MDM, there are 5 array devices with 7
        channels each, so the resulting recordings will have 35 channels.
    :param normalize_text: str, the text normalization to apply. Choose from "none",
        "upper", or "kaldi". "kaldi" is the default and is the same normalization
        used in Kaldi's CHiME-6 recipe.
    :param use_chime7_offset: bool, if True, offset session IDs (from CHiME-7 challenge).
    :return: a Dict whose key is the dataset part ("dev" and "eval"), and the value is
        Dicts with the keys 'recordings' and 'supervisions'.
    """
    import json

    import soundfile as sf

    assert mic in ["ihm", "mdm"], "mic must be one of 'ihm' or 'mdm'"

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for part in ["dev", "eval"]:
        recordings = []
        supervisions = []

        # First we create the recordings
        if mic == "ihm":
            global_spk_channel_map = {}
            for session in DATASET_PARTS[part]:
                audio_paths = [
                    p for p in (corpus_dir / "audio" / part).rglob(f"{session}_P*.wav")
                ]

                sources = []
                for idx, audio_path in enumerate(audio_paths):
                    sources.append(
                        AudioSource(type="file", channels=[idx], source=str(audio_path))
                    )
                    spk_id = audio_path.stem.split("_")[1]
                    global_spk_channel_map[(session, spk_id)] = idx

                audio_sf = sf.SoundFile(str(audio_paths[0]))

                recordings.append(
                    Recording(
                        id=get_session_id(session, use_chime7_offset),
                        sources=sources,
                        sampling_rate=int(audio_sf.samplerate),
                        num_samples=audio_sf.frames,
                        duration=audio_sf.frames / audio_sf.samplerate,
                    )
                )

        else:
            for session in DATASET_PARTS[part]:
                audio_paths = [
                    p for p in (corpus_dir / "audio" / part).rglob(f"{session}_U*.wav")
                ]

                sources = []
                for idx, audio_path in enumerate(sorted(audio_paths)):
                    sources.append(
                        AudioSource(type="file", channels=[idx], source=str(audio_path))
                    )

                audio_sf = sf.SoundFile(str(audio_paths[0]))

                recordings.append(
                    Recording(
                        id=get_session_id(session, use_chime7_offset),
                        sources=sources,
                        sampling_rate=int(audio_sf.samplerate),
                        num_samples=audio_sf.frames,
                        duration=audio_sf.frames / audio_sf.samplerate,
                    )
                )

        _get_time = lambda x: (
            dt.strptime(x, "%H:%M:%S.%f") - dt(1900, 1, 1)
        ).total_seconds()

        # Then we create the supervisions
        for session in DATASET_PARTS[part]:
            with open(corpus_dir / "transcriptions" / part / f"{session}.json") as f:
                transcript = json.load(f)
                for idx, segment in enumerate(transcript):
                    spk_id = segment["speaker_id"]
                    channel = (
                        global_spk_channel_map[(session, spk_id)]
                        if mic == "ihm"
                        else list(range(35))
                    )
                    start = _get_time(segment["start_time"]["close-talk"])
                    end = _get_time(segment["end_time"]["close-talk"])
                    session_id = get_session_id(session, use_chime7_offset)
                    supervisions.append(
                        SupervisionSegment(
                            id=f"{session_id}-{idx}",
                            recording_id=session_id,
                            start=start,
                            duration=add_durations(end, -start, sampling_rate=16000),
                            channel=channel,
                            text=normalize_text_chime6(
                                segment["words"], normalize=normalize_text
                            ),
                            language="English",
                            speaker=spk_id,
                            gender=segment["gender"],
                            custom={
                                "nativeness": segment["nativeness"],
                                "mother_tongue": segment["mother_tongue"],
                            },
                        )
                    )

        recording_set, supervision_set = fix_manifests(
            RecordingSet.from_recordings(recordings),
            SupervisionSet.from_segments(supervisions),
        )
        # Fix manifests
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"dipco-{mic}_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"dipco-{mic}_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests
