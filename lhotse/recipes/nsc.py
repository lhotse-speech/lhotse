"""
This is a data preparation recipe for the National Corpus of Speech in Singaporean English.

The entire corpus is organised into a few parts.

Part 1 features about 1000 hours of prompted recordings of phonetically-balanced scripts from about 1000 local English speakers.

Part 2 presents about 1000 hours of prompted recordings of sentences randomly generated from words based on people, food, location, brands, etc, from about 1000 local English speakers as well. Transcriptions of the recordings have been done orthographically and are available for download.

Part 3 consists of about 1000 hours of conversational data recorded from about 1000 local English speakers, split into pairs. The data includes conversations covering daily life and of speakers playing games provided.

Parts 1 and 2 were recorded in quiet rooms using 3 microphones: a headset/ standing microphone (channel 0), a boundary microphone (channel 1), and a mobile phone (channel 3). Recordings that are available for download here have been down-sampled to 16kHz. Details of the microphone models used for each speaker as well as some corresponding non-personal and anonymized information can be found in the accompanying spreadsheets.

Part 3's recordings were split into 2 environments. In the Same Room environment where speakers were in same room, the recordings were done using 2 microphones: a close-talk mic and a boundary mic. In the Separate Room environment, speakers were separated into individual rooms. The recordings were done using 2 microphones in each room: a standing mic and a telephone.

We currently only support the part 3 recordings, in "same room close mic" and "separate rooms phone mic" environments.
"""
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike

NSC_PARTS = ["PART3_SameCloseMic", "PART3_SeparateIVR"]


def check_dependencies():
    try:
        import textgrids
    except:
        raise ImportError(
            "NSC data preparation requires the forked 'textgrids' package to be installed. "
            "Please install it with 'pip install git+https://github.com/pzelasko/Praat-textgrids' "
            "and try again."
        )


def prepare_nsc(
    corpus_dir: Pathlike,
    dataset_part: str = "PART3_SameCloseMic",
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path to the raw corpus distribution.
    :param dataset_part: str, name of the dataset part to be prepared.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    check_dependencies()
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if dataset_part == "PART3_SameCloseMic":
        manifests = prepare_same_close_mic(corpus_dir / "PART3")
    elif dataset_part == "PART3_SeparateIVR":
        manifests = prepare_separate_phone_mic(corpus_dir / "PART3")
    else:
        raise ValueError(f"Unknown dataset part: {dataset_part}")

    validate_recordings_and_supervisions(**manifests)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifests["supervisions"].to_json(
            output_dir / f"supervisions_{dataset_part}.json"
        )
        manifests["recordings"].to_json(output_dir / f"recordings_{dataset_part}.json")

    return manifests


def prepare_same_close_mic(part3_path):
    check_dependencies()
    from textgrids import TextGrid

    recordings = []
    supervisions = []
    for audio_path in tqdm(
        (part3_path / "AudioSameCloseMic").glob("*.wav"),
        desc="Creating manifests for SameCloseMic",
    ):
        try:
            recording_id = audio_path.stem
            recording = Recording.from_file(audio_path)

            tg = TextGrid(
                part3_path / f"ScriptsSame/{recording_id}.TextGrid", coding="utf-16"
            )
            segments = [
                s
                for s in (
                    SupervisionSegment(
                        id=f"{recording_id}-{idx}",
                        recording_id=recording_id,
                        start=segment.xmin,
                        # We're trimming the last segment's duration as it exceeds the actual duration of the recording.
                        # This is safe because if we end up with a zero/negative duration, the validation will catch it.
                        duration=min(
                            round(segment.xmax - segment.xmin, ndigits=8),
                            recording.duration - segment.xmin,
                        ),
                        text=segment.text,
                        language="Singaporean English",
                        speaker=recording_id,
                    )
                    for idx, segment in enumerate(tg[recording_id])
                    if segment.text not in ("<S>", "<Z>")  # skip silences
                )
                if s.duration > 0  # NSC has some bad segments
            ]

            recordings.append(recording)
            supervisions.extend(segments)
        except:
            print(f"Error when processing {audio_path} - skipping...")
    return {
        "recordings": RecordingSet.from_recordings(recordings),
        "supervisions": SupervisionSet.from_segments(supervisions),
    }


def prepare_separate_phone_mic(part3_path):
    check_dependencies()
    from textgrids import TextGrid

    recordings = []
    supervisions = []
    for audio_path in tqdm(
        (part3_path / "AudioSeparateIVR").rglob("**/*.wav"),
        desc="Creating manifests for SeparateIVR",
    ):
        try:
            recording_id = f"{audio_path.parent.name}_{audio_path.stem}"
            recording = Recording.from_file(audio_path)

            tg = TextGrid(
                part3_path / f"ScriptsSeparate/{recording_id}.TextGrid", coding="utf-16"
            )
            segments = [
                s
                for s in (
                    SupervisionSegment(
                        id=f"{recording_id}-{idx}",
                        recording_id=recording_id,
                        start=segment.xmin,
                        # We're trimming the last segment's duration as it exceeds the actual duration of the recording.
                        # This is safe because if we end up with a zero/negative duration, the validation will catch it.
                        duration=min(
                            round(segment.xmax - segment.xmin, ndigits=8),
                            recording.duration - segment.xmin,
                        ),
                        text=segment.text,
                        language="Singaporean English",
                        speaker=recording_id,
                    )
                    for idx, segment in enumerate(tg[recording_id])
                    if segment.text not in ("<S>", "<Z>")  # skip silences
                )
                if s.duration > 0  # NSC has some bad segments
            ]

            supervisions.extend(segments)
            recordings.append(recording)
        except:
            print(f"Error when processing {audio_path} - skipping...")
    return {
        "recordings": RecordingSet.from_recordings(recordings),
        "supervisions": SupervisionSet.from_segments(supervisions),
    }
