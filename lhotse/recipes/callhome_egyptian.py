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
  Lexicon : LDC99L22
"""

from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import (
    CutSet,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import create_cuts_per_recording
from lhotse.utils import Pathlike, check_and_rglob

# The audio has two channels: speaker A is on channel 0 and speaker B on channel 1.
CHANNEL_OF_SPEAKER = {"A": 0, "B": 1}


def prepare_callhome_egyptian(
    audio_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: bool = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]:
    """
    Prepare manifests for the Callhome Egyptian Arabic Corpus
    We create two manifests: one with recordings, and the other one with text
    supervisions.

    :param audio_dir: Path to ``LDC97S45`` package.
    :param transcript_dir: Path to the ``LDC97T19`` content
    :param output_dir: Directory where the manifests should be written. Can be omitted
        to avoid writing.
    :param absolute_paths: Whether to return absolute or relative (to the corpus dir)
        paths for recordings.
    :return: A dict whose key is the dataset split (``train``, ``devtest``, ``evaltest``),
        and the value is a dict with the keys: ``{'recordings', 'supervisions', 'cuts'}``.
        Supervisions of speaker A are assigned to channel 0 and of speaker B to channel 1.
    """
    audio_dir = Path(audio_dir)
    transcript_dir = Path(transcript_dir)

    manifests = {}

    for split in ["train", "devtest", "evaltest"]:
        audio_paths = check_and_rglob(
            # The LDC distribution has a typo.
            audio_dir / "callhome/arabic" / split.replace("evaltest", "evltest"),
            "*.sph",
        )
        recordings = RecordingSet.from_recordings(
            Recording.from_file(p, relative_path_depth=None if absolute_paths else 4)
            for p in tqdm(audio_paths)
        )

        transcript_paths = check_and_rglob(
            transcript_dir / f"callhome_arabic_trans_970711/transcrp/{split}/roman",
            "*.txt",
        )

        # TODO: Add text normalization like in Kaldi recipe.
        #       Not doing this right now as it's not needed for VAD/diarization...
        supervisions = []
        for p in transcript_paths:
            idx = 0
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                recording_id = p.stem
                # example line:
                # 19.33 21.18 B: %ah Tayyib
                start, end, spk, text = line.split(maxsplit=3)
                spk = spk.replace(":", "")
                duration = float(Decimal(end) - Decimal(start))
                if duration <= 0:
                    continue
                start = float(start)
                supervisions.append(
                    SupervisionSegment(
                        id=f"{recording_id}_{idx}",
                        recording_id=recording_id,
                        start=start,
                        duration=duration,
                        channel=CHANNEL_OF_SPEAKER[spk],
                        speaker=f"{recording_id}_{spk}",
                        text=text,
                    )
                )
                idx += 1
        supervisions = SupervisionSet.from_segments(supervisions)

        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        cuts_path = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recordings.to_file(
                output_dir / f"callhome-egyptian_recordings_{split}.jsonl.gz"
            )
            supervisions.to_file(
                output_dir / f"callhome-egyptian_supervisions_{split}.jsonl.gz"
            )
            cuts_path = output_dir / f"callhome-egyptian_cuts_{split}.jsonl.gz"

        manifests[split] = {
            "recordings": recordings,
            "supervisions": supervisions,
            "cuts": create_cuts_per_recording(
                recordings, supervisions, output_path=cuts_path
            ),
        }

    return manifests
