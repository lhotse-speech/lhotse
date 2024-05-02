import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def prepare_reazonspeech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for part in ["train", "valid", "test"]:
        recordings = []
        supervisions = []
        with open("%s/%s.json" % (corpus_dir, part)) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                recordings.append(
                    Recording.from_file(item["audio_filepath"], recording_id=str(idx))
                )
                supervisions.append(
                    SupervisionSegment(
                        id=str(idx),
                        recording_id=str(idx),
                        start=0.0,
                        duration=item["duration"],
                        channel=0,
                        language="Japanese",
                        speaker=str(idx),
                        text=item["text"],
                    )
                )
                idx += 1

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"reazonspeech_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"reazonspeech_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
