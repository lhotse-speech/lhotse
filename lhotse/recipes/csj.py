"""
Corpus owner: https://clrd.ninjal.ac.jp/csj/en/index.html
Corpus description:
- http://www.lrec-conf.org/proceedings/lrec2000/pdf/262.pdf
- https://isca-speech.org/archive_open/archive_papers/sspr2003/sspr_mmo2.pdf

This script assumes that the transcript directory that was passed in has been
parsed by csj_make_transcript.py. Individual speaker IDs - or more precisely,
session IDs, to cater for the 'D' dialogue cases - each have their own
folder. These are omitted as '...' in the directory tree below.
Notice that the 'D' transcripts are split into respective (L)eft and (R)ight
channels.

{transcript_dir}
 - excluded
   - ...
 - core
   - ...
 - eval1
   - ...
 - eval2
   - ...
 - eval3
   - ...
 - noncore
   - ...
   - A01F0576
     - A01F0576.sdb (not used in this script)
     - A01F0576-{transcript_mode}.txt
     - A01F0576-segments (not used in this script)
     - A01F0576-wav.list
   - ...
   - D03M0038
     - D03M0038.sdb (not used in this script)
     - D03M0038-L-{transcript_mode}.txt
     - D03M0038-L-segments (not used in this script)
     - D03M0038-L-wav.list
     - D03M0038-R-{transcript_mode}.txt
     - D03M0038-R-segments (not used in this script)
     - D03M0038-R-wav.list

"""

import logging
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

ORI_DATA_PARTS = (
    "eval1",
    "eval2",
    "eval3",
    "core",
    "noncore",
)


def parse_transcript_header(line: str):
    sgid, start, end, line = line.split(" ", maxsplit=3)
    return (sgid, float(start), float(end), line)


def parse_one_recording(
    template: Path, wavlist_path: Path, recording_id: str
) -> Tuple[Recording, List[SupervisionSegment]]:
    transcripts = []

    for trans in template.glob(f"{recording_id}*.txt"):
        trans_type = trans.stem.replace(recording_id + "-", "")
        transcripts.append(
            [(trans_type, t) for t in Path(trans).read_text().split("\n")]
        )

    assert all(len(c) == len(transcripts[0]) for c in transcripts), transcripts
    wav = wavlist_path.read_text()

    recording = Recording.from_file(wav, recording_id=recording_id)

    supervision_segments = []

    for texts in zip(*transcripts):
        customs = {}
        for trans_type, line in texts:
            sgid, start, end, customs[trans_type] = parse_transcript_header(line)

        text = texts[0][1] if len(customs) == 1 else ""

        supervision_segments.append(
            SupervisionSegment(
                id=sgid,
                recording_id=recording_id,
                start=start,
                duration=(end - start),
                channel=0,
                language="Japanese",
                speaker=recording_id,
                gender=("Male" if recording_id[3] == "M" else "Female"),
                text=text,
                custom=customs,
            )
        )

    return recording, supervision_segments


def prepare_csj(
    transcript_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = None,
    output_dir: Pathlike = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will
    simply read and return them.

    :param transcript_dir: Pathlike, the path to the transcripts.
        Assumes that that the transcripts were processed by
        csj_make_transcript.py.
    :param dataset_parts: string or sequence of strings representing
        dataset part names, e.g. 'eval1', 'core', 'eval2'. This defaults to the
        full dataset - core, noncore, eval1, eval2, and eval3.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts
        with the keys 'recordings' and 'supervisions'.
    """

    transcript_dir = Path(transcript_dir)
    assert (
        transcript_dir.is_dir()
    ), f"No such directory for transcript_dir: {transcript_dir}"

    if not dataset_parts:
        dataset_parts = ORI_DATA_PARTS

    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    manifests = {}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exit: we can read them and
        # save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts,
            output_dir=output_dir,
            prefix="csj",
        )

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            logging.info(f"Processing CSJ subset: {part}")
            if manifests_exist(part=part, output_dir=output_dir, prefix="csj"):
                logging.info(f"CSJ subset: {part} already prepared - skipping.")
                continue

            recordings = []
            supervisions = []
            part_path = transcript_dir / part
            futures = []

            for wavlist in part_path.glob("*/*-wav.list"):
                spk = wavlist.name.rstrip("-wav.list")
                template = wavlist.parent

                futures.append(ex.submit(parse_one_recording, template, wavlist, spk))

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                assert result
                recording, segments = result
                recordings.append(recording)
                supervisions.extend(segments)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)
            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir:
                supervision_set.to_file(
                    output_dir / f"csj_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(output_dir / f"csj_recordings_{part}.jsonl.gz")

            manifests[part] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }

    return manifests
