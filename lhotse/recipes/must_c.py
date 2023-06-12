"""
About the MuST-C corpus

MuST-C is a multilingual speech translation corpus whose size and quality will
facilitate the training of end-to-end systems for SLT from English into a set
of languages.

For each target language, MuST-C comprises several hundred hours of audio
recordings from English TED Talks, which are automatically aligned at the
sentence level with their manual transcriptions and translations.

We don't provide download_mustc().

Please refer to
https://ict.fbk.eu/must-c-releases/
for downloading.

If you have downloaded and extracted the dataset to the directory

/ceph-data3/fangjun/data/must-c/v2.0/en-de
/ceph-data3/fangjun/data/must-c/v2.0/en-zh

You can call lhotse with the following commands

(1) When the target language is German:

    lhotse prepare must-c \
      --tgt-lang de \
      -j 10 \
      /ceph-data3/fangjun/data/must-c/v2.0/ \
      ./data/manifests/v2.0

(2) When the target language is Chinese:

    lhotse prepare must-c \
      --tgt-lang zh \
      -j 10 \
      /ceph-data3/fangjun/data/must-c/v2.0/ \
      ./data/manifests/v2.0
"""

import logging
from itertools import groupby, repeat
from pathlib import Path
from typing import Dict, List, Tuple, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.parallel import parallel_map
from lhotse.serialization import load_yaml
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds


def prepare_must_c(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    tgt_lang: str,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """Prepare manifests for the MUST-C corpus.

    :param: corpus_dir: We assume there is a folder {src_lang}-{tgt_lang} inside
        this directory.
    :param: output_dir: Directory where the manifests should be written.
    :param: tgt_lang: Target language, e.g., zh, de, etc.
    :param: src_lang: Source language, e.g., en.
    :param: num_jobs: Number of processes to use for parsing the data
    """
    src_lang = "en"

    in_data_dir = Path(corpus_dir) / f"{src_lang}-{tgt_lang}/data"
    assert in_data_dir.is_dir(), in_data_dir

    datasets = ["dev", "tst-COMMON", "tst-HE", "train"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for d in datasets:
        logging.info(f"Processing {d}")
        dataset_dir = in_data_dir / d
        # In the dataset_dir, we can find two directories: txt and wav
        # In the txt directory, we can find the following files:
        #   {d}.{src_lang}, {d}.{tgt_lang}, {d}.yaml
        assert dataset_dir.is_dir(), dataset_dir / d

        with open(dataset_dir / "txt" / f"{d}.{tgt_lang}") as f:
            transcripts = [line.strip() for line in f]

        segments = load_yaml(dataset_dir / "txt" / f"{d}.yaml")
        assert len(transcripts) == len(segments), (len(transcripts), len(segments))

        # segments[0] contains something like below:
        #  {'duration': 3.5, 'offset': 16.08, 'rW': 9, 'uW': 0,
        #  'speaker_id': 'spk.767', 'wav': 'ted_767.wav'}
        groups = []
        start = 0
        for _, group in groupby(segments, lambda x: x["wav"]):
            this_wave = [list(group)]
            num_sentences = len(this_wave[0])
            end = start + num_sentences

            this_wave.append(transcripts[start:end])
            start = end

            groups.append(this_wave)

        assert start == len(transcripts), (start, len(transcripts))

        recording_list = []
        supervision_list = []
        for recording, sup_segments in tqdm(
            parallel_map(
                parse_utterance,
                repeat(dataset_dir / "wav"),
                groups,
                repeat(tgt_lang),
                num_jobs=num_jobs,
            ),
            desc=f"Processing must-c {d}",
        ):
            recording_list.append(recording)
            supervision_list.extend(sup_segments)

        recordings, supervisions = fix_manifests(
            recordings=RecordingSet.from_recordings(recording_list),
            supervisions=SupervisionSet.from_segments(supervision_list),
        )

        validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )

        with RecordingSet.open_writer(
            output_dir / f"must_c_recordings_{src_lang}-{tgt_lang}_{d}.jsonl.gz"
        ) as rec_writer:
            for r in recordings:
                rec_writer.write(r)

        with SupervisionSet.open_writer(
            output_dir / f"must_c_supervisions_{src_lang}-{tgt_lang}_{d}.jsonl.gz"
        ) as sup_writer:
            for sup in supervisions:
                sup_writer.write(sup)


def parse_utterance(
    wave_dir: Path,
    groups: Tuple[List[dict], List[str]],
    tgt_lang: str,
) -> Tuple[Recording, List[SupervisionSegment]]:
    """
    :param: wave_dir: The wave directory. It contains *.wav files.
    :param: groups: A tuple containing two lists. The first one is a list
        of dict, where each dict contains something like below:

          {duration: 3.500000, offset: 16.080000, rW: 9, uW: 0,
           speaker_id: spk.767, wav: ted_767.wav}

        The second one is a list of transcripts.
    :param: tgt_lang: The language of the transcript, e.g., zh, en, de, etc.
    """
    wave_segments = groups[0]
    transcripts = groups[1]

    assert len(wave_segments) == len(transcripts), (
        len(wave_segments),
        len(transcripts),
    )

    wave_file = wave_dir / wave_segments[0]["wav"]
    recording = Recording.from_file(wave_file)

    segments = []
    for i, (wave_segment, transcript) in enumerate(zip(wave_segments, transcripts)):
        segments.append(
            SupervisionSegment(
                id=f"{recording.id}-seg-{i}",
                recording_id=recording.id,
                start=Seconds(wave_segment["offset"]),
                duration=round(Seconds(wave_segment["duration"]), ndigits=8),
                channel=0,
                language=tgt_lang,
                speaker=wave_segment["speaker_id"],
                text=transcript,
            )
        )
    return recording, segments
