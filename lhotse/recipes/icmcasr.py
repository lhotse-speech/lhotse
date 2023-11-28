"""
The ICMC-ASR Grand Challenge dataset is collected in a hybrid electric vehicle with speakers sitting in different positions, including the driver seat and passenger seats. The total number of speakers is over 160 and all of them are native Chinese speakers speaking Mandarin without strong accents. To comprehensively capture speech signals of the entire cockpit, two types of recording devices are used: far-field and near-field recording devices. 8 distributed microphones are placed at four seats in the car, which are the driver's seat (DS01C01, DX01C01), the passenger seat (DS02C01, DX02C01), the rear right seat (DS03C01, DX03C01) and the rear left seat (DS04C01, DX04C01). Additionally, 2 linear microphone arrays, each consisting of 2 microphones, are placed on the display screen (DL01C01, DL02C02) and at the center of the inner sunroof (DL02C01, DL02C02), respectively. All 12 channels of far-field data are time-synchronized and included in the released dataset as far-field data. For transcription purposes, each speaker wears a high-fidelity headphone to record near-field audio, denoted by the seat where the speaker is situated. Specifically, DA01, DA02, DA03, and DA04 represent the driver seat, passenger seat, rear right seat and rear left seat, respectively. The near-field data only have single-channel audio recordings. Additionally, a sizable real noise dataset is provided, following the recording setup of the far-filed data but without speaker talking, to facilitate research of in-car scenario data simulation technology.

Participants can obtain the datasets at https://icmcasr.org - please download the datasets manually.
"""

import logging
import os
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.audio.backend import info
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist, normalize_text_alimeeting
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available

ICMCASR = ("train", "dev")  # TODO: Support all subsets when released
POSITION = ("DA01", "DA02", "DA03", "DA04")
# ignore "DX05C01", "DX06C01",
# which are 2-channel reference signals for AEC.
# see https://github.com/MrSupW/ICMC-ASR_Baseline/tree/main
SDM_POSITION = ("DX01C01", "DX02C01", "DX03C01", "DX04C01")


def _parse_utterance(
    corpus_dir: Pathlike,
    section_path: Pathlike,
    mic: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    if not is_module_available("textgrid"):
        raise ValueError(
            "To prepare ICMC ASR data, please 'pip install textgrid' first."
        )
    import textgrid

    recordings = []
    segments = []
    for position in POSITION:
        text_path = (section_path / (position + ".TextGrid")).resolve()
        if not text_path.is_file():
            continue
        if mic == "ihm":
            audio_paths = [(section_path / (position + ".wav")).resolve()]
            recording_ids = [
                str(section_path / position)
                .replace(str(corpus_dir) + "/", "")
                .replace("/", "-")
            ]
        elif mic == "sdm":
            audio_paths = [
                (section_path / (sdm_position + ".wav")).resolve()
                for sdm_position in SDM_POSITION
            ]
            recording_ids = [
                str(section_path / sdm_position)
                .replace(str(corpus_dir) + "/", "")
                .replace("/", "-")
                + f"-{position}"
                for sdm_position in SDM_POSITION
            ]
        elif mic == "mdm":
            audio_paths = ["fake_audio_path_for_mdm"]
            recording_ids = [
                str(section_path / "DXmixC01")
                .replace(str(corpus_dir) + "/", "")
                .replace("/", "-")
                + f"-{position}"
            ]
        else:
            raise ValueError(f"Unsupported mic type: {mic}")

        for audio_path, recording_id in zip(audio_paths, recording_ids):
            if mic == "mdm":
                channel_paths = [
                    (section_path / (position + ".wav")).resolve()
                    for position in SDM_POSITION
                ]
                audio_info = info(
                    channel_paths[0],
                    force_opus_sampling_rate=None,
                    force_read_audio=False,
                )
                recordings.append(
                    Recording(
                        id=recording_id,
                        sources=[
                            AudioSource(
                                type="file",
                                channels=[idx],
                                source=str(audio_path),
                            )
                            for idx, audio_path in enumerate(channel_paths)
                        ],
                        sampling_rate=16000,
                        num_samples=audio_info.frames,
                        duration=audio_info.duration,
                    )
                )
            # check if audio_path exists, if not, then skip
            else:
                if not audio_path.is_file():
                    # give some warning
                    logging.warning(
                        f"Audio file {audio_path} does not exist - skipping."
                    )
                    continue
                recordings.append(
                    Recording.from_file(path=audio_path, recording_id=recording_id)
                )

            tg = textgrid.TextGrid.fromFile(str(text_path))
            assert len(tg.tiers) == 1, f"Expected 1 tier, found {len(tg.tiers)} tiers."
            tier = tg.tiers[0]
            speaker = tier.name
            for i, interval in enumerate(tier.intervals):
                if interval.mark != "":
                    start = interval.minTime
                    end = interval.maxTime
                    text = interval.mark
                    segment = SupervisionSegment(
                        id=f"{recording_id}-{i}",
                        recording_id=recording_id,
                        start=start,
                        duration=round(end - start, 4),
                        channel=0 if mic in ["sdm", "ihm"] else list(range(4)),
                        language="Chinese",
                        speaker=speaker,
                        text=normalize_text_alimeeting(text),
                    )
                    segments.append(segment)

    return recordings, segments


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    mic: str,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    part_path = corpus_dir / subset
    sections = os.listdir(part_path)

    with ThreadPoolExecutor(num_jobs) as ex:
        futures = []
        recording_set = []
        supervision_set = []
        for section in tqdm(sections, desc="Distributing tasks"):
            section_path = part_path / section
            futures.append(ex.submit(_parse_utterance, corpus_dir, section_path, mic))

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            recordings, segments = result
            recording_set.extend(recordings)
            supervision_set.extend(segments)

        recording_set = RecordingSet.from_recordings(recording_set)
        supervision_set = SupervisionSet.from_segments(supervision_set)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_icmcasr(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    mic: str = "ihm",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the ICMC-ASR dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing ICMC-ASR...")

    subsets = ICMCASR

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing ICMC-ASR subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix=f"icmcasr-{mic}",
            suffix="jsonl.gz",
        ):
            logging.info(f"ICMC-ASR subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(
            part, corpus_dir, mic, num_jobs
        )

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"icmcasr-{mic}_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"icmcasr-{mic}_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
