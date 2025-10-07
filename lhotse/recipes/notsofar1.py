import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from lhotse import CutSet, MonoCut, fix_manifests
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def download_notsofar1(
    target_dir: Pathlike = ".",
    parts: Tuple[str] = ("train", "dev", "test"),
    mic: str = "sdm",
    train_version: str = "240825.1_train",
    dev_version: str = "240825.1_dev1",
    test_version: str = "240629.1_eval_small_with_GT",
    force_download: Optional[bool] = False,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as import_error:
        raise RuntimeError(
            "huggingface_hub is required for NOTSOFAR downloads. Install it via:\n"
            "  pip install huggingface_hub\n"
        ) from import_error

    hugging_face_token = os.getenv("HF_TOKEN")
    if not hugging_face_token:
        raise RuntimeError(
            "HuggingFace token not found. Please set the HF_TOKEN environment variable. "
            "If you have set it, please restart the session. "
        )

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for part in parts:
        if part == "train":
            subset_name = "train_set"
            version = train_version
        elif part == "dev":
            subset_name = "dev_set"
            version = dev_version
        elif part == "test":
            subset_name = "eval_set"
            version = test_version
        else:
            raise ValueError(
                f"Unknown part: {part}. Expected one of: 'train', 'dev', 'test'."
            )

        download_patterns = [
            f"benchmark-datasets/{subset_name}/{version}/MTG/*/*.json",
        ]
        if mic == "sdm":
            download_patterns.append(
                f"benchmark-datasets/{subset_name}/{version}/MTG/*/sc_*"
            )
        elif mic == "mdm":
            download_patterns.append(
                f"benchmark-datasets/{subset_name}/{version}/MTG/*/mc_*"
            )

        snapshot_download(
            repo_id="microsoft/NOTSOFAR",
            repo_type="dataset",
            local_dir=target_dir,
            force_download=bool(force_download),
            allow_patterns=download_patterns,
        )

    return target_dir


def prepare_notsofar1(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir) / "benchmark-datasets"
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    output_dir = Path(output_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    os.makedirs(output_dir, exist_ok=True)

    manifests = defaultdict(dict)

    for part in _listdir_safe(corpus_dir):
        part_dir = corpus_dir / part
        manifests[part] = defaultdict(dict)

        for version in _listdir_safe(part_dir):
            version_dir = part_dir / version / "MTG"
            sc_cuts, mc_cuts = process_data(
                version_dir, word_level=False, create_word_alignment=True
            )
            manifests[part][version] = defaultdict(dict)

            if sc_cuts:
                sc_recs, sc_sups = fix_manifests(
                    *CutSet.from_cuts(sc_cuts).decompose()[:2]
                )
                tag = f"notsofar1_sdm_{part}_{version}"
                sc_recs.to_file(output_dir / f"{tag}_recordings.jsonl.gz")
                sc_sups.to_file(output_dir / f"{tag}_supervisions.jsonl.gz")
                manifests[part][version]["single_channel"] = {
                    "recordings": sc_recs,
                    "supervisions": sc_sups,
                }

            if mc_cuts:
                mc_recs, mc_sups = fix_manifests(
                    *CutSet.from_cuts(mc_cuts).decompose()[:2]
                )
                tag = f"notsofar1_mdm_{part}_{version}"
                mc_recs.to_file(output_dir / f"{tag}_recordings.jsonl.gz")
                mc_sups.to_file(output_dir / f"{tag}_supervisions.jsonl.gz")
                manifests[part][version]["multi_channel"] = {
                    "recordings": mc_recs,
                    "supervisions": mc_sups,
                }

    return manifests


def _listdir_safe(path: Pathlike) -> List[str]:
    return list(filter(lambda name: ".DS_Store" not in name, os.listdir(path)))


def process_data(
    dataset_path, word_level=False, create_word_alignment=True
) -> Tuple[List[MonoCut], List[MonoCut]]:
    meetings = sorted(_listdir_safe(dataset_path))
    sc_cuts = []
    mc_cuts = []

    for meeting in tqdm(meetings):
        meeting_root = dataset_path / meeting
        transcription_path = meeting_root / "gt_transcription.json"
        devices = sorted(
            list(
                filter(
                    lambda x: x != "close_talk" and os.path.isdir(meeting_root / x),
                    _listdir_safe(meeting_root),
                )
            )
        )

        with open(transcription_path, "r") as f:
            transcription_json = json.load(f)

        for device in devices:
            device_path = meeting_root / device
            device_id = f"{meeting}_{device}"
            is_multi_channel = "mc" in device
            if is_multi_channel:
                # We assume the channel numbers range from 0 to num_channels - 1.
                num_channels = len(_listdir_safe(device_path))
                recording = Recording.from_file(device_path / f"ch0.wav")
                recording.id = device_id
                recording.channel_ids = list(range(num_channels))
                recording.sources = [
                    AudioSource(
                        type="file",
                        channels=[i],
                        source=str(device_path / f"ch{i}.wav"),
                    )
                    for i in range(num_channels)
                ]
            else:
                recording_path = device_path / "ch0.wav"
                recording = Recording.from_file(recording_path)
                recording.id = device_id

            supervisions = []
            for segment in transcription_json:
                speaker_id = segment["speaker_id"]
                channel = recording.channel_ids
                start_time = float(segment["start_time"])
                end_time = float(segment["end_time"])
                text = segment["text"]
                alignment = None

                if create_word_alignment:
                    alignment = {"word": []}

                    for alig_text, alig_start_time, alig_end_time in segment[
                        "word_timing"
                    ]:
                        # Skip all the fillings.
                        if "<" in alig_text or ">" in alig_text:
                            continue
                        alig_start_time = float(alig_start_time)
                        alig_end_time = float(alig_end_time)
                        alignment["word"].append(
                            AlignmentItem(
                                symbol=alig_text,
                                start=alig_start_time,
                                duration=alig_end_time - alig_start_time,
                            )
                        )

                supervisions.append(
                    SupervisionSegment(
                        id=f"{device_id}_{str(int(start_time*100)).zfill(6)}_{str(int(end_time*100)).zfill(6)}",
                        recording_id=recording.id,
                        start=start_time,
                        duration=end_time - start_time,
                        channel=channel,
                        text=text,
                        speaker=speaker_id,
                        alignment=alignment,
                    )
                )

            if is_multi_channel:
                mc_cuts.append(
                    MonoCut(
                        id=recording.id,
                        start=0,
                        duration=recording.duration,
                        channel=recording.channel_ids,
                        supervisions=supervisions,
                        recording=recording,
                    )
                )
            else:
                sc_cuts.append(
                    MonoCut(
                        id=recording.id,
                        start=0,
                        duration=recording.duration,
                        channel=0,
                        supervisions=supervisions,
                        recording=recording,
                    )
                )

    return sc_cuts, mc_cuts
