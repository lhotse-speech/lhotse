import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from lhotse import CutSet, MonoCut, fix_manifests
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.recipes.utils import create_cuts_per_recording
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

MICS = ("sdm", "mdm", "ihm")


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
        elif mic == "ihm":
            download_patterns.append(
                f"benchmark-datasets/{subset_name}/{version}/MTG/*/close_talk*"
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
    mic: str = "sdm",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]:
    """
    Prepare one microphone setting of the NOTSOFAR-1 corpus (the files are named
    ``notsofar1-<mic>_<manifest>_<part>.jsonl.gz``).

    :param corpus_dir: the directory that contains ``benchmark-datasets``.
    :param output_dir: optional directory where the manifests are written.
    :param mic: ``"sdm"`` (single-channel recordings, ``sc_*`` devices),
        ``"mdm"`` (multi-channel recordings, ``mc_*`` devices) or ``"ihm"``
        (close-talk recordings, one per participant).
    :return: a Dict whose keys are the dataset parts, named ``<set>-<version>``
        (e.g. ``train_set-240825.1_train``, ``dev_set-240825.1_dev1``,
        ``eval_set-240629.1_eval_small_with_GT``), and the values are Dicts
        with the keys "recordings", "supervisions", and "cuts".
    """
    if mic not in MICS:
        raise ValueError(f"mic must be one of {MICS}, got: {mic!r}")
    corpus_dir = Path(corpus_dir) / "benchmark-datasets"
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    corpus_name = f"notsofar1-{mic}"

    manifests = {}
    for subset in sorted(_listdir_safe(corpus_dir)):
        for version in sorted(_listdir_safe(corpus_dir / subset)):
            part = f"{subset}-{version}"
            cuts = process_data(
                corpus_dir / subset / version / "MTG",
                mic=mic,
                word_level=False,
                create_word_alignment=True,
            )
            if not cuts:
                continue
            recs, sups = fix_manifests(*CutSet.from_cuts(cuts).decompose()[:2])
            cuts_path = None
            if output_dir is not None:
                recs.to_file(output_dir / f"{corpus_name}_recordings_{part}.jsonl.gz")
                sups.to_file(output_dir / f"{corpus_name}_supervisions_{part}.jsonl.gz")
                cuts_path = output_dir / f"{corpus_name}_cuts_{part}.jsonl.gz"
            manifests[part] = {
                "recordings": recs,
                "supervisions": sups,
                "cuts": create_cuts_per_recording(recs, sups, output_path=cuts_path),
            }
    if not manifests:
        raise ValueError(f"No NOTSOFAR-1 data for mic={mic!r} found in {corpus_dir}")
    return manifests


def _listdir_safe(path: Pathlike) -> List[str]:
    return list(filter(lambda name: ".DS_Store" not in name, os.listdir(path)))


def process_data(
    dataset_path, mic="sdm", word_level=False, create_word_alignment=True
) -> List[MonoCut]:
    meetings = sorted(_listdir_safe(dataset_path))
    cuts = []

    for meeting in tqdm(meetings):
        meeting_root = dataset_path / meeting
        transcription_path = meeting_root / "gt_transcription.json"
        devices = sorted(
            list(
                filter(
                    lambda x: os.path.isdir(meeting_root / x),
                    _listdir_safe(meeting_root),
                )
            )
        )

        metadata_path = meeting_root / "gt_meeting_metadata.json"
        ct_device_to_speaker = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            alias_to_ct = metadata.get("ParticipantAliasToCtDevice", {})
            ct_device_to_speaker = {v: k for k, v in alias_to_ct.items()}

        with open(transcription_path, "r") as f:
            transcription_json = json.load(f)

        for device in devices:
            device_path = meeting_root / device
            device_id = f"{meeting}_{device}"
            is_multi_channel = "mc" in device
            is_close_talk = "close_talk" in device
            device_mic = (
                "ihm" if is_close_talk else "mdm" if is_multi_channel else "sdm"
            )
            if device_mic != mic:
                continue

            if is_close_talk:
                ct_wav_files = sorted(
                    f for f in _listdir_safe(device_path) if f.endswith(".wav")
                )
                for ct_wav in ct_wav_files:
                    ct_device_id = Path(ct_wav).stem  # e.g. "CT_21"
                    speaker = ct_device_to_speaker.get(ct_device_id, ct_device_id)
                    ct_recording = Recording.from_file(device_path / ct_wav)
                    ct_recording_id = f"{meeting}_close_talk_{ct_device_id}"
                    ct_recording.id = ct_recording_id

                    speaker_supervisions = []
                    for segment in transcription_json:
                        if segment["speaker_id"] != speaker:
                            continue
                        start_time = float(segment["start_time"])
                        end_time = float(segment["end_time"])
                        alignment = None

                        if create_word_alignment:
                            alignment = {"word": []}
                            for alig_text, alig_start_time, alig_end_time in segment[
                                "word_timing"
                            ]:
                                if "<" in alig_text or ">" in alig_text:
                                    continue
                                alignment["word"].append(
                                    AlignmentItem(
                                        symbol=alig_text,
                                        start=float(alig_start_time),
                                        duration=float(alig_end_time)
                                        - float(alig_start_time),
                                    )
                                )

                        speaker_supervisions.append(
                            SupervisionSegment(
                                id=f"{ct_recording_id}_{str(int(start_time * 100)).zfill(6)}_{str(int(end_time * 100)).zfill(6)}",
                                recording_id=ct_recording_id,
                                start=start_time,
                                duration=end_time - start_time,
                                channel=0,
                                text=segment["text"],
                                speaker=speaker,
                                alignment=alignment,
                            )
                        )

                    cuts.append(
                        MonoCut(
                            id=ct_recording_id,
                            start=0,
                            duration=ct_recording.duration,
                            channel=0,
                            supervisions=speaker_supervisions,
                            recording=ct_recording,
                        )
                    )
                continue  # skip sc/mc append logic for ihm devices
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
                cuts.append(
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
                cuts.append(
                    MonoCut(
                        id=recording.id,
                        start=0,
                        duration=recording.duration,
                        channel=0,
                        supervisions=supervisions,
                        recording=recording,
                    )
                )

    return cuts
