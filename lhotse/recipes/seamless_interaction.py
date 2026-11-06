"""
The Seamless Interaction dataset contains over 4,000 hours of face-to-face, two-person
conversations recorded by Meta with over 4,000 participants. Every participant has their own
video and their own speaker-bleed denoised audio track (48 kHz mono WAV), together with a
time-aligned transcript with word timestamps and voice activity segments.

Conversations are either "improvised" (guided scenarios with at least one professional
actor) or "naturalistic" (prompted conversations), and both labels have train, dev and test
splits. Files are named ``V<vendor>_S<session>_I<interaction>_P<participant>``.

Each interaction becomes one two-channel ``Recording`` with one ``AudioSource`` per
participant, ordered by participant ID, and each supervision is placed on its speaker's
channel. ``CutSet.from_manifests`` turns these into two-channel ``MultiCut`` objects,
the input layout of full-duplex spoken dialogue models.

The dataset is released under the CC BY-NC 4.0 license.

Website: https://ai.meta.com/research/seamless-interaction/
Data: https://huggingface.co/datasets/facebook/seamless-interaction
Code: https://github.com/facebookresearch/seamless_interaction
"""

import csv
import json
import logging
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError

from tqdm.auto import tqdm

from lhotse import (
    AudioSource,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.serialization import load_json, save_to_json
from lhotse.supervision import AlignmentItem
from lhotse.utils import Pathlike, resumable_download

SEAMLESS_INTERACTION_URL = "https://dl.fbaipublicfiles.com/seamless_interaction"
# The list of files in the dataset, pinned to the latest revision of the official repository.
SEAMLESS_INTERACTION_FILELIST_URL = (
    "https://raw.githubusercontent.com/facebookresearch/seamless_interaction/"
    "29f819d1d6610fabd0e9d9d3bea67afb644dde9e/assets/filelist.csv"
)
LABELS = ("improvised", "naturalistic")
SPLITS = ("train", "dev", "test")
FILE_ID = re.compile(
    r"(?P<interaction>(?P<vendor>V\d+)_S\d+_I\d+)_(?P<participant>P\w+)"
)


def download_seamless_interaction(
    target_dir: Pathlike = ".",
    labels: Sequence[str] = LABELS,
    splits: Sequence[str] = SPLITS,
    force_download: bool = False,
) -> Path:
    """
    Download the audio, transcripts and voice activity of Seamless Interaction from Meta's
    file server, without the video and motion features.

    The files are stored like in the official archives,
    ``{label}/{split}/{batch:04d}/{archive:04d}/{file_id}.wav`` and ``{file_id}.json``,
    so this directory can also be extended with the official download tools.
    The audio is 48 kHz 32-bit float WAV (about 11.5 MB per minute per participant):
    the dev and test splits together have about 4,300 audio files,
    and the whole dataset has about 129,000.

    :param target_dir: Pathlike, the path of the directory where the corpus will be stored.
    :param labels: which conversation types to download, "improvised" and/or "naturalistic".
    :param splits: which splits to download, among "train", "dev" and "test".
    :param force_download: bool, if True, download the files even if they already exist.
    :return: the path to the downloaded corpus.
    """
    corpus_dir = Path(target_dir) / "seamless_interaction"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    filelist = corpus_dir / "filelist.csv"
    resumable_download(
        SEAMLESS_INTERACTION_FILELIST_URL, filelist, force_download=force_download
    )
    with open(filelist, newline="") as f:
        entries = [
            entry
            for entry in csv.DictReader(f)
            if entry["label"] in labels and entry["split"] in splits
        ]

    for entry in tqdm(entries, desc="Downloading Seamless Interaction"):
        file_id, label, split = entry["file_id"], entry["label"], entry["split"]
        url = f"{SEAMLESS_INTERACTION_URL}/{label}/{split}"
        part_dir = (
            corpus_dir
            / label
            / split
            / f"{int(entry['batch_idx']):04d}"
            / f"{int(entry['archive_idx']):04d}"
        )
        part_dir.mkdir(parents=True, exist_ok=True)
        audio_path = part_dir / f"{file_id}.wav"
        json_path = part_dir / f"{file_id}.json"
        try:
            resumable_download(
                f"{url}/audio/{file_id}.wav", audio_path, force_download=force_download
            )
            if force_download or not json_path.is_file():
                save_to_json(_download_metadata(url, file_id), json_path)
        except HTTPError as e:
            logging.warning(f"Could not download {file_id}: {e}. Skipping it.")
            # resumable_download creates the file before the request fails.
            if audio_path.is_file() and audio_path.stat().st_size == 0:
                audio_path.unlink()

    return corpus_dir


def _download_metadata(url: str, file_id: str) -> dict:
    # The official archives store a participant's transcript and voice activity in one JSON.
    metadata = {"id": file_id}
    for kind in ("transcript", "vad"):
        with urllib.request.urlopen(f"{url}/metadata/{kind}/{file_id}.jsonl") as f:
            metadata[f"metadata:{kind}"] = [
                json.loads(line) for line in f if line.strip()
            ]
    return metadata


def prepare_seamless_interaction(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    labels: Sequence[str] = LABELS,
    splits: Sequence[str] = SPLITS,
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Prepare manifests for Seamless Interaction, with one two-channel recording per interaction.

    ``corpus_dir`` must contain ``{label}/{split}`` directories with the participants'
    ``{file_id}.wav`` audio and ``{file_id}.json`` transcripts at any depth below them,
    as written by :func:`download_seamless_interaction` or the official download tools
    (extracted archives). Both participants of an interaction are usually stored in different
    archives of the same batch; when only one of them is available, the interaction becomes
    a single-channel recording. Interactions without any transcript are skipped.

    The manifests are written while they are created, because the train splits have
    millions of supervisions with word alignments.

    :param corpus_dir: Pathlike, the path to the corpus.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param labels: which conversation types to prepare, "improvised" and/or "naturalistic".
    :param splits: which splits to prepare, among "train", "dev" and "test".
    :return: a Dict whose keys are the labels and the values are Dicts whose keys are the splits
        and the values are Dicts with the keys 'recordings' and 'supervisions' (opened lazily).
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)
    for label in labels:
        for split in splits:
            interactions = defaultdict(list)
            # Sorting by file name orders the participants (channels) by participant ID.
            audio_paths = (corpus_dir / label / split).rglob("*.wav")
            for audio_path in sorted(audio_paths, key=lambda p: p.name):
                match = FILE_ID.fullmatch(audio_path.stem)
                if match is not None:
                    interactions[match["interaction"]].append(audio_path)
            if not interactions:
                logging.warning(f"No audio found in {corpus_dir / label / split}")
                continue
            num_single = sum(len(paths) == 1 for paths in interactions.values())
            if num_single:
                logging.warning(
                    f"{num_single} of {len(interactions)} interactions in {label}/{split} "
                    f"have the audio of only one participant."
                )

            prefix = f"seamless-interaction-{label}"
            num_untranscribed = 0
            with (
                RecordingSet.open_writer(
                    output_dir / f"{prefix}_recordings_{split}.jsonl.gz"
                ) as rec_writer,
                SupervisionSet.open_writer(
                    output_dir / f"{prefix}_supervisions_{split}.jsonl.gz"
                ) as sup_writer,
            ):
                for paths in tqdm(
                    interactions.values(),
                    desc=f"Preparing Seamless Interaction {label}/{split}",
                ):
                    recording, segments = _prepare_interaction(paths)
                    if recording is None:
                        continue
                    if not segments:
                        num_untranscribed += 1
                        continue
                    recordings, supervisions = fix_manifests(
                        RecordingSet.from_recordings([recording]),
                        SupervisionSet.from_segments(segments),
                    )
                    validate_recordings_and_supervisions(recordings, supervisions)
                    for recording in recordings:
                        rec_writer.write(recording)
                    for segment in supervisions:
                        sup_writer.write(segment)
            if num_untranscribed:
                logging.warning(
                    f"Skipped {num_untranscribed} interactions in {label}/{split} "
                    f"without any transcript."
                )
            recordings = rec_writer.open_manifest()
            if recordings is not None:
                manifests[label][split] = {
                    "recordings": recordings,
                    "supervisions": sup_writer.open_manifest(),
                }

    return dict(manifests)


def _prepare_interaction(
    audio_paths: List[Path],
) -> Tuple[Optional[Recording], List[SupervisionSegment]]:
    participants = [Recording.from_file(path) for path in audio_paths]
    interaction_id = FILE_ID.fullmatch(audio_paths[0].stem)["interaction"]
    first = participants[0]
    if first.num_samples == 0 or any(
        (p.num_channels, p.sampling_rate, p.num_samples)
        != (1, first.sampling_rate, first.num_samples)
        for p in participants
    ):
        logging.warning(
            f"Skipping interaction {interaction_id}: its audio files are empty "
            f"or differ in channels, sampling rate or length."
        )
        return None, []
    recording = Recording(
        id=interaction_id,
        sources=[
            AudioSource(
                type="file", channels=[channel], source=participant.sources[0].source
            )
            for channel, participant in enumerate(participants)
        ],
        sampling_rate=first.sampling_rate,
        num_samples=first.num_samples,
        duration=first.duration,
    )

    supervisions = []
    for channel, audio_path in enumerate(audio_paths):
        match = FILE_ID.fullmatch(audio_path.stem)
        metadata_path = audio_path.with_suffix(".json")
        if not metadata_path.is_file():
            logging.warning(f"No transcript found for {audio_path}")
            continue
        # The transcript is optional: some participants have an empty one.
        segments = load_json(metadata_path).get("metadata:transcript") or []
        for idx, segment in enumerate(segments):
            duration = round(segment["end"] - segment["start"], ndigits=4)
            if duration <= 0:
                continue
            words = segment["words"]
            # A few words, mostly numbers, have no timestamps: keep a segment's
            # word alignment only when it covers the whole text.
            alignment = None
            if words and all(
                w.get("start") is not None and w.get("end") is not None for w in words
            ):
                alignment = {
                    "word": [
                        AlignmentItem(
                            symbol=w["word"],
                            start=round(w["start"], ndigits=4),
                            duration=round(w["end"] - w["start"], ndigits=4),
                            score=w.get("score"),
                        )
                        for w in words
                    ]
                }
            supervisions.append(
                SupervisionSegment(
                    id=f"{audio_path.stem}-{idx:04d}",
                    recording_id=interaction_id,
                    start=round(segment["start"], ndigits=4),
                    duration=duration,
                    channel=channel,
                    text=segment["transcript"],
                    language="English",
                    # Participant IDs are only unique within a vendor.
                    speaker=f"{match['vendor']}_{match['participant']}",
                    alignment=alignment,
                )
            )
    return recording, supervisions
