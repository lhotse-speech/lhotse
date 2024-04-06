"""
The CHiME-6 dataset is a collection of over 50 hours of conversational speech recordings
collected from twenty real dinner parties that have taken place in real homes. The
recordings have been made using multiple 4-channel microphone arrays and have been
fully transcribed.

The dataset features:

- simultaneous recordings from multiple microphone arrays;
- real conversation, i.e. talkers speaking in a relaxed and unscripted fashion;
- a range of room acoustics from 20 different homes each with two or three separate
  recording areas;
- real domestic noise backgrounds, e.g., kitchen appliances, air conditioning,
  movement, etc.

Fully-transcribed utterances are provided in continuous audio with ground truth speaker
labels and start/end time annotations for segmentation.

The dataset was used for the 5th and 6th CHiME Speech Separation and Recognition
Challenge. Further information and an open source baseline speech recognition system
are available online (http://spandh.dcs.shef.ac.uk/chime_challenge/chime2018).

NOTE: The CHiME-5 and CHiME-6 datasets are the same, with the only difference that
additional software was provided in CHiME-6 to perform array synchronization. We expect
that users have already downloaded the CHiME-5 dataset here:
https://licensing.sheffield.ac.uk/product/chime5

NOTE: Users can also additionally perform array synchronization as described here:
https://github.com/kaldi-asr/kaldi/blob/master/egs/chime6/s5_track1/local/generate_chime6_data.sh
We also provide this option in the `prepare_chime6` function.
"""

import itertools
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.recipes.utils import TimeFormatConverter, normalize_text_chime6
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations, resumable_download

# fmt: off
DATASET_PARTS = {
    "train": [
        "S03", "S04", "S05", "S06", "S07", "S08", "S12", "S13",
        "S16", "S17", "S18", "S19", "S20", "S22", "S23", "S24",
    ],
    "dev": ["S02", "S09"],
    "eval": ["S01", "S21"],
}

DATASET_PARTS_CHIME7 = {
    "train": [
        "S03", "S04", "S05", "S06", "S07", "S08", "S12", "S13",
        "S16", "S17", "S18", "S22", "S23", "S24",
    ],
    "dev": ["S02", "S09"],
    "eval": ["S01", "S19", "S20", "S21"],
}

# fmt: on
CHIME6_AUDIO_EDITS_JSON = "https://raw.githubusercontent.com/chimechallenge/chime6-synchronisation/master/chime6_audio_edits.json"
CHIME6_MD5SUM_FILE = "https://raw.githubusercontent.com/chimechallenge/chime6-synchronisation/master/audio_md5sums.txt"


def download_chime6(
    target_dir: Pathlike = ".",
) -> Path:
    """
    Download the original dataset. This cannot be done automatically because of the
    license agreement. Please visit the following URL and download the dataset manually:
    https://licensing.sheffield.ac.uk/product/chime5
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :return: the path to downloaded and extracted directory with data.
    """
    print("We cannot download the CHiME-6 dataset automatically.")
    print("Please visit the following URL and download the dataset manually:")
    print("https://licensing.sheffield.ac.uk/product/chime5")
    print("Then, please extract the tar files to the following directory:")
    print(f"{target_dir}")
    return Path(target_dir)


def prepare_chime6(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dataset_parts: Optional[Union[str, Sequence[str]]] = "all",
    mic: str = "mdm",
    use_reference_array: bool = False,
    perform_array_sync: bool = False,
    verify_md5_checksums: bool = False,
    num_jobs: int = 1,
    num_threads_per_job: int = 1,
    sox_path: Pathlike = "/usr/bin/sox",
    normalize_text: str = "kaldi",
    use_chime7_split: bool = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir, either the original CHiME-5
        data or the synchronized CHiME-6 data. If former, the `perform_array_sync`
        must be True.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, the microphone type to use, choose from "ihm" (close-talk) or "mdm"
        (multi-microphone array) settings. For MDM, there are 6 array devices with 4
        channels each, so the resulting recordings will have 24 channels.
    :param use_reference_array: bool, if True, use the reference array for MDM setting.
        Only the supervision segments have the reference array information in the
        `channel` field. The recordings will still have all the channels in the array.
        Note that the train set does not have the reference array information.
    :param perform_array_sync: Bool, if True, perform array synchronization based on:
        https://github.com/chimechallenge/chime6-synchronisation
    :param num_jobs: int, the number of jobs to run in parallel for array synchronization.
    :param num_threads_per_job: int, number of threads to use per job for clock drift
        correction. Large values may require more memory, so we recommend using a job
        scheduler.
    :param sox_path: Pathlike, the path to the sox v14.4.2 binary. Note that different
        versions of sox may produce different results.
    :param normalize_text: str, the text normalization method, choose from "none", "upper",
        "kaldi". The "kaldi" method is the same as Kaldi's text normalization method for
        CHiME-6.
    :param verify_md5_checksums: bool, if True, verify the md5 checksums of the audio files.
        Note that this step is slow so we recommend only doing it once. It can be sped up
        by using the `num_jobs` argument.
    :param use_chime7_split: bool, if True, use the new split for CHiME-7 challenge.
    :return: a Dict whose key is the dataset part ("train", "dev" and "eval"), and the
        value is Dicts with the keys 'recordings' and 'supervisions'.

    NOTE: If `perform_array_sync` is True, the synchronized data will be written to
        `output_dir`/CHiME6. This may take a long time and the output will occupy
        approximately 160G of storage. We will also create a temporary directory for
        processing, so the required storage in total will be approximately 300G.
    """
    import soundfile as sf

    assert mic in ["ihm", "mdm"], "mic must be either 'ihm' or 'mdm'."

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if "all" in dataset_parts:
        dataset_parts = list(DATASET_PARTS.keys())
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]
    assert set(dataset_parts).issubset(
        set(DATASET_PARTS.keys())
    ), f"dataset_parts must be one of {list(DATASET_PARTS.keys())}. Found {dataset_parts}"

    sessions = list(
        itertools.chain.from_iterable([DATASET_PARTS[part] for part in dataset_parts])
    )

    if perform_array_sync:
        if not output_dir:
            raise ValueError(
                "If `perform_array_sync` is True, `output_dir` must be specified."
            )
        chime6_dir = output_dir / "CHiME6"
        chime6_dir.mkdir(parents=True, exist_ok=True)

        # Create directories for train, dev, and test audio and transcriptions
        for part in dataset_parts:
            (chime6_dir / "audio" / part).mkdir(parents=True, exist_ok=True)
            (chime6_dir / "transcriptions" / part).mkdir(parents=True, exist_ok=True)

        # Check sox version
        sox_version = (
            subprocess.check_output([sox_path, "--version"])
            .decode("utf-8")
            .strip()
            .split(" ")[-1]
        )
        assert sox_version == "v14.4.2", (
            "The sox version must be 14.4.2. "
            "Please download the sox v14.4.2 binary from "
            "https://sourceforge.net/projects/sox/files/sox/14.4.2/ "
            "and specify the path to the binary with the `sox_path` argument."
            "You can also install it in a Conda environment with the following command: "
            "conda install -c conda-forge sox=14.4.2"
        )

        chime6_array_synchronizer = Chime6ArraySynchronizer(
            corpus_dir=corpus_dir,
            output_dir=chime6_dir,
            sox_path=sox_path,
            num_workers=num_threads_per_job,
        )

        num_jobs = min(num_jobs, len(sessions))  # since there are 20 sessions
        logging.info(
            f"Performing array synchronization with {num_jobs} jobs. This may "
            "take a long time."
        )
        with ProcessPoolExecutor(max_workers=num_jobs) as ex:
            futures = [
                ex.submit(
                    chime6_array_synchronizer.synchronize_session,
                    session=session,
                )
                for session in sessions
            ]
            _ = wait(futures)
    else:
        chime6_dir = Path(corpus_dir)

    # Verify MD5 checksums for all audio files if requested
    if verify_md5_checksums:
        if _verify_md5_checksums(chime6_dir, num_workers=num_jobs, sessions=sessions):
            print("MD5 checksums verified. All OK.")
        else:
            raise RuntimeError(
                "MD5 checksums do not match. Please prepare the array-synchronized CHiME-6 "
                "dataset again."
            )

    # Reference array is only applicable for MDM setting
    use_reference_array = use_reference_array and mic == "mdm"

    manifests = defaultdict(dict)

    for part in dataset_parts:
        recordings = []
        supervisions = []

        # Since CHiME-7 uses a different split, we need to change the sessions
        if use_chime7_split:
            DATASET_PARTS[part] = DATASET_PARTS_CHIME7[part]
        # Also, if the session is S19 or S20, we will look for its audio and transcriptions
        # in the train set, since it was originally in train.

        # First we create the recordings
        if mic == "ihm":
            global_spk_channel_map = {}
            for session in DATASET_PARTS[part]:
                part_ = (
                    "train" if use_chime7_split and session in ["S19", "S20"] else part
                )

                audio_paths = [
                    p for p in (chime6_dir / "audio" / part_).rglob(f"{session}_P*.wav")
                ]

                if len(audio_paths) == 0:
                    raise FileNotFoundError(
                        f"No audio found for session {session} in {part_} set."
                    )

                sources = []
                # NOTE: Each headset microphone is binaural
                for idx, audio_path in enumerate(audio_paths):
                    sources.append(
                        AudioSource(
                            type="file",
                            channels=[2 * idx, 2 * idx + 1],
                            source=str(audio_path),
                        )
                    )
                    spk_id = audio_path.stem.split("_")[1]
                    global_spk_channel_map[(session, spk_id)] = [2 * idx, 2 * idx + 1]

                audio_sf = sf.SoundFile(str(audio_paths[0]))

                recordings.append(
                    Recording(
                        id=session,
                        sources=sources,
                        sampling_rate=int(audio_sf.samplerate),
                        num_samples=audio_sf.frames,
                        duration=audio_sf.frames / audio_sf.samplerate,
                    )
                )

        else:
            for session in DATASET_PARTS[part]:
                part_ = (
                    "train" if use_chime7_split and session in ["S19", "S20"] else part
                )

                audio_paths = [
                    p for p in (chime6_dir / "audio" / part_).rglob(f"{session}_U*.wav")
                ]

                sources = []
                for idx, audio_path in enumerate(sorted(audio_paths)):
                    sources.append(
                        AudioSource(type="file", channels=[idx], source=str(audio_path))
                    )

                audio_sf = sf.SoundFile(str(audio_paths[0]))

                recordings.append(
                    Recording(
                        id=session,
                        sources=sources,
                        sampling_rate=int(audio_sf.samplerate),
                        num_samples=audio_sf.frames,
                        duration=audio_sf.frames / audio_sf.samplerate,
                    )
                )

        recordings = RecordingSet.from_recordings(recordings)

        def _get_channel(spk_id, session, ref=None):
            if mic == "ihm":
                return global_spk_channel_map[(session, spk_id)]
            else:
                recording = recordings[session]
                return (
                    list(range(recording.num_channels))
                    if not ref
                    else [i for i, s in enumerate(recording.sources) if ref in s.source]
                )

        # Then we create the supervisions
        for session in DATASET_PARTS[part]:
            part_ = "train" if use_chime7_split and session in ["S19", "S20"] else part

            with open(chime6_dir / "transcriptions" / part_ / f"{session}.json") as f:
                transcript = json.load(f)
                for idx, segment in enumerate(transcript):
                    spk_id = segment["speaker"]
                    channel = _get_channel(
                        spk_id,
                        session,
                        ref=segment["ref"]
                        if use_reference_array and part != "train"
                        else None,
                    )
                    start = TimeFormatConverter.hms_to_seconds(segment["start_time"])
                    end = TimeFormatConverter.hms_to_seconds(segment["end_time"])
                    if start >= end:  # some segments may have negative duration
                        continue
                    supervisions.append(
                        SupervisionSegment(
                            id=f"{session}-{idx}",
                            recording_id=session,
                            start=start,
                            duration=add_durations(end, -start, sampling_rate=16000),
                            channel=channel,
                            text=normalize_text_chime6(
                                segment["words"], normalize=normalize_text
                            ),
                            language="English",
                            speaker=spk_id,
                            custom={
                                "location": segment["location"],
                            }
                            if part != "train" and "location" in segment
                            else None,
                        )
                    )

        supervisions = SupervisionSet.from_segments(supervisions)

        recording_set, supervision_set = fix_manifests(
            recordings=recordings, supervisions=supervisions
        )
        # Fix manifests
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            mic_affix = f"{mic}-ref" if use_reference_array else mic
            supervision_set.to_file(
                output_dir / f"chime6-{mic_affix}_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"chime6-{mic}_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests


def _verify_md5_checksums(
    corpus_dir: Pathlike, num_workers: int = 1, sessions: Optional[List[str]] = None
) -> bool:
    import hashlib

    # First download checksum file and read it into a dictionary
    temp_dir = Path(tempfile.mkdtemp())
    checksum_file = temp_dir / "md5sums.txt"
    resumable_download(
        CHIME6_MD5SUM_FILE, str(checksum_file), desc="Downloading checksum file"
    )
    checksums = {}
    with open(checksum_file, "r") as f:
        for line in f:
            checksum, filename = line.strip().split(" ", maxsplit=1)
            checksums[Path(filename).stem] = checksum

    # Now verify the checksums
    def _verify_checksum(file: Pathlike) -> bool:
        checksum = hashlib.md5(open(str(file), "rb").read()).hexdigest()
        filename = str(file.stem)
        if filename in checksums and checksum != checksums[filename]:
            return False
        return True

    all_files = list(corpus_dir.rglob("*.wav"))
    if sessions is not None:
        all_files = [f for f in all_files if f.stem.split("_")[0] in sessions]

    print(f"Verifying checksum with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(tqdm(ex.map(_verify_checksum, all_files), total=len(all_files)))

    return all(results)


# The following class is based on functions from:
# https://github.com/chimechallenge/chime6-synchronisation
# We have made 2 changes to get some speed-up:
# 1. We combine all channels in an array for applying frame drop correction.
# 2. We apply multi-threading with 4 threads for clock drift correction.


class Chime6ArraySynchronizer:
    """
    Class for synchronizing CHiME6 array recordings.
    """

    def __init__(
        self,
        corpus_dir: Pathlike,
        output_dir: Pathlike,
        sox_path: Pathlike = "sox",
        num_workers: int = 1,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.sox_path = Path(sox_path)
        self.num_workers = num_workers

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Download the audio edits JSON file
        audio_edits_json = self.output_dir / "audio_edits.json"
        resumable_download(CHIME6_AUDIO_EDITS_JSON, str(audio_edits_json))
        with open(audio_edits_json) as f:
            self.audio_edits = dict(json.load(f))

        # Check SOX path
        if not str(self.sox_path).endswith("sox"):
            self.sox_path = self.sox_path / "sox"

    def synchronize_session(self, session: str) -> None:
        """
        Synchronize a single CHiME6 session.
        """
        temp_dir = Path(
            tempfile.mkdtemp(prefix=f"chime6_{session}_", dir=self.output_dir)
        )

        if session not in self.audio_edits:
            logging.warning(f"No audio edits found for session {session}")
            return

        session_audio_edits = self.audio_edits[session]

        print(f"Correcting {session} for frame drops...")
        self.correct_frame_drops(temp_dir, session, frame_drops=session_audio_edits)

        print(f"Correcting {session} for clock drift...")
        self.correct_clock_drift(
            temp_dir,
            session,
            linear_fit=session_audio_edits,
            num_threads=self.num_workers,
        )

        print(f"Adjusting timestamps in {session} JSON files...")
        self.adjust_json_timestamps(session, linear_fit=session_audio_edits)

        # clean up
        shutil.rmtree(temp_dir)

        return

    def correct_frame_drops(
        self,
        output_dir: Pathlike,
        session: str,
        frame_drops: Optional[Dict[str, Any]] = None,
    ) -> None:

        # For binaural recordings, we just create symbolic links to the original files
        session_binaural_wavs = sorted(
            (self.corpus_dir / "audio").rglob(f"{session}_P*.wav")
        )
        for wav in session_binaural_wavs:
            wav_relative_path = wav.relative_to(self.corpus_dir)
            wav_output_path = output_dir / wav_relative_path
            wav_output_path.parent.mkdir(exist_ok=True, parents=True)
            os.symlink(wav, wav_output_path)

        # For array recordings, we need to apply the edits. We first group the channels by
        # their corresponding array. For example, the wav names are like: S02_U01.CH2.wav
        session_array_wavs = sorted(
            (self.corpus_dir / "audio").rglob(f"{session}_U*.wav")
        )
        array_wavs = defaultdict(list)
        for wav in session_array_wavs:
            array_id = wav.stem.split(".")[0].split("_")[-1]
            array_wavs[array_id].append(wav)

        # Then we apply the edits to each array
        for array_id, wavs in array_wavs.items():
            if array_id not in frame_drops:
                logging.warning(
                    f"Array {array_id} in session {session} has no frame drops information."
                )
                continue

            in_wavs, out_wavs = [], []
            for wav in wavs:
                wav_relative_path = wav.relative_to(self.corpus_dir)
                wav_output_path = output_dir / wav_relative_path
                in_wavs.append(wav)
                out_wavs.append(wav_output_path)

            # Apply the edits to the wavs
            self._apply_edits_to_wav(in_wavs, out_wavs, frame_drops[array_id]["edits"])

        return

    def _apply_edits_to_wav(
        self, in_wavs: Pathlike, out_wavs: Pathlike, edits: List[List[int]]
    ) -> None:
        import soundfile as sf

        x = np.concatenate(
            [Recording.from_file(wav).load_audio() for wav in in_wavs], axis=0
        )
        # Pre-allocate space for editted signal
        max_space = edits[-1][2] + edits[-1][1] - edits[-1][0]
        x_new = np.zeros(shape=(x.shape[0], max_space), dtype=x.dtype)
        length_x = x.shape[1]

        for edit in edits:
            in_from = edit[0]
            in_to = min(edit[1], length_x)
            out_from = edit[2]
            out_to = out_from + in_to - in_from
            if in_from > length_x:
                break
            x_new[:, out_from - 1 : out_to] = x[:, in_from - 1 : in_to]

        # Trim off excess
        x_new = x_new[:, 0:out_to]

        # Write to file
        for i, wav in enumerate(out_wavs):
            sf.write(
                file=str(wav),
                data=np.expand_dims(x_new[i], axis=1),
                samplerate=16000,
                format="WAV",
            )
        return

    def correct_clock_drift(
        self,
        corpus_dir: Pathlike,
        session: str,
        linear_fit: Optional[Dict[str, Any]] = None,
        num_threads: int = 1,
    ) -> None:
        session_wavs = sorted((corpus_dir / "audio").rglob(f"{session}_*.wav"))
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = []
            for wav in session_wavs:
                wav_relative_path = wav.relative_to(corpus_dir)
                wav_output_path = self.output_dir / wav_relative_path

                # Get binaural mic id or array id
                mic_id = wav.name.split(".")[0].split("_")[-1]

                if mic_id not in linear_fit:
                    logging.warning(
                        f"Channel {mic_id} in session {session} has no clock drift information."
                    )
                    continue

                futures.append(
                    ex.submit(
                        self._apply_clock_drift_correction,
                        wav,
                        wav_output_path,
                        linear_fit[mic_id],
                    )
                )

            for future in tqdm(futures):
                future.result()
        return

    def _apply_clock_drift_correction(
        self,
        in_wav: Pathlike,
        out_wav: Pathlike,
        linear_fit: Dict[str, Union[int, float, List]],
    ) -> None:
        speeds = linear_fit["speed"]
        padding = linear_fit["padding"]
        filename = in_wav.stem

        sox_cmd = str(self.sox_path)
        in_wav = str(in_wav)
        out_wav = str(out_wav)

        if isinstance(speeds, list):
            # multisegment fit - only needed for S01_U02 and S01_U05
            starts = padding
            ends = padding[1:] + [-1]  # -1 means end of signal
            command_concat = [sox_cmd]
            samples_to_lose = 0

            tmpdir = tempfile.mkdtemp(dir=self.output_dir)

            for seg, (start, end, speed) in enumerate(zip(starts, ends, speeds)):
                # print(seg, start, speed, end)
                of1 = tmpdir + "/" + filename + "." + str(seg) + ".wav"
                of2 = tmpdir + "/" + filename + ".x" + str(seg) + ".wav"

                command1 = [sox_cmd, "-D", "-R", in_wav, of1]
                if seg == 0:
                    # 'start' has a different meaning for first segment
                    # Acts like padding does in the simple one-piece case
                    if start < 0:
                        start = -start
                        trim = ["trim", f"{start}s"]
                    else:
                        trim = ["pad", f"{start}s", "0s", "trim", "0s"]
                else:
                    start += samples_to_lose  # may need to truncate some samples
                    trim = ["trim", f"{int(start)}s"]
                command1 += trim
                if end > 0:  # segment of given duration
                    duration = end - start
                    command1 += [f"{duration}s"]

                if speed < 0:
                    # Negative speed means backward in time,
                    # Effectively have to remove some samples
                    # This happen in S01_U05.
                    samples_to_lose = -duration / speed
                else:
                    samples_to_lose = 0
                    command2 = [sox_cmd, "-D", "-R", of1, of2, "speed", str(speed)]
                    subprocess.call(command1)
                    subprocess.call(command2)
                    command_concat.append(of2)

            command_concat.append(out_wav)
            subprocess.call(command_concat)

            # Clean up
            shutil.rmtree(tmpdir)
        else:
            # The -R to suppress dithering so that command produces identical results each time
            command = [sox_cmd, "-D", "-R", in_wav, out_wav, "speed", str(speeds)]
            if padding > 0:
                # Change speed and pad
                command += ["pad", f"{padding}s", "0s"]
            else:
                # Change speed and trim
                command += ["trim", f"{-padding}s"]
                # Note, the order of speed vs trim/pad makes a difference
                # I believe sox actually applies the speed transform after the padding.
                # but speed is so close to 1 and the padding so short that it will
                # come out about the same either way around.

            logging.info(f"Running command: {' '.join(command)}")
            subprocess.call(command)
            return

    def adjust_json_timestamps(
        self, session: str, linear_fit: Optional[Dict[str, Any]] = None
    ) -> None:
        in_json = next((self.corpus_dir / "transcriptions").rglob(f"{session}.json"))
        relative_path = in_json.relative_to(self.corpus_dir)
        out_json = self.output_dir / relative_path

        corrected_utts = []
        with open(in_json, "r") as fin, open(out_json, "w") as fout:
            data = json.load(fin)
            for segment in data:
                if "speaker" not in segment:
                    continue
                pid = segment["speaker"]
                speed = linear_fit[pid]["speed"]
                padding = linear_fit[pid]["padding"]
                delta_t = padding / 16000  # convert to seconds
                start_time = (
                    TimeFormatConverter.hms_to_seconds(
                        segment["start_time"]["original"]
                    )
                    / speed
                    + delta_t
                )
                end_time = (
                    TimeFormatConverter.hms_to_seconds(segment["end_time"]["original"])
                    / speed
                    + delta_t
                )
                segment["start_time"] = TimeFormatConverter.seconds_to_hms(start_time)
                segment["end_time"] = TimeFormatConverter.seconds_to_hms(end_time)
                corrected_utts.append(segment)

            json.dump(corrected_utts, fout, indent=2)
        return
