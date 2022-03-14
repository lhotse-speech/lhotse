import logging
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress

# files containing transcripts
heroico_dataset_answers = "heroico-answers.txt"
heroico_dataset_recordings = "heroico-recordings.txt"
usma_dataset = "usma-prompts.txt"

folds = ("train", "devtest", "test")


def download_heroico(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    url: Optional[str] = "http://www.openslr.org/resources/39",
) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_name = f"LDC2006S37.tar.gz"
    tar_path = target_dir / tar_name
    completed_detector = target_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {tar_name} because {completed_detector} exists.")
        return
    if force_download or not tar_path.is_file():
        urlretrieve_progress(
            f"{url}/{tar_name}", filename=tar_path, desc="Downloading Heroico"
        )
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    completed_detector.touch()

    return target_dir


class HeroicoMetaData(NamedTuple):
    audio_path: Pathlike
    audio_info: Any
    text: str


class UttInfo(NamedTuple):
    fold: str
    speaker: str
    prompt_id: str
    subcorpus: str
    utterance_id: str
    transcript: str


def prepare_heroico(
    speech_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
        Returns the manifests which consist of the Recordings and Supervisions

        :param speech_dir: Pathlike, the path of the speech data dir.
    param transcripts_dir: Pathlike, the path of the transcript data dir.
        :param output_dir: Pathlike, the path where to write the manifests.
        :return: a Dict whose key is the fold, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    import soundfile

    speech_dir = Path(speech_dir)
    transcript_dir = Path(transcript_dir)
    assert speech_dir.is_dir(), f"No such directory: {speech_dir}"
    assert transcript_dir.is_dir(), f"No such directory: {transcript_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    manifests = defaultdict(dict)

    # set some patterns to match fields in transcript files and filenames
    answers_line_pattern = re.compile("\d+/\d+\t.+")
    answers_path_pattern = re.compile("Answers_Spanish")
    heroico_recitations_line_pattern = re.compile("\d+\t.+")
    heroico_recitations_devtest_path_pattern = re.compile("Recordings_Spanish")
    heroico_recitations_train_path_pattern = re.compile("Recordings_Spanish")
    usma_line_pattern = re.compile("s\d+\t.+")
    usma_native_demo_pattern = re.compile(
        "usma/native\-[fm]\-\w+\-\S+\-\S+\-\S+\-\S+\-\w+\d+"
    )
    usma_native_path_pattern = re.compile("usma/native")
    usma_native_prompt_id_pattern = re.compile("s\d+")
    usma_nonnative_demo_pattern = re.compile(
        "nonnative\-[fm]\-[a-zA-Z]+\d*\-[a-zA-Z]+\-[a-zA-Z]+\-[a-zA-Z]+\-[a-zA-Z]+\-[a-zA-Z]+\d+"
    )
    usma_nonnative_path_pattern = re.compile("nonnative.+\.wav")

    # Generate a mapping: utt_id -> (audio_path, audio_info, text)

    transcripts = defaultdict(dict)
    # store answers trnscripts
    answers_trans_path = Path(transcript_dir, heroico_dataset_answers)
    with open(answers_trans_path, encoding="iso-8859-1") as f:
        for line in f:
            line = line.rstrip()
            # some recordings do not have a transcript, skip them here
            if not answers_line_pattern.match(line):
                continue
            # IDs have the form speaker/prompt_id
            spk_utt, text = line.split(maxsplit=1)
            spk_id, prompt_id = spk_utt.split("/")
            utt_id = "-".join(["answers", spk_id, prompt_id])
            transcripts[utt_id] = text

    # store heroico recitations transcripts
    heroico_recitations_trans_path = Path(transcript_dir, heroico_dataset_recordings)
    with open(heroico_recitations_trans_path, encoding="iso-8859-1") as f:
        for line in f:
            line = line.rstrip()
            if not heroico_recitations_line_pattern.match(line):
                continue
            idx, text = line.split(maxsplit=1)
            utt_id = "-".join(["heroico-recitations", idx])
            transcripts[utt_id] = text

    # store usma transcripts
    usma_trans_path = Path(transcript_dir, usma_dataset)
    with open(usma_trans_path, encoding="iso-8859-1") as f:
        for line in f:
            line = line.rstrip()
            if not usma_line_pattern.match(line):
                continue
            idx, text = line.split(maxsplit=1)
            utt_id = "-".join(["usma", idx])
            transcripts[utt_id] = text

    # store utterance info
    audio_paths = speech_dir.rglob("*.wav")
    uttdata = {}
    for wav_file in audio_paths:
        wav_path = Path(wav_file)
        path_components = wav_path.parts
        pid = wav_path.stem
        if re.findall(answers_path_pattern, str(wav_file)):
            # store utternce info for Heroico Answers
            spk = wav_path.parts[-2]
            utt_id = "-".join(["answers", spk, pid])
            if utt_id not in transcripts:
                uttdata[str(wav_file)] = None
                continue
            uttdata[str(wav_file)] = UttInfo(
                fold="train",
                speaker=spk,
                prompt_id=pid,
                subcorpus="answers",
                utterance_id=utt_id,
                transcript=transcripts[utt_id],
            )
        elif re.findall(usma_native_path_pattern, str(wav_file)):
            # store utterance info for usma native data
            spk = wav_path.parts[-2]
            utt_id = "-".join(["usma", spk, pid])
            trans_id = "-".join(["usma", pid])
            if not usma_native_demo_pattern.match(spk):
                uttdata[str(wav_file)] = None
            if not usma_native_prompt_id_pattern.match(pid):
                uttdata[str(wav_file)] = None
                continue
            uttdata[str(wav_file)] = UttInfo(
                fold="test",
                speaker=spk,
                prompt_id=pid,
                subcorpus="usma",
                utterance_id=utt_id,
                transcript=transcripts[trans_id],
            )
        elif re.findall(usma_nonnative_path_pattern, str(wav_file)):
            # store utterance data for usma nonnative data
            spk = wav_path.parts[-2]
            utt_id = "-".join(["usma", spk, pid])
            trans_id = "-".join(["usma", pid])
            if not usma_nonnative_demo_pattern.match(spk):
                uttdata[str(wav_file)] = None
                continue
            uttdata[str(wav_file)] = UttInfo(
                fold="test",
                speaker=spk,
                prompt_id=pid,
                subcorpus="usma",
                utterance_id=utt_id,
                transcript=transcripts[trans_id],
            )
        elif int(pid) <= 354 or int(pid) >= 562:
            # store utterance info for heroico recitations for train dataset
            spk = wav_path.parts[-2]
            utt_id = "-".join(["heroico-recitations", spk, pid])
            trans_id = "-".join(["heroico-recitations", pid])
            uttdata[str(wav_file)] = UttInfo(
                fold="train",
                speaker=spk,
                prompt_id=pid,
                subcorpus="heroico-recitations",
                utterance_id=utt_id,
                transcript=transcripts[trans_id],
            )
        elif int(pid) > 354 and int(pid) < 562:
            spk = wav_path.parts[-2]
            utt_id = "-".join(["heroico-recitations-repeats", spk, pid])
            trans_id = "-".join(["heroico-recitations-repeats", pid])
            uttdata[str(wav_file)] = UttInfo(
                fold="devtest",
                speaker=spk,
                prompt_id=pid,
                subcorpus="heroico-recitations-repeats",
                utterance_id=utt_id,
                transcript=transcripts[trans_id],
            )
        else:
            logging.warning(f"No such file: {wav_file}")

    audio_paths = speech_dir.rglob("*.wav")
    audio_files = [w for w in audio_paths]

    for fld in folds:
        metadata = {}
        for wav_file in audio_files:
            wav_path = Path(wav_file)
            # skip files with no record
            if not uttdata[str(wav_file)]:
                continue
            # only process the current fold
            if uttdata[str(wav_file)].fold != fld:
                continue
            path_components = wav_path.parts
            prompt_id = wav_path.stem
            # info[0]: info of the raw audio (e.g. channel number, sample rate, duration ... )
            # info[1]: info about the encoding (e.g. FLAC/ALAW/ULAW ...)
            info = soundfile.info(str(wav_file))
            spk = wav_path.parts[-2]
            utt_id = "-".join([uttdata[str(wav_file)].subcorpus, spk, prompt_id])
            metadata[utt_id] = HeroicoMetaData(
                audio_path=wav_file,
                audio_info=info,
                text=uttdata[str(wav_file)].transcript,
            )

        # Audio
        audio = RecordingSet.from_recordings(
            Recording(
                id=idx,
                sources=[
                    AudioSource(
                        type="file", channels=[0], source=str(metadata[idx].audio_path)
                    )
                ],
                sampling_rate=int(metadata[idx].audio_info.samplerate),
                num_samples=metadata[idx].audio_info.frames,
                duration=metadata[idx].audio_info.duration,
            )
            for idx in metadata
        )

        # Supervision
        supervision = SupervisionSet.from_segments(
            SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=audio.recordings[idx].duration,
                channel=0,
                language="Spanish",
                speaker=idx.split("-")[-2],
                text=metadata[idx].text,
            )
            for idx in audio.recordings
        )

        validate_recordings_and_supervisions(audio, supervision)

        if output_dir is not None:
            supervision.to_json(output_dir / f"supervisions_{fld}.json")
            audio.to_json(output_dir / f"recordings_{fld}.json")

        manifests[fld] = {"recordings": audio, "supervisions": supervision}

    return manifests
