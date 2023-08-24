"""
This is a data preparation recipe for the National Corpus of Speech in Singaporean English.

The entire corpus is organised into a few parts.

Part 1 features about 1000 hours of prompted recordings of phonetically-balanced scripts from about 1000 local English speakers.

Part 2 presents about 1000 hours of prompted recordings of sentences randomly generated from words based on people, food, location, brands, etc, from about 1000 local English speakers as well. Transcriptions of the recordings have been done orthographically and are available for download.

Part 3 consists of about 1000 hours of conversational data recorded from about 1000 local English speakers, split into pairs. The data includes conversations covering daily life and of speakers playing games provided.

Parts 1 and 2 were recorded in quiet rooms using 3 microphones: a headset/ standing microphone (channel 0), a boundary microphone (channel 1), and a mobile phone (channel 3). Recordings that are available for download here have been down-sampled to 16kHz. Details of the microphone models used for each speaker as well as some corresponding non-personal and anonymized information can be found in the accompanying spreadsheets.

Part 3's recordings were split into 2 environments. In the Same Room environment where speakers were in same room, the recordings were done using 2 microphones: a close-talk mic and a boundary mic. In the Separate Room environment, speakers were separated into individual rooms. The recordings were done using 2 microphones in each room: a standing mic and a telephone.

Under Part 4, speakers were encouraged as best as possible to switch from Singapore English to their Mother Tongue languages. These recordings were done under two environments. In the Same Room recordings, speakers sit at least two metres apart and record using their mobile phones. In the Different Room environment, speakers would speak through each other via Zoom on their laptops, and recording using their mobile phones.

Under Part 5, speakers were made to speak following the 4 styles: Debate, Finance topics, Positive Emotion and Negative Emotions. All recordings were done in a Separate room session, via Zoom, where the audio is recorded using the mobile phone.

Under Part 6, speakers were made to speak following the 3 styles within either of the 3 designs: Design 1 (holiday/hotel/restaurant), Design 2 (bank, telephone, insurance), Design 3 (HDB, MOE, MSF). All recordings were done in a Separate room session, via Zoom, where the audio is recorded using the mobile phone.

We currently only support the part 3 recordings, in "same room close mic" and "separate rooms phone mic" environments.
"""
import itertools
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.parallel import parallel_map
from lhotse.qa import fix_manifests
from lhotse.utils import Pathlike

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NSC_PARTS = [
    "PART1_CHANNEL0",
    "PART1_CHANNEL1",
    "PART1_CHANNEL2",
    "PART2_CHANNEL0",
    "PART2_CHANNEL1",
    "PART2_CHANNEL2",
    "PART3_SameBoundaryMic",
    "PART3_SameCloseMic",
    "PART3_SeparateIVR",
    "PART3_SeparateStandingMic",
    "PART4_CodeswitchingDiffRoom",
    "PART4_CodeswitchingSameRoom",
    "PART5_Debate",
    "PART5_FinanceEmotion",
    "PART6_CallCentreDesign1",
    "PART6_CallCentreDesign2",
    "PART6_CallCentreDesign3",
]


@dataclass
class ScriptAudioDir:
    script_dir: Union[str, Path]
    audio_dir: Union[str, Path]

    def relative_to(self, parent: Union[str, Path]) -> "ScriptAudioDir":
        parent = Path(parent)
        return ScriptAudioDir(
            script_dir=parent / self.script_dir,
            audio_dir=parent / self.audio_dir,
        )


@dataclass
class HandlerMapping:
    handler: Callable[[str, ScriptAudioDir, int], dict]
    script_audio: ScriptAudioDir


# function forward declaration is not supported, wrap static data into function
def get_part_handler_map(corpus_dir: Path) -> Dict[str, HandlerMapping]:
    part_1_3_parent_dir = corpus_dir / "IMDA - National Speech Corpus"
    part_4_6_parent_dir = (
        corpus_dir
        / "IMDA - National Speech Corpus - Additional"
        / "IMDA - National Speech Corpus (Additional)"
    )
    # fmt: off
    part_handler_mapping: Dict[str, HandlerMapping] = {
        "PART1_CHANNEL0": HandlerMapping(handler=prepare_part1, script_audio=ScriptAudioDir(script_dir="PART1/DATA/CHANNEL0/SCRIPT", audio_dir="PART1/DATA/CHANNEL0/WAVE").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART1_CHANNEL1": HandlerMapping(handler=prepare_part1, script_audio=ScriptAudioDir(script_dir="PART1/DATA/CHANNEL1/SCRIPT", audio_dir="PART1/DATA/CHANNEL1/WAVE").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART1_CHANNEL2": HandlerMapping(handler=prepare_part1, script_audio=ScriptAudioDir(script_dir="PART1/DATA/CHANNEL2/SCRIPT", audio_dir="PART1/DATA/CHANNEL2/WAVE").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART2_CHANNEL0": HandlerMapping(handler=prepare_part2, script_audio=ScriptAudioDir(script_dir="PART2/DATA/CHANNEL0/SCRIPT", audio_dir="PART2/DATA/CHANNEL0/WAVE").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART2_CHANNEL1": HandlerMapping(handler=prepare_part2, script_audio=ScriptAudioDir(script_dir="PART2/DATA/CHANNEL1/SCRIPT", audio_dir="PART2/DATA/CHANNEL1/WAVE").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART2_CHANNEL2": HandlerMapping(handler=prepare_part2, script_audio=ScriptAudioDir(script_dir="PART2/DATA/CHANNEL2/SCRIPT", audio_dir="PART2/DATA/CHANNEL2/WAVE").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART3_SameBoundaryMic": HandlerMapping(handler=prepare_part3, script_audio=ScriptAudioDir(script_dir="PART3/Scripts Same", audio_dir="PART3/Audio Same BoundaryMic").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART3_SameCloseMic": HandlerMapping(handler=prepare_part3,script_audio= ScriptAudioDir(script_dir="PART3/Scripts Same", audio_dir="PART3/Audio Same CloseMic").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART3_SeparateIVR": HandlerMapping(handler=prepare_part3, script_audio=ScriptAudioDir(script_dir="PART3/Scripts Separate", audio_dir="PART3/Audio Separate IVR").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART3_SeparateStandingMic": HandlerMapping(handler=prepare_part3,script_audio=ScriptAudioDir(script_dir="PART3/Scripts Separate", audio_dir="PART3/Audio Separate StandingMic").relative_to(part_1_3_parent_dir)), # noqa: E501
        "PART4_CodeswitchingDiffRoom": HandlerMapping(handler=prepare_part4, script_audio=ScriptAudioDir(script_dir="PART4/Codeswitching/Diff Room Scripts", audio_dir="PART4/Codeswitching/Diff Room Audio").relative_to(part_4_6_parent_dir)), # noqa: E501
        "PART4_CodeswitchingSameRoom": HandlerMapping(handler=prepare_part4, script_audio=ScriptAudioDir(script_dir="PART4/Codeswitching/Same Room Scripts", audio_dir="PART4/Codeswitching/Same Room Audio").relative_to(part_4_6_parent_dir)), # noqa: E501
        "PART5_Debate": HandlerMapping(handler=prepare_part5, script_audio=ScriptAudioDir(script_dir="PART5/Debate Scripts", audio_dir="PART5/Debate Audio").relative_to(part_4_6_parent_dir)), # noqa: E501
        "PART5_FinanceEmotion": HandlerMapping(handler=prepare_part5, script_audio=ScriptAudioDir(script_dir="PART5/Finance + Emotion Scripts", audio_dir="PART5/Finance + Emotions Audio").relative_to(part_4_6_parent_dir)), # noqa: E501
        "PART6_CallCentreDesign1": HandlerMapping(handler=prepare_part6, script_audio=ScriptAudioDir(script_dir="PART6/Call Centre Design 1/Scripts", audio_dir="PART6/Call Centre Design 1/Audio").relative_to(part_4_6_parent_dir)), # noqa: E501
        "PART6_CallCentreDesign2": HandlerMapping(handler=prepare_part6, script_audio=ScriptAudioDir(script_dir="PART6/Call Centre Design 2/Scripts", audio_dir="PART6/Call Centre Design 2/Audio").relative_to(part_4_6_parent_dir)), # noqa: E501
        "PART6_CallCentreDesign3": HandlerMapping(handler=prepare_part6, script_audio=ScriptAudioDir(script_dir="PART6/Call Centre Design 3/Scripts", audio_dir="PART6/Call Centre Design 3/Audio").relative_to(part_4_6_parent_dir)), # noqa: E501
    }
    return part_handler_mapping
    # fmt: on


def check_dependencies():
    try:
        import textgrids  # noqa
    except:
        raise ImportError(
            "NSC data preparation requires the forked 'textgrids' package to be installed. "
            "Please install it with 'pip install git+https://github.com/pzelasko/Praat-textgrids' "
            "and try again."
        )


def prepare_nsc(
    corpus_dir: Pathlike,
    dataset_part: str = "PART3_SameCloseMic",
    output_dir: Optional[Pathlike] = None,
    num_jobs=1,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path to the raw corpus distribution.
    :param dataset_part: str, name of the dataset part to be prepared.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    check_dependencies()
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    part_handler_map = get_part_handler_map(corpus_dir)
    if dataset_part in part_handler_map:
        handler_map = part_handler_map[dataset_part]
        manifests = handler_map.handler(
            dataset_part,
            handler_map.script_audio,
            num_jobs,
        )
    else:
        raise ValueError(f"Unknown dataset part: {dataset_part}")

    # Fix the manifests to make sure they are valid
    recordings, supervisions = fix_manifests(**manifests)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        supervisions.to_file(output_dir / f"nsc_supervisions_{dataset_part}.jsonl.gz")
        recordings.to_file(output_dir / f"nsc_recordings_{dataset_part}.jsonl.gz")

    return manifests


def prepare_part1(
    part_name: str,
    script_audio_dir: ScriptAudioDir,
    num_jobs: int = 1,
    *args,
    **kwargs,
):
    recordings = []
    supervisions = []
    audio_dir = Path(script_audio_dir.audio_dir)
    script_dir = Path(script_audio_dir.script_dir)
    channel = int(part_name[-1:])  # E.g. PART1_CHANNEL0
    assert channel in {0, 1, 2}
    extract_to_dir = audio_dir / "extracted"
    if not extract_to_dir.exists():
        extract_to_dir.mkdir()
    speaker_zip_files = [f for f in audio_dir.glob("SPEAKER*.zip")]
    for speaker_manifests in tqdm(
        parallel_map(
            _parse_part1_speaker,
            speaker_zip_files,
            itertools.repeat(script_dir),
            itertools.repeat(channel),
            itertools.repeat(extract_to_dir),
            num_jobs=num_jobs,
        ),
        total=len(speaker_zip_files),
        desc=f"Creating manifests for {part_name}",
    ):
        recordings.extend(speaker_manifests["recordings"])
        supervisions.extend(speaker_manifests["supervisions"])
    return {
        "recordings": RecordingSet.from_recordings(recordings),
        "supervisions": SupervisionSet.from_segments(supervisions),
    }


def prepare_part2(
    part_name: str,
    script_audio_dir: ScriptAudioDir,
    num_jobs: int = 1,
    *args,
    **kwargs,
):
    return prepare_part1(
        part_name=part_name,
        script_audio_dir=script_audio_dir,
        num_jobs=num_jobs,
        *args,
        **kwargs,
    )


def prepare_part3(
    part_name: str,
    script_audio_dir: ScriptAudioDir,
    num_jobs: int = 1,
    *args,
    **kwargs,
):
    from textgrids import TextGrid

    assert (
        part_name != "PART3_SameBoundaryMic"
    ), "The recipe too different, currently not supported"

    def textgrid_field_id_resolver(audio_file: Union[str, Path]) -> Tuple[Dict, str]:
        audio_file = Path(audio_file)
        script_dir = Path(script_audio_dir.script_dir)
        if part_name == "PART3_SeparateIVR":
            script_file_stem = audio_file.parent.name + "_" + audio_file.stem
            script_file = script_dir / f"{script_file_stem}.TextGrid"
            textgrid_key = script_file_stem
        else:
            script_file = script_dir / f"{audio_file.stem}.TextGrid"
            textgrid_key = audio_file.stem
        coding = _detect_textgrid_encoding(script_file)
        tg = TextGrid(script_file, coding=coding)
        return tg, textgrid_key

    return prepare_textgrid_based_part(
        part_name=part_name,
        script_audio_dir=script_audio_dir,
        textgrid_loader=textgrid_field_id_resolver,
        num_jobs=num_jobs,
        *args,
        **kwargs,
    )


def prepare_part4(
    part_name: str,
    script_audio_dir: ScriptAudioDir,
    num_jobs: int = 1,
    *args,
    **kwargs,
):
    from textgrids import TextGrid

    def textgrid_field_id_resolver(
        audio_file: Union[str, Path]
    ) -> Tuple[TextGrid, str]:
        audio_file = Path(audio_file)
        script_dir = Path(script_audio_dir.script_dir)
        script_file = script_dir / f"{audio_file.stem}.TextGrid"
        coding = _detect_textgrid_encoding(script_file)
        tg = TextGrid(script_file, coding=coding)
        # NSC textgrid change unpredictably, getting it from dict key
        textgrid_key = next(iter(tg.keys()))
        return tg, textgrid_key

    return prepare_textgrid_based_part(
        part_name=part_name,
        script_audio_dir=script_audio_dir,
        textgrid_loader=textgrid_field_id_resolver,
        num_jobs=num_jobs,
        *args,
        **kwargs,
    )


def prepare_part5(
    part_name: str,
    script_audio_dir: ScriptAudioDir,
    num_jobs: int = 1,
    *args,
    **kwargs,
):
    from textgrids import TextGrid

    def textgrid_field_id_resolver(
        audio_file: Union[str, Path]
    ) -> Tuple[TextGrid, str]:
        audio_file = Path(audio_file)
        script_dir = Path(script_audio_dir.script_dir)
        script_file = script_dir / f"{audio_file.stem}.TextGrid"
        coding = _detect_textgrid_encoding(script_file)
        tg = TextGrid(script_file, coding=coding)
        # NSC textgrid change unpredictably, getting it from dict key
        textgrid_key = next(iter(tg.keys()))
        return tg, textgrid_key

    return prepare_textgrid_based_part(
        part_name=part_name,
        script_audio_dir=script_audio_dir,
        textgrid_loader=textgrid_field_id_resolver,
        num_jobs=num_jobs,
        *args,
        **kwargs,
    )


def prepare_part6(
    part_name: str,
    script_audio_dir: ScriptAudioDir,
    num_jobs: int = 1,
    *args,
    **kwargs,
):
    return prepare_part5(
        part_name=part_name,
        script_audio_dir=script_audio_dir,
        num_jobs=num_jobs,
        *args,
        **kwargs,
    )


def prepare_textgrid_based_part(
    part_name: str,
    script_audio_dir: ScriptAudioDir,
    textgrid_loader: Callable[[Union[str, Path]], Tuple[Dict, str]],
    num_jobs: int = 1,
    *args,
    **kwargs,
):
    """Prepare part that use textgrid to storing script like: PART3, PART4, PART5, PART6

    Args:
        part_path (Path): path to part
        script_audio_dir (Path): root dir of audios
        script_file_resolver (Callable): a function resolve an audio path in to textgrid script file and it's key
        num_jobs (int): number of workers to process data
    """
    check_dependencies()

    recordings = []
    supervisions = []
    audio_dir = Path(script_audio_dir.audio_dir)
    audio_files = [
        f
        for f in itertools.chain(
            audio_dir.rglob("**/*.wav"), audio_dir.rglob("**/*.WAV")
        )
    ]
    processed_recordings = set()
    for audio_path in tqdm(
        audio_files,
        total=len(audio_files),
        desc=f"Creating manifests for {part_name}",
    ):
        try:
            recording_id = f"{part_name}_{audio_path.stem}"
            assert (
                recording_id not in processed_recordings
            ), f'Duplicated recording id "{recording_id}", audio path: "{audio_path}"'
            processed_recordings.add(recording_id)
            recording = Recording.from_file(audio_path, recording_id=recording_id)
            tg, textgrid_key = textgrid_loader(audio_path)
            segments = [
                s
                for s in (
                    SupervisionSegment(
                        id=f"{recording.id}-{idx}",
                        recording_id=recording.id,
                        start=segment.xmin,
                        # We're trimming the last segment's duration as it exceeds the actual duration of the recording.
                        # This is safe because if we end up with a zero/negative duration, the validation will catch it.
                        duration=min(
                            round(segment.xmax - segment.xmin, ndigits=8),
                            recording.duration - segment.xmin,
                        ),
                        text=segment.text,
                        language="Singaporean English",
                        speaker=recording_id,
                    )
                    for idx, segment in enumerate(tg[textgrid_key])
                    if segment.text not in ("<S>", "<Z>")  # skip silences
                )
                if s.duration > 0  # NSC has some bad segments
            ]

            supervisions.extend(segments)
            recordings.append(recording)
        except Exception:
            with logging_redirect_tqdm():
                logger.warning(f'Error when processing "{audio_path}" - skipping...')
    return {
        "recordings": RecordingSet.from_recordings(recordings),
        "supervisions": SupervisionSet.from_segments(supervisions),
    }


def _parse_part1_speaker(
    speaker_zip_file: Pathlike,
    script_dir: Pathlike,
    channel: int,
    extract_to_dir: Optional[Path] = None,
):
    script_session_dir_mapping = _preprocess_part1_speaker(
        speaker_zip_file=speaker_zip_file,
        script_dir=script_dir,
        channel=channel,
        extract_to_dir=extract_to_dir,
    )
    sessions_dir = []
    scripts_file = []
    for script_file, session_dir in script_session_dir_mapping.items():
        scripts_file.append(script_file)
        sessions_dir.append(session_dir)
    recordings: List[Recording] = []
    supervisions: List[SupervisionSegment] = []
    for r, s in [
        _parse_part1_script(sc_f, ss_d)
        for sc_f, ss_d in zip(scripts_file, sessions_dir)
    ]:
        recordings.extend(r)
        supervisions.extend(s)

    return {
        "recordings": RecordingSet.from_recordings(recordings),
        "supervisions": SupervisionSet.from_segments(supervisions),
    }


def _preprocess_part1_speaker(
    speaker_zip_file: Pathlike,
    script_dir: Pathlike,
    channel: int,
    extract_to_dir: Optional[Path] = None,
) -> Dict[Path, Path]:
    """Extract PART1/PART2 speaker audio

    Args:
        speaker_zip_file (Pathlike): Path to speaker zipped audio file
        script_dir (Pathlike): Path to script dir of the channel
        channel (int): Channel of the PART, we can parse from $script_dir but it is not necessary
        extract_to_dir (Optional[Path]): Directory to extract zipped audio file, default to parent dir of $speaker_zip_file

    Returns:
        Dict[Path, Path]: Mapping of script file -> speaker's session dir
    """
    speaker_zip_file = Path(speaker_zip_file)
    script_dir = Path(script_dir)
    if extract_to_dir is None:
        extract_to_dir = speaker_zip_file.parent
    speaker_audio_dir = extract_to_dir / speaker_zip_file.stem
    if not speaker_audio_dir.exists():
        with zipfile.ZipFile(speaker_zip_file) as zf:
            zf.extractall(extract_to_dir)
    else:
        with logging_redirect_tqdm():
            logger.warning(
                f'Reusing "{speaker_audio_dir}" as extracted "{speaker_zip_file}"'
                " since it is existed already"
            )
    sessions_dir = [f for f in speaker_audio_dir.glob("SESSION*")]
    scripts_file: List[Path] = []
    for session_dir in sessions_dir:
        spk_id = speaker_audio_dir.stem.removeprefix("SPEAKER")
        session_number = session_dir.stem.removeprefix("SESSION")
        session_script_file = script_dir / f"{channel}{spk_id}{session_number}.TXT"
        scripts_file.append(session_script_file)
    return {sr_f: ss_d for sr_f, ss_d in zip(scripts_file, sessions_dir)}


def _parse_part1_script(
    script_file: Path, session_dir
) -> Tuple[List[Recording], List[SupervisionSegment]]:
    recordings = []
    supervisions = []

    with open(script_file, "r", encoding="utf-8-sig") as fr:
        previous_audio_id = ""
        previous_text = ""
        for line in fr:
            columns = line.rstrip("\n").split("\t")
            if previous_audio_id != "" and columns[0] != previous_audio_id:
                # empty id indicate normalized text of prior line
                if columns[0] == "":
                    previous_text = columns[1]
                recording, segment = _create_part1_single_record(
                    session_dir, previous_audio_id, previous_text
                )
                if recording:
                    recordings.append(recording)
                    supervisions.append(segment)
                previous_audio_id = previous_text = ""
            else:
                previous_audio_id = columns[0]
                previous_text = columns[1]
        if previous_audio_id:
            recording, segment = _create_part1_single_record(
                session_dir, previous_audio_id, previous_text
            )
            if recording:
                recordings.append(recording)
                supervisions.append(segment)
    return recordings, supervisions


def _create_part1_single_record(
    session_dir: Path, _audio_id: str, text: str
) -> Tuple[Optional[Recording], Optional[SupervisionSegment]]:
    audio_file = session_dir / f"{_audio_id}.WAV"
    try:
        recording = Recording.from_file(audio_file, recording_id=_audio_id)
        segment = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0,
            duration=recording.duration,
            text=text,
        )
        return recording, segment
    except FileNotFoundError:
        with logging_redirect_tqdm():
            logger.warning(
                f'Recording audio of script "{_audio_id}" '
                f'can not be found in "{session_dir}"'
            )
    except Exception as e:
        with logging_redirect_tqdm():
            logger.error(f"Error occurred with {audio_file}", e)
    return None, None


def _detect_textgrid_encoding(textgrid_file: Pathlike) -> Optional["str"]:
    """_summary_

    Returns:
        str: encoding of the file or None if it is binary file or undetectable
    """
    import charset_normalizer

    textgrid_binary_mark = b"ooBinaryFile\x08TextGrid"
    with open(textgrid_file, "rb") as fr:
        checking_bytes = fr.read(10 * 2**10)  # 10kB
    if checking_bytes[: len(textgrid_binary_mark)] == textgrid_binary_mark:
        return None
    else:
        charset_match = charset_normalizer.from_bytes(checking_bytes).best()  # type: ignore
        if charset_match is None:
            return None  # the most widely used I know
        else:
            # prefer utf-8 over ascii
            return (
                charset_match.encoding if charset_match.encoding != "ascii" else "utf-8"
            )
