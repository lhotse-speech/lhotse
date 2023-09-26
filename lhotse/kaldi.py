import logging
import math
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lhotse.audio import AudioSource, Recording, RecordingSet, info
from lhotse.features import Features, FeatureSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
    Seconds,
    add_durations,
    compute_num_samples,
    fastcopy,
    is_module_available,
    to_list,
)


def floor_duration_to_milliseconds(
    duration: float,
) -> float:
    """
    Floor the duration to multiplies of 0.001 seconds.
    This is to avoid float precision problems with workflows like:
      lhotse kaldi import ...
      lhotse fix ...
      ./local/compute_fbank_imported.py (from icefall)
      lhotse cut trim-to-supervisions ...
      ./local/validate_manifest.py ... (from icefall)

    Without flooring, there were different lengths:
      Supervision end time 1093.33995833 is larger than cut end time 1093.3399375

    This is still within the 2ms tolerance in K2SpeechRecognitionDataset::validate_for_asr():
      https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L201
    """
    return math.floor(1000 * duration) / 1000


def get_duration(
    path: Pathlike,
) -> Optional[float]:
    """
    Read a audio file, it supports pipeline style wave path and real waveform.

    :param path: Path to an audio file or a Kaldi-style pipe.
    :return: float duration of the recording, in seconds or `None` in case of read error.
    """
    path = str(path)
    if path.strip().endswith("|"):
        if not is_module_available("kaldi_native_io"):
            raise ValueError(
                "To read Kaldi's data dir where wav.scp has 'pipe' inputs, "
                "please 'pip install kaldi_native_io' first."
            )
        import kaldi_native_io

        try:
            wave = kaldi_native_io.read_wave(path)
            assert (
                wave.data.shape[0] == 1
            ), f"Expect 1 channel. Given {wave.data.shape[0]}"

            return floor_duration_to_milliseconds(wave.duration)
        except:  # exception type from kaldi_native_io ? (std::runtime_error via pybind11)
            return None  # report a read error (recovery from C++ exception)

    audio_info = info(path)
    return floor_duration_to_milliseconds(audio_info.duration)


def load_kaldi_data_dir(
    path: Pathlike,
    sampling_rate: int,
    frame_shift: Optional[Seconds] = None,
    map_string_to_underscores: Optional[str] = None,
    use_reco2dur: bool = True,
    num_jobs: int = 1,
    feature_type: str = "kaldi-fbank",
) -> Tuple[RecordingSet, Optional[SupervisionSet], Optional[FeatureSet]]:
    """
    Load a Kaldi data directory and convert it to a Lhotse RecordingSet and
    SupervisionSet manifests. For this to work, at least the wav.scp file must exist.
    SupervisionSet is created only when a segments file exists. reco2dur is used by
    default when exists (to enforce reading the duration from the audio files
    themselves, please set use_reco2dur = False.
    All the other files (text, utt2spk, etc.) are optional, and some of them might
    not be handled yet. In particular, feats.scp files are ignored.

    :param path: Path to the Kaldi data directory.
    :param sampling_rate: Sampling rate of the recordings.
    :param frame_shift: Optional, if specified, we will create a Features manifest
        and store the frame_shift value in it.
    :param map_string_to_underscores: optional string, when specified, we will replace
        all instances of this string in SupervisonSegment IDs to underscores.
        This is to help with handling underscores in Kaldi
        (see :func:`.export_to_kaldi`). This is also done for speaker IDs.
    :param use_reco2dur: If True, we will use the reco2dur file to read the durations
        of the recordings. If False, we will read the durations from the audio files
        themselves.
    :param num_jobs: Number of parallel jobs to use when reading the audio files.
    """
    path = Path(path)
    assert path.is_dir()

    def fix_id(t: str) -> str:
        if map_string_to_underscores is None:
            return t
        return t.replace(map_string_to_underscores, "_")

    # must exist for RecordingSet
    recordings = load_kaldi_text_mapping(path / "wav.scp", must_exist=True)
    reco2dur = path / "reco2dur"
    if use_reco2dur and reco2dur.is_file():
        durations = load_kaldi_text_mapping(reco2dur, float_vals=True)
        assert len(durations) == len(recordings), (
            "The duration file reco2dur does not "
            "have the same length as the  wav.scp file"
        )
    else:
        # ProcessPoolExecutor hanging observed for datasets with >100k recordings.
        # Using large chunks to be processed per child processes is advised here:
        # https://docs.python.org/3/library/concurrent.futures.html
        #
        # num_chunks = num_jobs * 10, e.g. 250
        chunksize = max(1, len(recordings) // (num_jobs * 10))
        with ProcessPoolExecutor(max_workers=num_jobs) as ex:
            dur_vals = list(
                ex.map(get_duration, recordings.values(), chunksize=chunksize)
            )

        durations = dict(zip(recordings.keys(), dur_vals))

    # remove recordings with 'None' duration (i.e. there was a read error)
    for recording_id, dur_value in durations.items():
        if dur_value is None:
            logging.warning(
                f"[{recording_id}] Could not get duration. "
                f"Failed to read audio from `{recordings[recording_id]}`. "
                "Dropping the recording from manifest."
            )
            del recordings[recording_id]
    # make sure not too many utterances were dropped
    if len(recordings) < len(durations) * 0.8:
        raise RuntimeError(
            f'Failed to load more than 20% utterances of the dataset: "{path}"'
        )

    # assemble the new RecordingSet
    recording_set = RecordingSet.from_recordings(
        Recording(
            id=recording_id,
            sources=[
                AudioSource(
                    type="command" if path_or_cmd.endswith("|") else "file",
                    channels=[0],
                    source=path_or_cmd[:-1]
                    if path_or_cmd.endswith("|")
                    else path_or_cmd,
                )
            ],
            sampling_rate=sampling_rate,
            num_samples=compute_num_samples(durations[recording_id], sampling_rate),
            duration=durations[recording_id],
        )
        for recording_id, path_or_cmd in recordings.items()
    )

    supervision_set = None
    segments = path / "segments"
    utt2spk_f = path / "utt2spk"
    feats_scp = path / "feats.scp"

    # load mapping from utt_id to start and duration
    utt_id_to_start_and_duration = load_start_and_duration(
        segments_path=segments,
        feats_path=feats_scp,
        frame_shift=frame_shift,
    )

    if segments.is_file():
        supervisions = []
        with segments.open() as f:
            supervision_segments = [sup_string.strip().split() for sup_string in f]

        texts = load_kaldi_text_mapping(path / "text")
        speakers = load_kaldi_text_mapping(path / "utt2spk")
        genders = load_kaldi_text_mapping(path / "spk2gender")
        languages = load_kaldi_text_mapping(path / "utt2lang")

        for segment_id, recording_id, start, end in supervision_segments:
            if utt_id_to_start_and_duration:
                # use duration computed from feats.scp
                _, duration = utt_id_to_start_and_duration[segment_id]
            else:
                # to support <end-time> == -1 in segments file
                # https://kaldi-asr.org/doc/extract-segments_8cc.html
                # <end-time> of -1 means the segment runs till the end of the WAV file
                duration = add_durations(
                    float(end) if end != "-1" else durations[recording_id],
                    -float(start),
                    sampling_rate=sampling_rate,
                )
            supervisions.append(
                SupervisionSegment(
                    id=fix_id(segment_id),
                    recording_id=recording_id,
                    start=float(start),
                    duration=duration,
                    channel=0,
                    text=texts[segment_id],
                    language=languages[segment_id],
                    speaker=fix_id(speakers[segment_id]),
                    gender=genders[speakers[segment_id]],
                )
            )
        supervision_set = SupervisionSet.from_segments(supervisions)
    elif utt2spk_f.is_file():
        # segments file does not exist => provided supervision
        # corresponds to whole recordings
        speakers = load_kaldi_text_mapping(path / "utt2spk")
        assert len(speakers) == len(recording_set)

        texts = load_kaldi_text_mapping(path / "text")
        genders = load_kaldi_text_mapping(path / "spk2gender")
        languages = load_kaldi_text_mapping(path / "utt2lang")
        supervision_set = SupervisionSet.from_segments(
            SupervisionSegment(
                id=fix_id(rec_id),
                recording_id=rec_id,
                start=0.0,
                duration=durations[rec_id],
                channel=0,
                text=texts[rec_id],
                language=languages[rec_id],
                speaker=fix_id(spkr),
                gender=genders[spkr],
            )
            for rec_id, spkr in speakers.items()
        )

    feature_set = None
    if feats_scp.exists() and is_module_available("kaldi_native_io"):
        if frame_shift is not None:
            import kaldi_native_io

            from lhotse.features.io import KaldiReader

            features = []
            with open(feats_scp) as f:
                for line in f:
                    utt_id, ark = line.strip().split(maxsplit=1)
                    mat_shape = kaldi_native_io.MatrixShape.read(ark)

                    # start time is from segments
                    if utt_id_to_start_and_duration:
                        start, duration = utt_id_to_start_and_duration[utt_id]
                    else:
                        start = 0
                        duration = mat_shape.num_rows * frame_shift

                    features.append(
                        Features(
                            type=feature_type,
                            num_frames=mat_shape.num_rows,
                            num_features=mat_shape.num_cols,
                            frame_shift=frame_shift,
                            sampling_rate=sampling_rate,
                            start=start,
                            duration=duration,
                            storage_type=KaldiReader.name,
                            storage_path=ark,
                            storage_key=utt_id,
                            recording_id=supervision_set[fix_id(utt_id)].recording_id
                            if supervision_set is not None
                            else utt_id,
                            channels=0,
                        )
                    )
            feature_set = FeatureSet.from_features(features)
        else:
            warnings.warn(
                "Failed to import Kaldi 'feats.scp' to Lhotse: "
                "frame_shift must be not None. "
                "Feature import omitted."
            )

    return recording_set, supervision_set, feature_set


def export_to_kaldi(
    recordings: RecordingSet,
    supervisions: SupervisionSet,
    output_dir: Pathlike,
    map_underscores_to: Optional[str] = None,
    prefix_spk_id: Optional[bool] = False,
):
    """
    Export a pair of ``RecordingSet`` and ``SupervisionSet`` to a Kaldi data
    directory. It even supports recordings that have multiple channels but
    the recordings will still have to have a single ``AudioSource``.

    The ``RecordingSet`` and ``SupervisionSet`` must be compatible, i.e. it must
    be possible to create a ``CutSet`` out of them.

    :param recordings: a ``RecordingSet`` manifest.
    :param supervisions: a ``SupervisionSet`` manifest.
    :param output_dir: path where the Kaldi-style data directory will be created.
    :param map_underscores_to: optional string with which we will replace all
        underscores. This helps avoid issues with Kaldi data dir sorting.
    :param prefix_spk_id: add speaker_id as a prefix of utterance_id (this is to
        ensure correct sorting inside files which is required by Kaldi)

    .. note:: If you export a ``RecordingSet`` with multiple channels, then the
        resulting Kaldi data directory may not be back-compatible with Lhotse
        (i.e. you won't be able to import it back to Lhotse in the same form).
        This is because Kaldi does not inherently support multi-channel recordings,
        so we have to break them down into single-channel recordings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if map_underscores_to is not None:
        supervisions = supervisions.map(
            lambda s: fastcopy(
                s,
                id=s.id.replace("_", map_underscores_to),
                speaker=s.speaker.replace("_", map_underscores_to),
            )
        )

    if prefix_spk_id:
        supervisions = supervisions.map(lambda s: fastcopy(s, id=f"{s.speaker}-{s.id}"))

    if all(r.num_channels == 1 for r in recordings):
        # if all the recordings are single channel, we won't add
        # the channel id affix to retain back compatibility
        # and the ability to receive back the same utterances after
        # importing the exported directory back

        # wav.scp
        save_kaldi_text_mapping(
            data={
                recording.id: make_wavscp_channel_string_map(
                    source,
                    sampling_rate=recording.sampling_rate,
                    transforms=recording.transforms,
                )[0]
                for recording in recordings
                for source in recording.sources
            },
            path=output_dir / "wav.scp",
        )
        # segments
        save_kaldi_text_mapping(
            data={
                sup.id: f"{sup.recording_id} {sup.start} {sup.end}"
                for sup in supervisions
            },
            path=output_dir / "segments",
        )
        # reco2dur
        save_kaldi_text_mapping(
            data={recording.id: recording.duration for recording in recordings},
            path=output_dir / "reco2dur",
        )

        # text
        save_kaldi_text_mapping(
            data={sup.id: sup.text for sup in supervisions},
            path=output_dir / "text",
        )
        # utt2spk
        save_kaldi_text_mapping(
            data={sup.id: sup.speaker for sup in supervisions},
            path=output_dir / "utt2spk",
        )
        # utt2dur
        save_kaldi_text_mapping(
            data={sup.id: sup.duration for sup in supervisions},
            path=output_dir / "utt2dur",
        )
        # utt2lang [optional]
        if all(s.language is not None for s in supervisions):
            save_kaldi_text_mapping(
                data={sup.id: sup.language for sup in supervisions},
                path=output_dir / "utt2lang",
            )
        # utt2gender [optional]
        if all(s.gender is not None for s in supervisions):
            save_kaldi_text_mapping(
                data={sup.id: sup.gender for sup in supervisions},
                path=output_dir / "utt2gender",
            )

    else:

        save_kaldi_text_mapping(
            data={
                f"{recording.id}_{channel}": make_wavscp_channel_string_map(
                    source,
                    sampling_rate=recording.sampling_rate,
                    transforms=recording.transforms,
                )[channel]
                for recording in recordings
                for source in recording.sources
                for channel in source.channels
            },
            path=output_dir / "wav.scp",
        )

        # reco2dur
        save_kaldi_text_mapping(
            data={
                f"{recording.id}_{channel}": recording.duration
                for recording in recordings
                for source in recording.sources
                for channel in source.channels
            },
            path=output_dir / "reco2dur",
        )

        # segments
        save_kaldi_text_mapping(
            data={
                sup.id
                + f"-{channel}": f"{sup.recording_id}_{channel} {sup.start} {sup.end}"
                for sup in supervisions
                for channel in to_list(sup.channel)
            },
            path=output_dir / "segments",
        )

        # text
        save_kaldi_text_mapping(
            data={
                sup.id + f"-{channel}": sup.text
                for sup in supervisions
                for channel in to_list(sup.channel)
            },
            path=output_dir / "text",
        )
        # utt2spk
        save_kaldi_text_mapping(
            data={
                sup.id + f"-{channel}": sup.speaker
                for sup in supervisions
                for channel in to_list(sup.channel)
            },
            path=output_dir / "utt2spk",
        )
        # utt2dur
        save_kaldi_text_mapping(
            data={
                sup.id + f"-{channel}": sup.duration
                for sup in supervisions
                for channel in to_list(sup.channel)
            },
            path=output_dir / "utt2dur",
        )
        # utt2lang [optional]
        if all(s.language is not None for s in supervisions):
            save_kaldi_text_mapping(
                data={
                    sup.id + f"-{channel}": sup.language
                    for sup in supervisions
                    for channel in to_list(sup.channel)
                },
                path=output_dir / "utt2lang",
            )
        # utt2gender [optional]
        if all(s.gender is not None for s in supervisions):
            save_kaldi_text_mapping(
                data={
                    sup.id + f"-{channel}": sup.gender
                    for sup in supervisions
                    for channel in to_list(sup.channel)
                },
                path=output_dir / "utt2gender",
            )


def load_start_and_duration(
    segments_path: Path = None,
    feats_path: Path = None,
    frame_shift: Optional[Seconds] = None,
) -> Dict[Tuple, None]:
    """
    Load start time from segments and duration from feats,
    when both segments and feats.scp are available.
    """
    utt_id_to_start_and_duration = {}
    if (
        segments_path.is_file()
        and feats_path.is_file()
        and is_module_available("kaldi_native_io")
        and frame_shift is not None
    ):
        import kaldi_native_io

        with segments_path.open() as segments_f, feats_path.open() as feats_f:
            for segments_line, feats_line in zip(segments_f, feats_f):
                segment_id, _, start, _ = segments_line.strip().split()
                utt_id, ark = feats_line.strip().split(maxsplit=1)
                if segment_id != utt_id:
                    raise ValueError(f"{segments_path} and {feats_path} not aligned.")

                mat_shape = kaldi_native_io.MatrixShape.read(ark)
                duration = mat_shape.num_rows * frame_shift

                utt_id_to_start_and_duration[utt_id] = (
                    float(start),
                    duration,
                )
    return utt_id_to_start_and_duration


def load_kaldi_text_mapping(
    path: Path, must_exist: bool = False, float_vals: bool = False
) -> Dict[str, Optional[str]]:
    """Load Kaldi files such as utt2spk, spk2gender, text, etc. as a dict."""
    mapping = defaultdict(lambda: None)
    if path.is_file():
        with path.open() as f:
            mapping = dict(line.strip().split(maxsplit=1) for line in f)
        if float_vals:
            mapping = {key: float(val) for key, val in mapping.items()}
    elif must_exist:
        raise ValueError(f"No such file: {path}")
    return mapping


def save_kaldi_text_mapping(data: Dict[str, Any], path: Path):
    """Save flat dicts to Kaldi files such as utt2spk, spk2gender, text, etc."""
    with path.open("w") as f:
        for key, value in sorted(data.items()):
            print(key, value, file=f)


def make_wavscp_channel_string_map(
    source: AudioSource, sampling_rate: int, transforms: Optional[List[Dict]] = None
) -> Dict[int, str]:
    if source.type == "url":
        raise ValueError("URL audio sources are not supported by Kaldi.")
    elif source.type == "command":
        if len(source.channels) != 1:
            raise ValueError(
                "Command audio multichannel sources are not supported yet."
            )
        return {0: f"{source.source} |"}
    elif source.type == "file":
        if (
            Path(source.source).suffix == ".wav"
            and len(source.channels) == 1
            and transforms is None
        ):
            # Note: for single-channel waves, we don't need to invoke ffmpeg; but
            #       for multi-channel waves, Kaldi is going to complain.
            audios = dict()
            for channel in source.channels:
                audios[channel] = source.source
            return audios
        if Path(source.source).suffix == ".sph":
            # we will do this specifically using the sph2pipe because
            # ffmpeg does not support shorten compression, which is sometimes
            # used in the sph files
            audios = dict()
            for channel in source.channels:
                audios[channel] = (
                    f"sph2pipe {source.source} -f wav -c {channel+1} -p | "
                    "ffmpeg -threads 1"
                    f" -i pipe:0 -ar {sampling_rate} -f wav -threads 1 pipe:1 |"
                )

            return audios
        else:
            # Handles non-WAVE audio formats and multi-channel WAVEs.
            audios = dict()
            for channel in source.channels:
                if len(source.channels) == 1:
                    # it is single channel
                    audios[channel] = (
                        f"ffmpeg -threads 1 -i {source.source} -ar {sampling_rate} "
                        f"-map_channel 0.0.0  -f wav -threads 1 pipe:1 |"
                    )
                else:
                    audios[channel] = (
                        f"ffmpeg -threads 1 -i {source.source} -ar {sampling_rate} "
                        f"-map_channel 0.0.{channel}  -f wav -threads 1 pipe:1 |"
                    )
            return audios

    else:
        raise ValueError(f"Unknown AudioSource type: {source.type}")
