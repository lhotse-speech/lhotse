import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lhotse import FeatureSet, Features, Seconds
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.audio import audioread_info
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
    add_durations,
    compute_num_samples,
    fastcopy,
    is_module_available,
)


def get_duration(
    path: Pathlike,
) -> float:
    """
    Read a audio file, it supports pipeline style wave path and real waveform.

    :param path: Path to an audio file or a Kaldi-style pipe.
    :return: float duration of the recording, in seconds.
    """
    path = str(path)
    if path.strip().endswith("|"):
        if not is_module_available("kaldi_native_io"):
            raise ValueError(
                "To read Kaldi's data dir where wav.scp has 'pipe' inputs, "
                "please 'pip install kaldi_native_io' first."
            )
        import kaldi_native_io

        wave_info = kaldi_native_io.read_wave_info(path)
        assert wave_info.num_channels == 1, wave_info.num_channels

        return wave_info.duration
    try:
        # Try to parse the file using pysoundfile first.
        import soundfile

        info = soundfile.info(path)
    except:
        # Try to parse the file using audioread as a fallback.
        info = audioread_info(path)
    return info.duration


def load_kaldi_data_dir(
    path: Pathlike,
    sampling_rate: int,
    frame_shift: Optional[Seconds] = None,
    map_string_to_underscores: Optional[str] = None,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, Optional[SupervisionSet], Optional[FeatureSet]]:
    """
    Load a Kaldi data directory and convert it to a Lhotse RecordingSet and
    SupervisionSet manifests. For this to work, at least the wav.scp file must exist.
    SupervisionSet is created only when a segments file exists.
    All the other files (text, utt2spk, etc.) are optional, and some of them might
    not be handled yet. In particular, feats.scp files are ignored.

    :param map_string_to_underscores: optional string, when specified, we will replace
        all instances of this string in SupervisonSegment IDs to underscores.
        This is to help with handling underscores in Kaldi
        (see :func:`.export_to_kaldi`). This is also done for speaker IDs.
    """
    path = Path(path)
    assert path.is_dir()

    def fix_id(t: str) -> str:
        if map_string_to_underscores is None:
            return t
        return t.replace(map_string_to_underscores, "_")

    # must exist for RecordingSet
    recordings = load_kaldi_text_mapping(path / "wav.scp", must_exist=True)

    with ProcessPoolExecutor(num_jobs) as ex:
        dur_vals = ex.map(get_duration, recordings.values())
    durations = dict(zip(recordings.keys(), dur_vals))

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
    if segments.is_file():
        with segments.open() as f:
            supervision_segments = [sup_string.strip().split() for sup_string in f]

        texts = load_kaldi_text_mapping(path / "text")
        speakers = load_kaldi_text_mapping(path / "utt2spk")
        genders = load_kaldi_text_mapping(path / "spk2gender")
        languages = load_kaldi_text_mapping(path / "utt2lang")

        supervision_set = SupervisionSet.from_segments(
            SupervisionSegment(
                id=fix_id(segment_id),
                recording_id=recording_id,
                start=float(start),
                duration=add_durations(
                    float(end), -float(start), sampling_rate=sampling_rate
                ),
                channel=0,
                text=texts[segment_id],
                language=languages[segment_id],
                speaker=fix_id(speakers[segment_id]),
                gender=genders[speakers[segment_id]],
            )
            for segment_id, recording_id, start, end in supervision_segments
        )

    feature_set = None
    feats_scp = path / "feats.scp"
    if feats_scp.exists() and is_module_available("kaldi_native_io"):
        if frame_shift is not None:
            import kaldi_native_io
            from lhotse.features.io import KaldiReader

            feature_set = FeatureSet.from_features(
                Features(
                    type="kaldi_native_io",
                    num_frames=mat.shape[0],
                    num_features=mat.shape[1],
                    frame_shift=frame_shift,
                    sampling_rate=sampling_rate,
                    start=0,
                    duration=mat.shape[0] * frame_shift,
                    storage_type=KaldiReader.name,
                    storage_path=str(feats_scp),
                    storage_key=utt_id,
                    recording_id=supervision_set[utt_id].recording_id
                    if supervision_set is not None
                    else utt_id,
                    channels=0,
                )
                for utt_id, mat in kaldi_native_io.SequentialFloatMatrixReader(
                    f"scp:{feats_scp}"
                )
            )
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
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert all(len(r.sources) == 1 for r in recordings), (
        "Kaldi export of Recordings with multiple audio sources "
        "is currently not supported."
    )

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
                    source, sampling_rate=recording.sampling_rate
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

    else:
        # wav.scp
        save_kaldi_text_mapping(
            data={
                f"{recording.id}_{channel}": make_wavscp_channel_string_map(
                    source, sampling_rate=recording.sampling_rate
                )[channel]
                for recording in recordings
                for source in recording.sources
                for channel in source.channels
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
            data={
                f"{recording.id}_{channel}": recording.duration
                for recording in recordings
                for channel in recording.sources[0].channels
            },
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


def load_kaldi_text_mapping(
    path: Path, must_exist: bool = False
) -> Dict[str, Optional[str]]:
    """Load Kaldi files such as utt2spk, spk2gender, text, etc. as a dict."""
    mapping = defaultdict(lambda: None)
    if path.is_file():
        with path.open() as f:
            mapping = dict(line.strip().split(maxsplit=1) for line in f)
    elif must_exist:
        raise ValueError(f"No such file: {path}")
    return mapping


def save_kaldi_text_mapping(data: Dict[str, Any], path: Path):
    """Save flat dicts to Kaldi files such as utt2spk, spk2gender, text, etc."""
    with path.open("w") as f:
        for key, value in sorted(data.items()):
            print(key, value, file=f)


def make_wavscp_channel_string_map(
    source: AudioSource, sampling_rate: int
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
        if Path(source.source).suffix == ".wav" and len(source.channels) == 1:
            # Note: for single-channel waves, we don't need to invoke ffmpeg; but
            #       for multi-channel waves, Kaldi is going to complain.
            audios = dict()
            for channel in source.channels:
                audios[channel] = source.source
            return audios
        elif Path(source.source).suffix == ".sph":
            # we will do this specifically using the sph2pipe because
            # ffmpeg does not support shorten compression, which is sometimes
            # used in the sph files
            audios = dict()
            for channel in source.channels:
                audios[
                    channel
                ] = f"sph2pipe {source.source} -f wav -c {channel+1} -p | ffmpeg -threads 1 -i pipe:0 -ar {sampling_rate} -f wav -threads 1 pipe:1 |"

            return audios
        else:
            # Handles non-WAVE audio formats and multi-channel WAVEs.
            audios = dict()
            for channel in source.channels:
                audios[
                    channel
                ] = f"ffmpeg -threads 1 -i {source.source} -ar {sampling_rate} -map_channel 0.0.{channel}  -f wav -threads 1 pipe:1 |"
            return audios

    else:
        raise ValueError(f"Unknown AudioSource type: {source.type}")
