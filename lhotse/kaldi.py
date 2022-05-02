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
import contextlib


class KaldiFormatterBase:
    def __init__(self, options: dict):
        self.options = options

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        return True


class DefaultOutputKaldiFormatter(KaldiFormatterBase):
    def __init__(
        self, recordings: RecordingSet, supervisions: SupervisionSet, options: dict
    ):
        super().__init__(options)
        self.recordings = recordings
        self.supervisions = supervisions

    def _maybe_remap_supervision_names(
        self, supervisions: SupervisionSet
    ) -> SupervisionSet:
        if self.options.get("map_underscores_to") is not None:
            supervisions = supervisions.map(
                lambda s: fastcopy(
                    s,
                    id=s.id.replace("_", self.options["map_underscores_to"]),
                    speaker=s.speaker.replace("_", self.options["map_underscores_to"]),
                )
            )

        if self.options.get("prefix_spk_id"):
            supervisions = supervisions.map(
                lambda s: fastcopy(s, id=f"{s.speaker}-{s.id}")
            )
        return supervisions

    def has_utt2gender(self):
        return all(s.gender is not None for s in self.supervisions)

    def utt2gender(self):
        supervisions = self._maybe_remap_supervision_names(self.supervisions)
        if all(s.gender is not None for s in supervisions):
            for s in supervisions:
                yield s.id, s.gender
        else:
            return

    def has_utt2lang(self):
        return all(s.language is not None for s in self.supervisions)

    def utt2lang(self):
        supervisions = self._maybe_remap_supervision_names(self.supervisions)
        if all(s.language is not None for s in supervisions):
            for s in supervisions:
                yield s.id, s.language
        else:
            return

    def utt2dur(self):
        supervisions = self._maybe_remap_supervision_names(self.supervisions)
        for s in supervisions:
            yield s.id, s.duration

    def utt2spk(self):
        supervisions = self._maybe_remap_supervision_names(self.supervisions)
        for s in supervisions:
            yield s.id, s.speaker

    def text(self):
        supervisions = self._maybe_remap_supervision_names(self.supervisions)
        for s in supervisions:
            yield s.id, s.text

    def reco2dur(self):
        recordings = self.recordings
        if all(r.num_channels == 1 for r in recordings):
            for r in recordings:
                yield r.id, r.duration
        else:
            for r in recordings:
                for channel in r.sources[0].channels:
                    yield f"{r.id}_{channel}", r.duration

    def segments(self):
        supervisions = self._maybe_remap_supervision_names(self.supervisions)
        for s in supervisions:
            yield s.id, s.recording_id, s.start, s.end

    def wav_scp(self):
        recordings = self.recordings
        if all(r.num_channels == 1 for r in recordings):
            for r in recordings:
                for source in r.sources:
                    yield r.id, make_wavscp_channel_string_map(source, r.sampling_rate)[
                        0
                    ]
        else:
            for r in recordings:
                for source in r.sources:
                    for channel in source.channels:
                        yield f"{r.id}_{channel}", make_wavscp_channel_string_map(
                            source, r.sampling_rate
                        )[channel]


class DefaultInputKaldiFormatter(KaldiFormatterBase):
    def __init__(self, kaldi_data_dir, options: dict):
        super().__init__(options)
        self.kaldi_data_dir = kaldi_data_dir

    @classmethod
    def durations(cls, recordings, num_jobs):
        with ProcessPoolExecutor(num_jobs) as ex:
            dur_vals = ex.map(get_duration, recordings.values())
        durations = dict(zip(recordings.keys(), dur_vals))
        return durations

    def _fix_segment_id(self, segment_id: str) -> str:
        if self.options.get("map_string_to_underscores", None):
            return segment_id.replace(self.options["map_string_to_underscores"], "_")
        else:
            return segment_id

    def recordings(self):
        self.recs = self.kaldi_data_dir.get_map("wav.scp")
        num_jobs = self.options.get("num_jobs", 1)
        self.durs = self.durations(self.recs, num_jobs)

        sampling_rate = self.options["sampling_rate"]

        for id, path_or_cmd in self.recs.items():
            duration = self.durs[id]
            yield Recording(
                id=id,
                sources=[
                    AudioSource(
                        type="command" if path_or_cmd.endswith("|") else "file",
                        channels=[0],
                        source=(path_or_cmd[:-1])
                        if path_or_cmd.endswith("|")
                        else path_or_cmd,
                    )
                ],
                sampling_rate=sampling_rate,
                num_samples=compute_num_samples(duration, sampling_rate),
                duration=duration,
            )

    def segments(self):
        if self.kaldi_data_dir.has("text") and not self.kaldi_data_dir.has("segments"):
            # this situation assumes the files are indexed by utterance id
            recs = self.kaldi_data_dir.get_map("wav.scp")
            num_jobs = self.options.get("num_jobs", 1)
            durations = self.durations(recs, num_jobs)
            segments = {id: (id, 0, durations[id]) for id in self.recs.keys()}
        elif self.kaldi_data_dir.has("text"):
            segments = self.kaldi_data_dir.get_map("segments")
            for id in segments.keys():
                entries = segments[id].split()
                assert len(entries) == 3
                segments[id] = (entries[0], float(entries[1]), float(entries[2]))
        else:
            # assert self.kaldi_data_dir.has("text")
            return

        sampling_rate = self.options["sampling_rate"]
        texts = self.kaldi_data_dir.get_map("text")
        utt2spk = self.kaldi_data_dir.get_map("utt2spk")
        spk2gender = self.kaldi_data_dir.get_map("spk2gender")
        utt2lang = self.kaldi_data_dir.get_map("utt2lang")
        for seg_id, (recording_id, start, end) in segments.items():
            yield SupervisionSegment(
                id=self._fix_segment_id(seg_id),
                recording_id=recording_id,
                start=start,
                duration=add_durations(end, -start, sampling_rate=sampling_rate),
                channel=0,
                text=texts[seg_id],
                language=utt2lang[seg_id],
                speaker=self._fix_segment_id(utt2spk[seg_id]),
                gender=spk2gender[utt2spk[seg_id]],
            )
        pass


@contextlib.contextmanager
def formatter(*argv) -> KaldiFormatterBase:
    if len(argv) == 4:
        format = argv[0]
        recordings = argv[1]
        supervisions = argv[2]
        options = argv[3]
        yield DefaultOutputKaldiFormatter(recordings, supervisions, options)
    elif len(argv) == 3:
        format = argv[0]
        options = argv[2]
        kaldi_data_dir = argv[1]
        yield DefaultInputKaldiFormatter(kaldi_data_dir, options)
    else:
        assert False


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

        wave = kaldi_native_io.read_wave(path)
        assert wave.data.shape[0] == 1, f"Expect 1 channel. Given {wave.data.shape[0]}"

        return wave.duration
    try:
        # Try to parse the file using pysoundfile first.
        import soundfile

        info = soundfile.info(path)
    except Exception:
        # Try to parse the file using audioread as a fallback.
        info = audioread_info(path)
    return info.duration


class KaldiDirectory:
    def __init__(self, path: Pathlike, options: dict):
        self.path = Path(path)
        self.options = options

    def has(self, filename: Pathlike):
        return (self.path / filename).is_file()

    def get_map(self, map):
        return load_kaldi_text_mapping(self.path / map)

    def get_value(self, map):
        with open(self.path / map) as f:
            return f.readline().strip()


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

    options = {
        "map_string_to_underscores": map_string_to_underscores,
        "frame_shift": frame_shift,
        "sampling_rate": sampling_rate,
        "num_jobs": num_jobs,
    }

    kaldi_data = KaldiDirectory(path, options)
    format = "default"
    recording_set = None
    supervision_set = None
    feature_set = None
    with formatter(format, kaldi_data, options) as fmt:
        recording_set = RecordingSet.from_recordings(
            recording for recording in fmt.recordings()
        )

        if kaldi_data.has("text") or kaldi_data.has("segments"):
            supervision_set = SupervisionSet.from_segments(
                segment for segment in fmt.segments()
            )

        if kaldi_data.has("feats.scp") and is_module_available("kaldi_native_io"):
            # feature_set = FeatureSet.from_features(
            pass

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

    options = {"map_underscores_to": map_underscores_to, "prefix_spk_id": prefix_spk_id}

    format = "default"
    with formatter(format, recordings, supervisions, options) as fmt:

        save_kaldi_text_mapping(
            data={id: audio_spec for id, audio_spec in fmt.wav_scp()},
            path=output_dir / "wav.scp",
        )

        save_kaldi_text_mapping(
            data={
                id: f"{recording_id} {start} {end}"
                for id, recording_id, start, end in fmt.segments()
            },
            path=output_dir / "segments",
        )

        # reco2dur
        save_kaldi_text_mapping(
            data={id: duration for id, duration in fmt.reco2dur()},
            path=output_dir / "reco2dur",
        )

        # text
        save_kaldi_text_mapping(
            data={id: text for id, text in fmt.text()},
            path=output_dir / "text",
        )

        # utt2spk
        save_kaldi_text_mapping(
            data={id: speaker for id, speaker in fmt.utt2spk()},
            path=output_dir / "utt2spk",
        )

        # utt2dur
        save_kaldi_text_mapping(
            data={id: duration for id, duration in fmt.utt2dur()},
            path=output_dir / "utt2dur",
        )

        # utt2lang [optional]
        if fmt.has_utt2lang:
            save_kaldi_text_mapping(
                data={id: language for id, language in fmt.utt2lang()},
                path=output_dir / "utt2lang",
            )

        # utt2gender [optional]
        if fmt.has_utt2gender:
            save_kaldi_text_mapping(
                data={id: gender for id, gender in fmt.utt2gender()},
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
