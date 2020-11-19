from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lhotse import CutSet
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def load_kaldi_data_dir(path: Pathlike, sampling_rate: int) -> Tuple[RecordingSet, Optional[SupervisionSet]]:
    """
    Load a Kaldi data directory and convert it to a Lhotse RecordingSet and SupervisionSet manifests.
    For this to work, at least the wav.scp file must exist.
    SupervisionSet is created only when a segments file exists.
    All the other files (text, utt2spk, etc.) are optional, and some of them might not be handled yet.
    In particular, feats.scp files are ignored.
    """
    path = Path(path)
    assert path.is_dir()

    # must exist for RecordingSet
    recordings = load_kaldi_text_mapping(path / 'wav.scp', must_exist=True)

    durations = defaultdict(float)
    reco2dur = path / 'reco2dur'
    if not reco2dur.is_file():
        raise ValueError(f"No such file: '{reco2dur}' -- fix it by running: utils/data/get_reco2dur.sh <data-dir>")
    with reco2dur.open() as f:
        for line in f:
            recording_id, dur = line.strip().split()
            durations[recording_id] = float(dur)

    audio_set = RecordingSet.from_recordings(
        Recording(
            id=recording_id,
            sources=[
                AudioSource(
                    type='command' if path_or_cmd.endswith('|') else 'file',
                    channels=[0],
                    source=path_or_cmd[:-1] if path_or_cmd.endswith('|') else path_or_cmd
                )
            ],
            sampling_rate=sampling_rate,
            num_samples=int(durations[recording_id] * sampling_rate),
            duration=durations[recording_id]
        )
        for recording_id, path_or_cmd in recordings.items()
    )

    # must exist for SupervisionSet
    segments = path / 'segments'
    if not segments.is_file():
        return audio_set, None

    with segments.open() as f:
        supervision_segments = [l.strip().split() for l in f]

    texts = load_kaldi_text_mapping(path / 'text')
    speakers = load_kaldi_text_mapping(path / 'utt2spk')
    genders = load_kaldi_text_mapping(path / 'spk2gender')
    languages = load_kaldi_text_mapping(path / 'utt2lang')

    supervision_set = SupervisionSet.from_segments(
        SupervisionSegment(
            id=segment_id,
            recording_id=recording_id,
            start=float(start),
            duration=float(end) - float(start),
            channel=0,
            text=texts[segment_id],
            language=languages[segment_id],
            speaker=speakers[segment_id],
            gender=genders[speakers[segment_id]]
        )
        for segment_id, recording_id, start, end in supervision_segments
    )

    return audio_set, supervision_set


def export_to_kaldi(recordings: RecordingSet, supervisions: SupervisionSet, output_dir: Pathlike):
    """
    Export a pair of ``RecordingSet`` and ``SupervisionSet`` to a Kaldi data directory.
    Currently, it only supports single-channel recordings that have a single ``AudioSource``.

    The ``RecordingSet`` and ``SupervisionSet`` must be compatible, i.e. it must be possible to create a
    ``CutSet`` out of them.

    :param recordings: a ``RecordingSet`` manifest.
    :param supervisions: a ``SupervisionSet`` manifest.
    :param output_dir: path where the Kaldi-style data directory will be created.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert all(len(r.sources) == 1 for r in recordings), "Kaldi export of Recordings with multiple audio sources " \
                                                         "is currently not supported."
    assert all(r.num_channels == 1 for r in recordings), "Kaldi export of multi-channel Recordings is currently " \
                                                         "not supported."

    # Create a simple CutSet that ties together the recording <-> supervision information.
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions).trim_to_supervisions()

    # wav.scp
    save_kaldi_text_mapping(
        data={
            recording.id: f'{source.source} |' if source.type == 'command' else source.source
            for recording in recordings
            for src_idx, source in enumerate(recording.sources)
        },
        path=output_dir / 'wav.scp'
    )
    # segments
    save_kaldi_text_mapping(
        data={cut.supervisions[0].id: f'{cut.recording_id} {cut.start} {cut.end}' for cut in cuts},
        path=output_dir / 'segments'
    )
    # text
    save_kaldi_text_mapping(
        data={cut.supervisions[0].id: cut.supervisions[0].text for cut in cuts},
        path=output_dir / 'text'
    )
    # utt2spk
    save_kaldi_text_mapping(
        data={cut.supervisions[0].id: cut.supervisions[0].speaker for cut in cuts},
        path=output_dir / 'utt2spk'
    )
    # utt2dur
    save_kaldi_text_mapping(
        data={cut.supervisions[0].id: cut.duration for cut in cuts},
        path=output_dir / 'utt2dur'
    )
    # reco2dur
    save_kaldi_text_mapping(
        data={recording.id: recording.duration for recording in recordings},
        path=output_dir / 'reco2dur'
    )
    # utt2lang [optional]
    if all(s.language is not None for s in supervisions):
        save_kaldi_text_mapping(
            data={cut.supervisions[0].id: cut.supervisions[0].language for cut in cuts},
            path=output_dir / 'utt2lang'
        )
    # utt2gender [optional]
    if all(s.gender is not None for s in supervisions):
        save_kaldi_text_mapping(
            data={cut.supervisions[0].id: cut.supervisions[0].gender for cut in cuts},
            path=output_dir / 'utt2gender'
        )


def load_kaldi_text_mapping(path: Path, must_exist: bool = False) -> Dict[str, Optional[str]]:
    """Load Kaldi files such as utt2spk, spk2gender, text, etc. as a dict."""
    mapping = defaultdict(lambda: None)
    if path.is_file():
        with path.open() as f:
            mapping = dict(line.strip().split(' ', maxsplit=1) for line in f)
    elif must_exist:
        raise ValueError(f"No such file: {path}")
    return mapping


def save_kaldi_text_mapping(data: Dict[str, Any], path: Path):
    """Save flat dicts to Kaldi files such as utt2spk, spk2gender, text, etc."""
    with path.open('w') as f:
        for key, value in data.items():
            print(key, value, file=f)
