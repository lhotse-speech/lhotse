from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional

from lhotse.audio import AudioSet, Recording, AudioSource
from lhotse.supervision import SupervisionSet, SupervisionSegment
from lhotse.utils import Pathlike


def load_kaldi_data_dir(path: Pathlike, sampling_rate: int) -> Tuple[AudioSet, Optional[SupervisionSet]]:
    path = Path(path)
    assert path.is_dir()

    # must exist for AudioSet
    wav_scp = path / 'wav.scp'
    assert wav_scp.is_file()
    with wav_scp.open() as f:
        recordings = dict(line.strip().split(' ', maxsplit=1) for line in f)

    durations = defaultdict(float)
    reco2dur = path / 'reco2dur'
    if reco2dur.is_file():
        with reco2dur.open() as f:
            for line in f:
                recording_id, dur = line.strip().split()
                durations[recording_id] = float(dur)

    audio_set = AudioSet({
        recording_id: Recording(
            id=recording_id,
            sources=[
                AudioSource(
                    type='command' if path_or_cmd.endswith('|') else 'file',
                    channel_ids=[0],
                    source=path_or_cmd[:-1] if path_or_cmd.endswith('|') else path_or_cmd
                )
            ],
            sampling_rate=sampling_rate,
            num_samples=int(durations[recording_id] * sampling_rate),
            duration_seconds=durations[recording_id]
        )
        for recording_id, path_or_cmd in recordings.items()
    })

    # must exist for SupervisionSet
    segments = path / 'segments'
    if not segments.is_file():
        return audio_set, None

    with segments.open() as f:
        supervision_segments = [l.strip().split() for l in f]

    texts = defaultdict(lambda: None)
    text_path = path / 'text'
    if text_path.is_file():
        with text_path.open() as f:
            texts = dict(line.strip().split(' ', maxsplit=1) for line in f)

    speakers = defaultdict(lambda: None)
    utt2spk = path / 'utt2spk'
    if utt2spk.is_file():
        with utt2spk.open() as f:
            speakers = dict(line.strip().split(' ', maxsplit=1) for line in f)

    # TODO: spk2gender

    supervision_set = SupervisionSet({
        segment_id: SupervisionSegment(
            id=segment_id,
            recording_id=recording_id,
            start=float(start),
            duration=float(duration),  # TODO: check again what's the second time field in segments
            text=texts[segment_id],
            language=None,
            speaker=speakers[segment_id]
        )
        for segment_id, recording_id, start, duration in supervision_segments
    })

    return audio_set, supervision_set