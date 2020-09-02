"""
About the Switchboard corpus

    This is conversational telephone speech collected as 2-channel, 8kHz-sampled
    data.  We are using just the Switchboard-1 Phase 1 training data.
    The catalog number LDC97S62 (Switchboard-1 Release 2) corresponds, we believe,
    to what we have.  We also use the Mississippi State transcriptions, which
    we download separately from
    http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
"""
import tarfile
import urllib
from itertools import chain
from pathlib import Path
from typing import Dict, Union, Optional

from lhotse.audio import RecordingSet, Recording
from lhotse.supervision import SupervisionSet, SupervisionSegment
from lhotse.utils import Pathlike


SWBD_TEXT_URL = 'http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz'


def prepare_switchboard(
        audio_dir: Pathlike,
        transcripts_dir: Optional[Pathlike] = None,
        output_dir: Optional[Pathlike] = None,
        omit_silence: bool = True
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the Switchboard corpus.
    We create two manifests: one with recordings, and the other one with text supervisions.

    :param audio_dir: Path to ``LDC97S62`` package.
    :param transcripts_dir: Path to the transcripts directory (typically named "swb_ms98_transcriptions").
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    if transcripts_dir is None:
        transcripts_dir = download_and_untar()
    audio_paths = sorted(Path(audio_dir).rglob('*.sph'))
    text_paths = sorted(Path(transcripts_dir).rglob('*trans.text'))

    groups = []
    name_to_text = {p.stem.split('-')[0]: p for p in text_paths}
    for ap in audio_paths:
        name = ap.stem.replace('sw0', 'sw')
        groups.append({'audio': ap, 'text-0': name_to_text[f'{name}A'], 'text-1': name_to_text[f'{name}B']})

    recordings = RecordingSet.from_recordings(
        Recording.from_sphere(group['audio'], relative_path_depth=3) for group in groups
    )
    supervisions = SupervisionSet.from_segments(chain.from_iterable(
        make_segments(
            transcript_path=group[f'text-{channel}'],
            recording=recording,
            channel=channel,
            omit_silence=omit_silence
        )
        for group, recording in zip(groups, recordings)
        for channel in [0, 1]
    ))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_yaml(output_dir / 'recordings.yml')
        supervisions.to_yaml(output_dir / 'supervisions.yml')
    return {
        'recordings': recordings,
        'supervisions': supervisions
    }


def make_segments(transcript_path: Path, recording: Recording, channel: int, omit_silence: bool = True):
    lines = transcript_path.read_text().splitlines()
    return [
        SupervisionSegment(
            id=segment_id,
            recording_id=recording.id,
            start=float(start),
            duration=round(float(end) - float(start), ndigits=3),
            channel_id=channel,
            text=' '.join(words),
            language='English',
            speaker=f'{recording.id}A'
        )
        for segment_id, start, end, *words in map(str.split, lines)
        if words[0] != '[silence]' or not omit_silence
    ]


def download_and_untar(
        target_dir: Pathlike = '.',
        force_download: bool = False,
        url: str = SWBD_TEXT_URL
) -> Path:
    target_dir = Path(target_dir)
    transcript_dir = target_dir / 'swb_ms98_transcriptions'
    if transcript_dir.is_dir():
        return transcript_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_name = 'switchboard_word_alignments.tar.gz'
    tar_path = target_dir / tar_name
    if force_download or not tar_path.is_file():
        urllib.request.urlretrieve(url, filename=tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    return transcript_dir
