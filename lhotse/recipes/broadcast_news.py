"""
This is a part of English HUB4 corpus.
It contains Broadcast News data, i.e. audio and transcripts of TV news.
We currently support the following LDC packages:

1997 English Broadcast News Train (HUB4)
  Speech       LDC98S71
  Transcripts  LDC98T28
"""

import re
from itertools import chain
from pathlib import Path
from typing import Dict, Union, Optional, List

from bs4 import BeautifulSoup
from cytoolz import sliding_window
from sphfile import SPHFile

from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

# Since BroadcastNews SGML does not include </time> tags, BeautifulSoup hallucinates them
# in incorrect positions - it nests the <time> segments in each other, making parsing more difficult...
# We are using BS4 for parsing automatically up to <turn> level, and then write a custom parsing function.
EXCLUDE_BEGINNINGS = ['</time', '<overlap', '</overlap']


def prepare_broadcast_news(
        audio_dir: Pathlike,
        transcripts_dir: Pathlike,
        output_dir: Optional[Pathlike] = None
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for 1997 English Broadcast News corpus.
    We create three manifests: one with recordings, one with segments supervisions,
    and one with section supervisions. The latter can be used e.g. for topic segmentation.

    :param audio_dir: Path to ``LDC98S71`` package.
    :param transcripts_dir: Path to ``LDC98T28`` package.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :return: A dict with manifests. The keys are: ``{'recordings', 'sections', 'segments'}``.
    """
    audio_dir = Path(audio_dir)
    transcripts_dir = Path(transcripts_dir)

    audio_paths = sorted(audio_dir.rglob('*.sph'))
    sgml_paths = sorted(transcripts_dir.rglob('*.sgml'))

    recordings = RecordingSet.from_recordings(make_recording(p) for p in audio_paths)

    supervisions_list = [make_supervisions(p, r) for p, r in zip(sgml_paths, recordings)]
    section_supervisions = SupervisionSet.from_segments(
        chain.from_iterable(sups['sections'] for sups in supervisions_list)
    )
    segment_supervisions = SupervisionSet.from_segments(
        chain.from_iterable(sups['segments'] for sups in supervisions_list)
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_yaml(output_dir / 'recordings.yml')
        section_supervisions.to_yaml(output_dir / 'sections.yml')
        segment_supervisions.to_yaml(output_dir / 'segments.yml')

    return {
        'recordings': recordings,
        'sections': section_supervisions,
        'segments': segment_supervisions
    }


def make_recording(sph_path: Path) -> Recording:
    """Read a SPHERE file's header and create the corresponding ``Recording``."""
    sphf = SPHFile(sph_path)
    return Recording(
        id=sph_path.stem,
        sampling_rate=sphf.format['sample_rate'],
        num_samples=sphf.format['sample_count'],
        duration_seconds=sphf.format['sample_count'] / sphf.format['sample_rate'],
        sources=[
            AudioSource(
                type='file',
                channel_ids=[0],
                source='/'.join(sph_path.parts[-2:])
            )
        ]
    )


def try_parse(sgml_path: Path):
    """
    Return a BeautifulSoup object created from an SGML file.
    If it runs into Unicode decoding errors, it will try to determine the file's encoding
    and use iconv to automatically convert it to UTF-8.
    """
    try:
        return BeautifulSoup(sgml_path.read_text())
    except UnicodeDecodeError:
        import subprocess
        from tempfile import NamedTemporaryFile
        encoding = subprocess.check_output(f'file -bi {sgml_path}', shell=True, text=True).split(';')[-1].replace(
            'charset=', '').strip()
        with NamedTemporaryFile() as f:
            subprocess.run(f'iconv -f {encoding} -t utf-8 -o {f.name} {sgml_path}', shell=True, check=True, text=True)
            return BeautifulSoup(f.read())


def make_supervisions(sgml_path: Pathlike, recording: Recording) -> Dict[str, List[SupervisionSegment]]:
    """Create supervisions for sections and segments for a given HUB4 recording."""
    doc = try_parse(sgml_path)
    episode = doc.find('episode')
    section_supervisions = []
    text_supervisions = []
    text_idx = 0
    for sec_idx, section in enumerate(doc.find('episode').find_all('section')):
        sec_start = float(section.attrs['starttime'])
        section_supervisions.append(SupervisionSegment(
            id=f'{recording.id}_section{sec_idx:03d}',
            recording_id=recording.id,
            start=sec_start,
            duration=round(float(section.attrs['endtime']) - sec_start, ndigits=3),
            channel_id=0,
            language=episode.attrs['language'],
            custom={
                'section': section.attrs['type'],
                'program': episode.attrs['program']
            }
        ))
        for turn in section.find_all('turn'):
            # An example of the format in each turn:
            #
            # <turn speaker=Peter_Jennings spkrtype=male startTime=336.704 endTime=338.229>
            # <overlap startTime=336.704 endTime=337.575>
            # <time sec=336.704>
            #  time served up until
            # </overlap>
            # <time sec=337.575>
            #  this point?
            # </turn>
            for child in turn.children:
                # Here, we switch to custom parsing code as explained at the top of this script.
                lines = [l for l in str(child).split('\n') if
                         len(l) and not any(l.startswith(b) for b in EXCLUDE_BEGINNINGS)]
                if not lines:
                    continue
                times = []
                texts = []
                for time_marker, text in group_lines_in_time_marker(lines):
                    match = re.search(r'sec="?(\d+\.?\d*)"?', time_marker)
                    try:
                        times.append(float(match.group(1)))
                        texts.append(text)
                    except:
                        from pprint import pprint
                        print(time_marker)
                        print(text)
                        print(match)
                        print('===')
                        pprint(lines)
                        raise
                times.append(float(turn.attrs['endtime']))

                for (start, end), text in zip(sliding_window(2, times), texts):
                    text_supervisions.append(SupervisionSegment(
                        id=f'{recording.id}_segment{text_idx:04d}',
                        recording_id=recording.id,
                        start=start,
                        duration=round(end - start, ndigits=3),
                        channel_id=0,
                        language=episode.attrs['language'],
                        text=text.strip(),
                        speaker=turn.attrs['speaker'],
                        gender=turn.attrs['spkrtype']
                    ))
                text_idx += 1
    return {
        'sections': section_supervisions,
        'segments': text_supervisions
    }


def group_lines_in_time_marker(sgml_lines):
    """This is a helper for the situation when a <time> marker contains multiple lines of text."""
    from itertools import groupby
    # Top-level group allows pairwise iteration with step size of 2, e.g.
    # ['<time...', 'text', '<time...', 'text'] -> [('<time...', 'text'), ('<time...', 'text')]
    return group(
        [
            # Connect multi-lines with a whitespace
            ' '.join(l.strip() for l in lines)
            # groupby() will group the lines depending on whether they start with '<time' or not;
            # is_marker is a bool saying whether the group is a <time> marker,
            # and lines contains the actual lines in that group.
            for is_marker, lines
            in groupby(sgml_lines, key=lambda l: l.startswith('<time'))
        ],
        2
    )


def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]

    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.

    Source code by Brian Quinlan from:
    https://code.activestate.com/recipes/303060-group-a-list-into-sequential-n-tuples/

    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return zip(*[lst[i::n] for i in range(n)])
