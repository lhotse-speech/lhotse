from typing import Dict, Union

from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import is_module_available

if not is_module_available('speechcolab'):
    raise ImportError('To process the GigaSpeech corpus, please install optional dependency: pip install '
                      'speechcolab')
else:
    from speechcolab.datasets.gigaspeech import GigaSpeech


def prepare_gigaspeech(
        gigaspeech: GigaSpeech,
        subset: str
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    recordings = []
    segments = []
    for audio in gigaspeech.audios(subset):
        recordings.append(Recording.from_file(gigaspeech.root_path / audio['path'],
                                              recording_id=audio['aid']))
        for seg in audio['segments']:
            segments.append(SupervisionSegment(id=seg['sid'],
                                               recording_id=audio['aid'],
                                               start=seg['begin_time'],
                                               duration=seg['end_time'] - seg['begin_time'],
                                               channel=0,
                                               language='English',
                                               speaker=seg['speaker'],
                                               text=seg['text_tn']))

    manifests = {
        'recordings': RecordingSet.from_recordings(recordings),
        'supervisions': SupervisionSet.from_segments(segments)
    }
    return dict(manifests)
