from collections import defaultdict
from typing import Dict, Optional, Sequence, Union

from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available

if is_module_available('speechcolab'):
    from speechcolab.datasets.gigaspeech import GigaSpeech
else:
    raise ImportError('To process the GigaSpeech corpus, please install optional dependency: pip install speechcolab')


def prepare_gigaspeech(
        gigaspeech: GigaSpeech,
        dataset_parts: Union[str, Sequence[str]] = 'auto',
        output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    subsets = ('{XL}', '{DEV}', '{TEST}') if dataset_parts == 'auto' else dataset_parts

    manifests = defaultdict(dict)
    for subset in subsets:
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
        manifests[subset] = {
            'recordings': RecordingSet.from_recordings(recordings),
            'supervisions': SupervisionSet.from_segments(segments)
        }
    return dict(manifests)
