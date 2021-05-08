from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available


def prepare_gigaspeech(
        gigaspeech: Any,
        dataset_parts: Union[str, Sequence[str]] = 'auto',
        output_dir: Optional[Pathlike] = None,
        num_jobs: int = 1
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    if is_module_available('speechcolab'):
        from speechcolab.datasets.gigaspeech import GigaSpeech
    else:
        raise ImportError(
            'To process the GigaSpeech corpus, please install optional dependency: pip install speechcolab')

    subsets = ('{XL}', '{DEV}', '{TEST}') if dataset_parts == 'auto' else dataset_parts

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        maybe_manifests = read_manifests_if_cached(dataset_parts=dataset_parts, output_dir=output_dir)
        if maybe_manifests is not None:
            return maybe_manifests

    manifests = defaultdict(dict)
    with ThreadPoolExecutor(num_jobs) as ex:
        for part in subsets:
            futures = []
            for audio in tqdm(gigaspeech.audios(part), desc='Distributing tasks', leave=False):
                futures.append(ex.submit(parse_utterance, audio, gigaspeech.root_path))

            recordings = []
            supervisions = []
            for future in tqdm(futures, desc='Processing', leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segments = result
                recordings.append(recording)
                supervisions += segments

            manifests[part] = {
                'recordings': RecordingSet.from_recordings(recordings),
                'supervisions': SupervisionSet.from_segments(supervisions)
            }

            if output_dir is not None:
                manifests[part]['recordings'].to_json(output_dir / f'recordings_{part}.json')
                manifests[part]['supervisions'].to_json(output_dir / f'supervisions_{part}.json')

    return dict(manifests)


def parse_utterance(
        audio: Any,
        root_path: Path
) -> Optional[Tuple[Recording, List[SupervisionSegment]]]:
    recording = Recording.from_file(root_path / audio['path'], recording_id=audio['aid'])
    segments = []
    for seg in audio['segments']:
        segments.append(SupervisionSegment(id=seg['sid'],
                                           recording_id=audio['aid'],
                                           start=Seconds(seg['begin_time']),
                                           duration=Seconds(seg['end_time'] - seg['begin_time']),
                                           channel=0,
                                           language='English',
                                           speaker=seg['speaker'],
                                           text=seg['text_tn']))
    return recording, segments
