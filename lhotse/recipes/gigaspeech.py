from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import AudioSource, Recording, RecordingSet
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
        maybe_manifests = read_manifests_if_cached(dataset_parts=dataset_parts, output_dir=output_dir, suffix='jsonl')
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
                manifests[part]['recordings'].to_file(output_dir / f'recordings_{part}.jsonl')
                manifests[part]['supervisions'].to_file(output_dir / f'supervisions_{part}.jsonl')

    return dict(manifests)


def parse_utterance(
        audio: Any,
        root_path: Path
) -> Optional[Tuple[Recording, List[SupervisionSegment]]]:
    # Opus-format audio would be decoded at 48kHz by force, with the original sampling rate being ignored.
    opus_decoding_sample_rate = 48000

    recording = Recording(id=audio['aid'],
                          sources=[AudioSource(type='file',
                                               channels=list(range(int(audio['channels']))),
                                               source=f'{root_path}/{audio["path"]}')],
                          num_samples=round(opus_decoding_sample_rate * Seconds(audio['duration']), ndigits=8),
                          sampling_rate=opus_decoding_sample_rate,
                          duration=Seconds(audio['duration'])).resample(int(audio['sample_rate']))
    segments = []
    for seg in audio['segments']:
        segments.append(SupervisionSegment(id=seg['sid'],
                                           recording_id=audio['aid'],
                                           start=Seconds(seg['begin_time']),
                                           duration=round(Seconds(seg['end_time'] - seg['begin_time']), ndigits=8),
                                           channel=0,
                                           language='English',
                                           speaker=seg['speaker'],
                                           text=seg['text_tn']))
    return recording, segments
