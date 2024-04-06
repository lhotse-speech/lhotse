"""
Description taken from the abstract of paper:
"GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio"
https://arxiv.org/abs/2106.06909

This paper introduces GigaSpeech, an evolving, multi-domain English speech recognition corpus with 10,000 hours of high quality labeled audio suitable for supervised training, and 40,000 hours of total audio suitable for semi-supervised and unsupervised training. Around 40,000 hours of transcribed audio is first collected from audiobooks, podcasts and YouTube, covering both read and spontaneous speaking styles, and a variety of topics, such as arts, science, sports, etc. A new forced alignment and segmentation pipeline is proposed to create sentence segments suitable for speech recognition training, and to filter out segments with low-quality transcription. For system training, GigaSpeech provides five subsets of different sizes, 10h, 250h, 1000h, 2500h, and 10000h. For our 10,000-hour XL training subset, we cap the word error rate at 4% during the filtering/validation stage, and for all our other smaller training subsets, we cap it at 0%. The DEV and TEST evaluation sets, on the other hand, are re-processed by professional human transcribers to ensure high transcription quality. Baseline systems are provided for popular speech recognition toolkits, namely Athena, ESPnet, Kaldi and Pika.
"""
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import (
    CutSet,
    compute_num_samples,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.parallel import parallel_map
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available

GIGASPEECH_PARTS = ("XL", "L", "M", "S", "XS", "DEV", "TEST")


def download_gigaspeech(
    password: str,
    target_dir: Pathlike = ".",
    dataset_parts: Optional[Union[str, Sequence[str]]] = "auto",
    host: Optional[str] = "tsinghua",
) -> Path:
    if is_module_available("speechcolab"):
        from speechcolab.datasets.gigaspeech import GigaSpeech
    else:
        raise ImportError(
            "To process the GigaSpeech corpus, please install optional dependency: pip install speechcolab"
        )
    gigaspeech = GigaSpeech(target_dir)

    if dataset_parts == "auto":
        dataset_parts = ("XL", "DEV", "TEST")
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    for part in dataset_parts:
        logging.info(f"Downloading GigaSpeech part: {part}")
        gigaspeech.download(password, "{" + part + "}", host=host)

    return target_dir


def prepare_gigaspeech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike],
    dataset_parts: Union[str, Sequence[str]] = "auto",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    if is_module_available("speechcolab"):
        from speechcolab.datasets.gigaspeech import GigaSpeech
    else:
        raise ImportError(
            "To process the GigaSpeech corpus, please install optional dependency: pip install speechcolab"
        )

    subsets = ("XL", "DEV", "TEST") if dataset_parts == "auto" else dataset_parts
    if isinstance(subsets, str):
        subsets = [subsets]
    corpus_dir = Path(corpus_dir)
    gigaspeech = GigaSpeech(corpus_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Maybe some manifests already exist: we can read them and save a bit of preparation time.
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=output_dir,
        prefix="gigaspeech",
        suffix="jsonl.gz",
        lazy=True,
    )

    for part in subsets:
        logging.info(f"Processing GigaSpeech subset: {part}")
        if manifests_exist(
            part=part, output_dir=output_dir, prefix="gigaspeech", suffix="jsonl.gz"
        ):
            logging.info(f"GigaSpeech subset: {part} already prepared - skipping.")
            continue

        with RecordingSet.open_writer(
            output_dir / f"gigaspeech_recordings_{part}.jsonl.gz"
        ) as rec_writer, SupervisionSet.open_writer(
            output_dir / f"gigaspeech_supervisions_{part}.jsonl.gz"
        ) as sup_writer, CutSet.open_writer(
            output_dir / f"gigaspeech_cuts_{part}.jsonl.gz"
        ) as cut_writer:
            for recording, segments in tqdm(
                parallel_map(
                    parse_utterance,
                    gigaspeech.audios("{" + part + "}"),
                    repeat(gigaspeech.gigaspeech_dataset_dir),
                    num_jobs=num_jobs,
                ),
                desc="Processing GigaSpeech JSON entries",
            ):
                # Fix and validate the recording + supervisions
                recordings, segments = fix_manifests(
                    recordings=RecordingSet.from_recordings([recording]),
                    supervisions=SupervisionSet.from_segments(segments),
                )
                validate_recordings_and_supervisions(
                    recordings=recordings, supervisions=segments
                )
                # Create the cut since most users will need it anyway.
                # There will be exactly one cut since there's exactly one recording.
                cuts = CutSet.from_manifests(
                    recordings=recordings, supervisions=segments
                )
                # Write the manifests
                rec_writer.write(recordings[0])
                for s in segments:
                    sup_writer.write(s)
                cut_writer.write(cuts[0])

        manifests[part] = {
            "recordings": RecordingSet.from_jsonl_lazy(rec_writer.path),
            "supervisions": SupervisionSet.from_jsonl_lazy(sup_writer.path),
            "cuts": CutSet.from_jsonl_lazy(cut_writer.path),
        }

    return dict(manifests)


def parse_utterance(
    audio: Any, root_path: Path
) -> Optional[Tuple[Recording, List[SupervisionSegment]]]:
    sampling_rate = int(audio["sample_rate"])
    recording = Recording(
        id=audio["aid"],
        sources=[
            AudioSource(
                type="file",
                channels=list(range(int(audio["channels"]))),
                source=str(root_path / audio["path"]),
            )
        ],
        num_samples=compute_num_samples(
            duration=Seconds(audio["duration"]), sampling_rate=sampling_rate
        ),
        sampling_rate=sampling_rate,
        duration=Seconds(audio["duration"]),
    )
    segments = []
    for seg in audio["segments"]:
        segments.append(
            SupervisionSegment(
                id=seg["sid"],
                recording_id=audio["aid"],
                start=Seconds(seg["begin_time"]),
                duration=round(Seconds(seg["end_time"] - seg["begin_time"]), ndigits=8),
                channel=0,
                language="English",
                speaker=seg["speaker"],
                text=seg["text_tn"],
            )
        )
    return recording, segments
