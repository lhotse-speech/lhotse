"""
The People’s Speech Dataset is among the world’s largest English speech recognition corpus today 
that is licensed for academic and commercial usage under CC-BY-SA and CC-BY 4.0. 
It includes 30,000+ hours of transcribed speech in English languages with a diverse set of speakers. 
This open dataset is large enough to train speech-to-text systems and crucially is available with 
a permissive license. 
Just as ImageNet catalyzed machine learning for vision, the People’s Speech will unleash innovation
in speech research and products that are available to users across the globe.

Source: https://mlcommons.org/en/peoples-speech/
Full paper: https://openreview.net/pdf?id=R8CwidgJ0yT
"""

from pathlib import Path
from typing import Dict, Union
import warnings

from tqdm.auto import tqdm
from lhotse.qa import validate_recordings_and_supervisions

from lhotse.utils import Pathlike, compute_num_samples
from lhotse.audio import Recording, RecordingSet, AudioSource, info
from lhotse.serialization import load_jsonl
from lhotse.supervision import SupervisionSet, SupervisionSegment


def prepare_peoples_speech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare :class:`~lhotse.RecordingSet` and :class:`~lhotse.SupervisionSet` manifests
    for The People's Speech.

    The metadata is read lazily and written to manifests in a stream to minimize
    the CPU RAM usage. If you want to convert this data to a :class:`~lhotse.CutSet`
    without using excessive memory, we suggest to call it like::

        >>> peoples_speech = prepare_peoples_speech(corpus_dir=..., output_dir=...)
        >>> cuts = CutSet.from_manifests(
        ...     recordings=peoples_speech["recordings"],
        ...     supervisions=peoples_speech["supervisions"],
        ...     output_path=...,
        ...     lazy=True,
        ... )

    :param corpus_dir: Pathlike, the path of the main data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a dict with keys "recordings" and "supervisions" with lazily opened manifests.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recs_path = output_dir / "peoples-speech_recordings.jsonl.gz"
    sups_path = output_dir / "peoples-speech_supervisions.jsonl.gz"

    if recs_path.is_file() and sups_path.is_file():
        # Nothing to do: just open the manifests in lazy mode.
        return {
            "recordings": RecordingSet.from_jsonl_lazy(recs_path),
            "supervisions": SupervisionSet.from_jsonl_lazy(sups_path),
        }

    exist = 0
    tot = 0
    err = 0
    with RecordingSet.open_writer(recs_path,) as rec_writer, SupervisionSet.open_writer(
        sups_path,
    ) as sup_writer:
        for item in tqdm(
            # Note: People's Speech manifest.json is really a JSONL.
            load_jsonl(corpus_dir / "manifest.json"),
            desc="Converting People's Speech manifest.json to Lhotse manifests",
        ):
            for duration_ms, text, audio_path in zip(*item["training_data"].values()):
                full_path = corpus_dir / audio_path

                tot += 1
                if not full_path.exists():
                    # If we can't find some data, we'll just continue and some items
                    # were missing later.
                    continue
                exist += 1

                try:
                    audio_info = info(full_path)
                    duration = duration_ms / 1000
                    r = Recording(
                        id=full_path.stem,
                        sampling_rate=audio_info.samplerate,
                        num_samples=compute_num_samples(
                            duration, audio_info.samplerate
                        ),
                        duration=duration,
                        sources=[
                            AudioSource(
                                type="file",
                                channels=[0],
                                source=str(full_path),
                            )
                        ],
                    )
                    s = SupervisionSegment(
                        id=r.id,
                        recording_id=r.id,
                        start=0,
                        duration=r.duration,
                        channel=0,
                        text=text,
                        language="English",
                        custom={"session_id": item["identifier"]},
                    )

                    validate_recordings_and_supervisions(recordings=r, supervisions=s)

                    rec_writer.write(r)
                    sup_writer.write(s)

                except Exception as e:
                    # If some files are missing (e.g. somebody is working on a subset
                    # of 30.000 hours), we won't interrupt processing; we will only
                    # do so for violated assertions.
                    if isinstance(e, AssertionError):
                        raise
                    err += 1
                    continue

    if exist < tot or err > 0:
        warnings.warn(
            f"We finished preparing The People's Speech Lhotse manifests. "
            f"Out of {tot} entries in the original manifest, we found {exist} "
            f"audio files existed, out of which {err} had errors during processing."
        )

    return {
        "recordings": rec_writer.open_manifest(),
        "supervisions": sup_writer.open_manifest(),
    }
