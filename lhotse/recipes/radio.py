"""
This recipe prepares data collected from radio streamed on the web. The data
have some metadata attached to them, including the geographic location of
broadcast, date and time of the recorded clip, as well as a unique station
identifier.

Obtaining the data
-----------------------------------------------------------
If you want to use this corpus please email: wiesner@jhu.edu

As the data are collected from radio stream, they cannot be broadly
disseminated or used for commercial purposes. In the email, include your
affiliated academic institution and the intended use for the data and we will
the data to you if it is indeed for non-commercial, academic purporses.

Description
------------------------------------------------------------
The data consist of ∼4000 hours of speech collected between
September 27, 2023 to October 1, 2023, in 9449 locations all over the world,
from 17171 stations. 

These data were used for Geolocation of speech in order to answer the question,
Where are you from? in the paper 

Where are you from? Geolocating Speech and Applications to Language
Identification, presented at NAACL 2024. Please read for a full descrption
and please cite as 
 
@inproceedings{foley2024you,
  title={Where are you from? Geolocating Speech and Applications to Language Identification},
  author={Foley, Patrick and Wiesner, Matthew and Odoom, Bismarck and Perera, Leibny Paola Garcia and Murray, Kenton and Koehn, Philipp},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={5114--5126},
  year={2024}
}
"""
import json
import re
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.parallel import parallel_map
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def _make_reco_and_sups_from_file(sf: str, msd: float = 0.5):
    corpus_dir = sf.parents[2]
    audio_dir = corpus_dir / "recos"
    fname = sf.with_suffix(".flac").stem

    # E.g. 2023_10_01_09h_02m_54s_dur30_ZnpbY9Zx_lat3.17_long113.04
    chunk_idx = int(sf.parent.suffix.strip("."))
    reco_file = audio_dir / f"recos.{chunk_idx}" / f"{fname}.flac"
    reco = Recording.from_file(reco_file, recording_id=fname)
    reco.channel_ids = [0]
    sups = []
    total = 0
    with open(sf) as f:
        segments = json.load(f)

    # Parse the file format, shown in the comment above, to get:
    # date, station, latitude, longitude, and the estimated gender
    lat, lon = re.search(r"lat[^_]+_long[^_]+", Path(sf).stem).group(0).split("_")
    lat = float(lat.replace("lat", ""))
    lon = float(lon.replace("long", ""))
    station = re.search(r"s_dur[0-9]+_(.*)_lat[^_]+_long[^_]+", fname).groups()[0]
    fname_vals = fname.split("_")
    date = [int(i.strip("hms")) for i in fname_vals[0:6]]  # YY MM DD hh mm ss
    for seg in segments:
        start, end = float(seg[1]), float(seg[2])
        dur = end - start
        if seg[0] in ("male", "female") and dur > msd:
            sups.append(
                SupervisionSegment(
                    id=f"{fname}_{int(100*start):04}",
                    recording_id=fname,
                    start=start,
                    duration=round(dur, 4),
                    channel=0,
                    custom={
                        "date": date,
                        "lat": lat,
                        "lon": lon,
                        "station": station,
                        "est_gender": seg[0],
                    },
                )
            )
    return sups, reco


def prepare_radio(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    min_segment_duration: float = 0.5,
    num_jobs: int = 4,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Return the manifests which consist of recordings and supervisions
    :param corpus_dir: Path to the collected radio samples
    :param output_dir: Pathlike, the path where manifests are written
    :return: A Dict whose key is the dataset part and the value is a Dict with
        keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    segment_files = corpus_dir.rglob("segs/*/*.json")
    supervisions, recordings = [], []
    fun = partial(_make_reco_and_sups_from_file, msd=min_segment_duration)
    output_dir = Path(output_dir) if output_dir is not None else None
    output_dir.mkdir(mode=511, parents=True, exist_ok=True)
    with RecordingSet.open_writer(
        output_dir / "radio_recordings.jsonl.gz"
    ) as rec_writer:
        with SupervisionSet.open_writer(
            output_dir / "radio_supervisions.jsonl.gz"
        ) as sup_writer:
            for sups, reco in tqdm(
                parallel_map(
                    fun,
                    segment_files,
                    num_jobs=num_jobs,
                ),
                desc=f"Making recordings and supervisions",
            ):
                rec_writer.write(reco)
                for sup in sups:
                    sup_writer.write(sup)

            manifests = {
                "recordings": RecordingSet.from_jsonl_lazy(rec_writer.path),
                "supervisions": SupervisionSet.from_jsonl_lazy(sup_writer.path),
            }

    return manifests
