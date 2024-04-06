"""
This script creates the BUT Reverb DB dataset, which is available at:
https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database

The following description is taken from the official website:

This is the first release of BUT Speech@FIT Reverb Database. The database is being built
with respect to collect a large number of various Room Impulse Responses,
Room environmental noises (or "silences"), Retransmitted speech (for ASR and SID testing),
and meta-data (positions of microphones, speakers etc.).

The goal is to provide speech community with a dataset for data enhancement and distant
microphone or microphone array experiments in ASR and SID.

The BUT Speech@FIT Reverb Dataset consists of 9 rooms:

Size [m x m x m]	Volume [m^3]	# RIRs	Ret.	Type	In RIR-Only set	In LibriSpeech-Only set
Q301	10.7x6.9x2.6	192	31 x 3	1	Office	Yes	Yes
L207	4.6x6.9x3.1	98	31 x 6	3	Office	Yes	Yes
L212	7.5x4.6x3.1	107	31 x 5	2	Office	Yes	Yes
L227	6.2x2.6x14.2	229	31 x 5	3	Stairs	Yes	Yes
R112	4.4x2.8x2.6*	~40	31 x 5	0	Hotel room	Yes	No
CR2	28.2x11.1x3.3	1033	31 x 4	0	Conf. room	Yes	No
E112	11.5x20.1x4.8*	~900	31 x 2	0	Lect. room	Yes	No
D105	17.2x22.8x6.9*	~2000	31 x 6	1	Lect. room	Yes	Yes
C236	7.0x4.1x3.6	102	31 x 10	0	Meeting room	Yes	No

We placed 31 microphones in both rooms. The source (a Hi-Fi loudspeaker) was placed on 5
positions in average. We measured RIRs (using exponential sine sweep method) for each
speaker position. Next we recorded environmental noise (silence). There was a radio at
background playing in one speaker position in the office.

All microphone positions are measured and stored in meta-files. We pre-calculated positions
of microphones and speakers in Cartesian and polar coordinates as absolute and relative (to the speaker).

The corpus can be cited as follows:
@ARTICLE{8717722,
  author={Szöke, Igor and Skácel, Miroslav and Mošner, Ladislav and Paliesek, Jakub and Černocký, Jan},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  title={Building and evaluation of a real room impulse response dataset},
  year={2019},
  volume={13},
  number={4},
  pages={863-876},
  doi={10.1109/JSTSP.2019.2917582}}
"""
import logging
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import CutSet, Recording, RecordingSet
from lhotse.utils import Pathlike, resumable_download

BUT_REVERB_DB_URL = (
    "http://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz"
)


def download_but_reverb_db(
    target_dir: Pathlike = ".",
    url: Optional[str] = BUT_REVERB_DB_URL,
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and untar the BUT Reverb DB dataset.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param url: str, the url that downloads file called BUT_ReverbDB.tgz.
    :param force_download: bool, if True, download the archive even if it already exists.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tgz_name = "BUT_ReverbDB.tgz"
    tgz_path = target_dir / tgz_name
    if tgz_path.exists() and not force_download:
        logging.info(f"Skipping {tgz_name} because file exists.")
    resumable_download(url, tgz_path, force_download=force_download)
    tgz_dir = target_dir / "BUT_ReverbDB"
    if not tgz_dir.exists():
        logging.info(f"Untarring {tgz_name}.")
        with tarfile.open(tgz_path) as tar:
            tar.extractall(path=target_dir)
    return tgz_dir


def prepare_but_reverb_db(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    parts: Sequence[str] = ("silence", "rir"),
) -> Dict[str, Dict[str, Union[RecordingSet, CutSet]]]:
    """
    Prepare the BUT Speech@FIT Reverb Database corpus.

    :param corpus_dir: Pathlike, the path of the dir to store the dataset.
    :param output_dir: Pathlike, the path of the dir to write the manifests.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if not parts:
        raise ValueError("No parts specified for manifest preparation.")
    if isinstance(parts, str):
        parts = [parts]

    recordings = defaultdict(list)
    for wav_file in tqdm(corpus_dir.rglob("*.wav")):
        part = wav_file.parent.name.lower()
        if part not in parts:
            continue
        # The file path is like:
        # VUT_FIT_C236/MicID01/SpkID01_20190503_S/01/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav
        # We convert it to the following for the recording ID:
        # VUT_FIT_C236-MicID01-SpkID01_20190503_S-01-v00
        room_id = wav_file.parent.parent.parent.parent.parent.stem
        mic_id = wav_file.parent.parent.parent.parent.stem
        spk_id = wav_file.parent.parent.parent.stem
        uid = wav_file.parent.parent.stem
        version = wav_file.stem.split(".")[-1]
        recording_id = f"{room_id}-{mic_id}-{spk_id}-{uid}-v{version}"
        recording = Recording.from_file(wav_file, recording_id=recording_id)
        recordings[part].append(recording)

    manifests = defaultdict(dict)
    for part in parts:
        manifests[part]["recordings"] = RecordingSet.from_recordings(recordings[part])

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for part in parts:
            manifests[part]["recordings"].to_file(
                output_dir / f"but-reverb-db_{part}_recordings.jsonl.gz"
            )

    return manifests
