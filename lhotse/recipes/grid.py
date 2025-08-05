"""
The Grid Corpus is a large multitalker audiovisual sentence corpus designed to support joint
computational-behavioral studies in speech perception. In brief, the corpus consists of high-quality
audio and video (facial) recordings of 1000 sentences spoken by each of 34 talkers (18 male, 16 female),
for a total of 34000 sentences. Sentences are of the form "put red at G9 now".

Source: https://zenodo.org/record/3625687
"""
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.supervision import AlignmentItem, SupervisionSegment
from lhotse.utils import Pathlike, is_module_available
import os
import shutil
import tempfile


GRID_ZENODO_ID = "10.5281/zenodo.3625687"



def download_grid(
    target_dir: Pathlike = ".",
    force_download: bool = False,
) -> Path:
    """
    Download and untar the dataset, supporting both LibriSpeech and MiniLibrispeech

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param dataset_parts: "librispeech", "mini_librispeech",
        or a list of splits (e.g. "dev-clean") to download.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param alignments: should we download the alignments. The original source is:
        https://github.com/CorentinJ/librispeech-alignments
    :param base_url: str, the url of the OpenSLR resources.
    :param alignments_url: str, the url of LibriSpeech word alignments
    :return: the path to downloaded and extracted directory with data.
    """
    if not is_module_available("zenodo_get"):
        raise RuntimeError(
            "To download Grid Audio-Visual Speech Corpus please 'pip install zenodo_get'."
        )
        
    corpus_dir = Path(target_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    download_marker = corpus_dir / ".downloaded"
    if not download_marker.exists() or force_download:
        subprocess.run(
            f"zenodo_get {GRID_ZENODO_ID}", shell=True, check=True, cwd=corpus_dir
        )
        download_marker.touch()

    for p in tqdm(corpus_dir.glob("*.zip"), desc="Unzipping files"):
        with zipfile.ZipFile(p) as f:
            f.extractall(corpus_dir)
        
    # Speaker mapping to fix mis-assigned alignment data
    speaker_fix_map = {
        's1': 's1', 's2': 's2', 's3': 's3', 's4': 's4',
        's5': 's6', 's6': 's5', 's7': 's7', 's8': 's8',
        's9': 's9', 's10': 's13', 's11': 's10', 's12': 's11',
        's13': 's12', 's14': 's15', 's15': 's14', 's16': 's16',
        's17': 's17', 's18': 's19', 's19': 's18', 's20': 's21',
        's22': 's23', 's23': 's22', 's24': 's24', 's25': 's25',
        's26': 's27', 's27': 's26', 's28': 's29', 's29': 's28',
        's30': 's30', 's31': 's31', 's32': 's33', 's33': 's32',
        's34': 's34',        
    }
    
    #Downloaded alignment folder 
    input_dir = corpus_dir / "alignments"
    #temporary folder to contain the corrected alignments
    temp_dir = corpus_dir / "tmp_alignments"

    os.makedirs(temp_dir, exist_ok=True)

    for source_folder, target_speaker in speaker_fix_map.items():
        src_path = os.path.join(input_dir, source_folder)
        tgt_path = os.path.join(temp_dir, target_speaker)
        
        if not os.path.exists(src_path):
            print(f"Source folder does not exist: {src_path}")
            continue
        
        if os.path.exists(tgt_path):
            shutil.rmtree(tgt_path)
        
        shutil.copytree(src_path, tgt_path)
        print(f"Copied entire folder from {source_folder} to {target_speaker}")
    shutil.rmtree(input_dir)
    os.rename(temp_dir, input_dir)
 
    return corpus_dir


def prepare_grid(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    with_supervisions: bool = True,
    num_jobs: int = 1,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param with_supervisions: bool, when False, we'll only return recordings; when True, we'll also
        return supervisions created from alignments, but might remove some recordings for which
        they are missing.
    :param num_jobs: int, number of parallel jobs.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    ali_dir = corpus_dir / "alignments"
    assert ali_dir.is_dir()
    audio_dir = corpus_dir / "audio_25k"
    assert audio_dir.is_dir()

    recordings = []
    supervisions = []

    video_dirs = [p for p in corpus_dir.glob("s*") if not p.suffix == ".zip"]
    futures = []
    with ProcessPoolExecutor(num_jobs) as ex:
        for speaker_dir in video_dirs:
            speaker = speaker_dir.name
            for video_path in speaker_dir.glob("*.mpg"):
                futures.append(
                    ex.submit(
                        process_single, video_path, speaker, ali_dir, with_supervisions
                    )
                )

        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Scanning videos"
        ):
            recording, maybe_supervision = f.result()
            recordings.append(recording)
            if maybe_supervision is not None:
                supervisions.append(maybe_supervision)

    recordings = RecordingSet.from_recordings(recordings)
    if with_supervisions:
        supervisions = SupervisionSet.from_segments(supervisions)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        recordings.to_file(output_dir / "grid_recordings.jsonl.gz")
        if with_supervisions:
            supervisions.to_file(output_dir / "grid_supervisions.jsonl.gz")

    ans = {"recordings": recordings}
    if with_supervisions:
        ans.update(supervisions=supervisions)
    return ans

def convert_to_mono_audio(video_path):
    
    original_mode = os.stat(video_path).st_mode
    # Create a temporary output file in the same directory
    with tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False, dir=video_path.parent) as tmp:
        temp_output = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y", 
            "-i", str(video_path),
            "-ac", "1",         # Mono audio
            "-ar", "16000",     
            "-c:v", "copy",     # Copy video stream
            str(temp_output)
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        # Overwrite original file
        shutil.move(str(temp_output), str(video_path))
        os.chmod(video_path, original_mode)

    finally:
        if temp_output.exists():
            temp_output.unlink(missing_ok=True)

def process_single(
    video_path: Path, speaker: str, ali_dir: Path, with_supervisions: bool
):
    #stereo data causes error for some video files
    convert_to_mono_audio(video_path)
    video_id = video_path.stem
    recording = Recording.from_file(video_path, recording_id=f"{speaker}_{video_id}")

    supervision = None
    ali_path = (ali_dir / speaker / video_id).with_suffix(".align")
    if with_supervisions and ali_path.is_file():
        ali = [
            AlignmentItem(
                symbol=w,
                start=float(b) / 1000,
                duration=float(int(e) - int(b)) / 1000,
            )
            for b, e, w in (line.split() for line in ali_path.read_text().splitlines())
        ]
        supervision = SupervisionSegment(
            id=recording.id,
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            channel=recording.channel_ids,
            text=" ".join(item.symbol for item in ali if item.symbol != "sil"),
            language="English",
            speaker=speaker,
            alignment={"word": ali},
        )

    return recording, supervision
