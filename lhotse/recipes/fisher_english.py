"""
About the Fisher English Part 1,2 corpus

    This is conversational telephone speech collected as 2-channel, 8kHz-sampled data. 
    The catalog number LDC2004S13 and LDC2005S13 for audio corpora and LDC2004T19 LDC2005T19 for transcripts.

    This data is not available for free - your institution needs to have an LDC subscription.
"""

import codecs
import os
import itertools as it

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional, Union

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import trim_supervisions_to_recordings, validate_recordings_and_supervisions
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob


FISHER_AUDIO_DIRS      = ['LDC2004S13', 'LDC2005S13']
FISHER_TRANSCRIPT_DIRS = ['LDC2004T19', 'LDC2005T19']


def get_paths(fold_path_and_pattern: Tuple[Pathlike, str]) -> List[Path]:
    return check_and_rglob(*fold_path_and_pattern)


def create_recording(audio_path_and_rel_path_depth: Tuple[Pathlike, Union[int, None]]) -> Recording:
    audio_path, rel_path_depth = audio_path_and_rel_path_depth
    return Recording.from_file(audio_path, relative_path_depth=rel_path_depth)


def create_supervision(
    sessions_and_transcript_path: Tuple[Dict[str, Dict[str, str]], Pathlike]
) -> List[SupervisionSegment]:
    
    sessions, transcript_path = sessions_and_transcript_path
    transcript_path = Path(transcript_path)
    channel_to_int = {'A': 0, 'B': 1}
    session_id = transcript_path.stem.split('_')[2]
    with codecs.open(transcript_path, 'r', 'utf8') as trans_f:
        lines = [l.rstrip('\n') for l in trans_f.readlines()][3:]
        lines = [l.split() for l in lines if l.strip() != '']
        lines = [[float(l[0]), float(l[1]), l[2][:-1], ' '.join([w for w in l[3:] if w.strip() != ''])] 
                 for l in lines]
        
        # fix obvious transcript error
        if session_id == '11487':
            lines = [[231.09, *l[1:]] if l[0] == 31.09 and l[1] == 234.06 else l for l in lines]
        
        segments = [
            SupervisionSegment(
                id=transcript_path.stem + '-' + str(k).zfill(len(str(len(lines)))),
                recording_id=transcript_path.stem,
                start=round(l[0], 3),
                duration=round(l[1] - l[0], 3),
                channel=channel_to_int[l[2]],
                text=l[3],
                language='English',
                speaker=sessions[session_id][l[2]]
            )
            for k, l in enumerate(lines)
        ]
        
    return segments


def walk_dirs_parallel(dirs: List[Pathlike], pattern: str, pbar_desc: str) -> List[Path]:

    get_path_inputs = [(Path(dir_path), pattern) for dir_path in dirs]
    output_paths = [None] * len(dirs)
    njobs = min(len(dirs), os.cpu_count() * 4)
    with ThreadPoolExecutor(njobs) as executor:
        with tqdm(total=len(get_path_inputs), desc=pbar_desc) as pbar:
            for k, tmp_output_paths in enumerate(executor.map(get_paths, get_path_inputs)):
                output_paths[k] = tmp_output_paths
                pbar.update()
    output_paths = sorted(it.chain.from_iterable(output_paths))
    
    return output_paths


def prepare_fisher_english(
    corpus_path: Pathlike,
    audio_dirs: List[str] = FISHER_AUDIO_DIRS,
    transcript_dirs: List[str] = FISHER_TRANSCRIPT_DIRS,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: bool = False
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    
    """
    Prepares manifests for Fisher English Part 1, 2. 
    Script assumes that audio_dirs and transcript_dirs are in the corpus_path.
    We create two manifests: one with recordings, and the other one with text supervisions.
    
    :param corpus_path: Path to Fisher corpus
    :param audio_dirs: List of dirs of audio corpora.
    :param transcripts_dirs: List of dirs of transcript corpora.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param absolute_paths: Whether to return absolute or relative (to the corpus dir) paths for recordings.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    
    corpus_path = Path(corpus_path)
    
    for workdir in audio_dirs + transcript_dirs:
        workdir_path = corpus_path / workdir
        if not workdir_path.is_dir():
            raise ValueError(f"Could not find '{workdir}' directory inside '{corpus_path}'.")
            
    audio_subdir_paths = []
    for audio_dir in audio_dirs:
        audio_dir_path = corpus_path / audio_dir
        for audio_partition_dir in audio_dir_path.iterdir():
            audio_partition_dir_path = audio_dir_path / audio_partition_dir / 'audio'
            audio_subdir_paths += [audio_partition_dir_path / audio_subdir 
                                   for audio_subdir in audio_partition_dir_path.iterdir()]
            
    transcript_subdir_paths = []
    for transcript_dir in transcript_dirs:
        transcript_dir_path = corpus_path / transcript_dir / 'data' / 'trans'
        transcript_subdir_paths += [transcript_dir_path / transcript_subdir 
                                    for transcript_subdir in transcript_dir_path.iterdir()]
    
    audio_paths = walk_dirs_parallel(audio_subdir_paths, '*.sph', 'Parsing audio sub-dirs')
    transcript_paths = walk_dirs_parallel(transcript_subdir_paths, '*.txt', 'Parsing transcript sub-dirs')
    
    sessions = {}
    for transcript_dir in transcript_dirs:
        sessions_data_path = check_and_rglob(corpus_path / transcript_dir / 'doc', '*_calldata.tbl')[0]
        with codecs.open(sessions_data_path, 'r', 'utf8') as sessions_data_f:
            tmp_sessions = [l.rstrip('\n').split(',') for l in sessions_data_f.readlines()][1:]
            sessions.update({l[0]: {'A': l[5], 'B': l[10]} for l in tmp_sessions})

    assert len(transcript_paths) == len(sessions) == len(audio_paths)
    
    create_recordings_input = [(p, None if absolute_paths else 5) for p in audio_paths]
    recordings = [None] * len(audio_paths)
    with ThreadPoolExecutor(os.cpu_count() * 4) as executor:
        with tqdm(total=len(create_recordings_input), desc='Collect recordings') as pbar:
            for i, reco in enumerate(executor.map(create_recording, create_recordings_input)):
                recordings[i] = reco
                pbar.update()
        
    recordings = RecordingSet.from_recordings(recordings)
    
    create_supervisions_input = [(sessions, p) for p in transcript_paths]
    supervisions = [None] * len(create_supervisions_input)
    with ThreadPoolExecutor(os.cpu_count() * 4) as executor:
        with tqdm(total=len(create_supervisions_input), desc='Create supervisions') as pbar:
            for i, tmp_supervisions in enumerate(executor.map(create_supervision, create_supervisions_input)):
                supervisions[i] = tmp_supervisions
                pbar.update()
    supervisions = list(it.chain.from_iterable(supervisions))
    supervisions = SupervisionSet.from_segments(supervisions)
    
    supervisions = trim_supervisions_to_recordings(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_file(output_dir / 'recordings.jsonl.gz')
        supervisions.to_file(output_dir / 'supervisions.jsonl.gz')
    
    return {
        'recordings': recordings,
        'supervisions': supervisions
    }
