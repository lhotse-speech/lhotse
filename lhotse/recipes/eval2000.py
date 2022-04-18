"""  
 About the eval2000 corpus
"""

import os
from typing import Dict, List, Optional, Tuple, Union   
from pathlib import Path
import numpy as np
from lhotse.audio import Recording, RecordingSet                                                                                             
from lhotse.qa import (                                                                                                                      
    fix_manifests,                                                                                                                           
    validate_recordings_and_supervisions,                                                                                                    
)                                                                                                                                            
from lhotse.supervision import SupervisionSegment, SupervisionSet 
from lhotse.utils import Pathlike, check_and_rglob

EVAL2000_AUDIO_DIR = "LDC2002S09"
EVAL2000_TRANSCRIPT_DIR = "LDC2002T43"

def prepare_eval2000(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    absolute_paths: bool = False,
    num_jobs: int = 1,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """                                                                                                                                      
    Prepares manifests for Eval2000.                                                                                         
                                                                                                                                        
    :param corpus_path: Path to global corpus                                                                                                
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.                                     
    :param absolute_paths: Whether to return absolute or relative (to the corpus dir) paths for recordings.                                  
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.                                                        
    """

    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(corpus_dir)
    print(output_dir)
    audio_partition_dir_path = corpus_dir / EVAL2000_AUDIO_DIR / "hub5e_00" / "english"
    print(audio_partition_dir_path)
    transcript_dir_path = corpus_dir / EVAL2000_TRANSCRIPT_DIR / "reference" / "english"
    print(transcript_dir_path)
    groups = []    
    for path in (audio_partition_dir_path).rglob("*.sph"):
        #print(path)
        base=Path(path).stem
        #print(base)
        groups.append(
            {
                "audio": path
            }
        )
    recordings = RecordingSet.from_recordings(
        Recording.from_file(
            group["audio"], relative_path_depth=None if absolute_paths else 3
        )
        for group in groups
    )
    segment_supervision = make_segments(transcript_dir_path)
    supervision_set = SupervisionSet.from_segments(segment_supervision)
    recordings, supervisions = fix_manifests(recordings, supervision_set)
    validate_recordings_and_supervisions(recordings, supervisions)  
    #print(segment_supervision)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_file(output_dir / "recordings_eval2000.jsonl")
        supervision_set.to_file(output_dir / "supervisions_unnorm_eval2000.jsonl")
    return {"recordings": recordings, "supervisions": supervisions}
                    
                    
def make_segments(transcript_dir_path,
                  omit_silence: bool = True):
    segment_supervision = []
    for text_path in (transcript_dir_path).rglob("*.txt"):
        trans_file=Path(text_path).stem
        #print(text_path)
        trans_file_lines = [ l.split() for l in open(text_path) ]
        #print(trans_file_lines)                                                                                                             
        id = -1
        for i in range(0, len(trans_file_lines)):
            if trans_file_lines[i]: # skip empty lines                                                                                       
                trans_line = trans_file_lines[i] # ref line                                                                                  
                if "#" not in trans_line[0]: #skip header lines of the file
                    id = id+1
                    start=float(trans_line[0])
                    end=float(trans_line[1])
                    duration=round(end - start, ndigits=8)
                    side=(trans_line[2].split(":"))[0]
                    if side == "A" :
                        channel=0
                    else:
                        channel=1
                    text_line=" ".join(trans_line[3::])
                    segment_id=trans_file+"-"+str(id)
                    recording_id=trans_file
                    speaker=trans_file+"-"+side
                    #print(segment_id, recording_id, start, duration, channel, text_line)
                    segment = SupervisionSegment(
                        id=segment_id,
                        recording_id=recording_id,
                        start=start,
                        duration=duration,
                        channel=channel,
                        language="English",
                        speaker=speaker,
                        text=text_line)
                    segment_supervision.append(segment)
    return segment_supervision
    # transcript lines  in one .txt file looks like this 
    """
    #Language: eng
    #File id: 5017  
    #Starting at 121 Ending at 421
    # 121 131 #BEGIN
    # 411 421 #END

    116.17 121.98 A: <contraction e_form="[we=>we]['re=>are]">we're starting the transition I you know told the students that they were going to you know, what the new plan was and   

    121.79 122.43 B: mhm   

    122.93 126.57 A: %um, <contraction e_form="[they=>they]['re=>are]">they're not that thrilled about it, but %uh    

    126.30 128.83 B: what to you mean? {breath} oh, about <contraction e_form="[you=>you]['re=>are]">you're leaving? 
    """
    
    
