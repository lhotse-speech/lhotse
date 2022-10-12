from glob import glob
from itertools import islice
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions, CutSet
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

import random

"""
Corpus owner: https://clrd.ninjal.ac.jp/csj/en/index.html
Corpus description: 
- http://www.lrec-conf.org/proceedings/lrec2000/pdf/262.pdf
- https://isca-speech.org/archive_open/archive_papers/sspr2003/sspr_mmo2.pdf 

This script assumes that the transcript directory that was passed in has been parsed by csj_make_transcript.py. 
Individual speaker IDs - or more precisely, session IDs, to cater for the 'D' dialogue cases - each have their own 
folder. These are omitted as '...' in the directory tree below. 
Notice that the 'D' transcripts are split into respective (L)eft and (R)ight channels.

{transcript_dir}
 - excluded
   - ...
 - core 
   - ...
 - eval1
   - ...
 - eval2
   - ...
 - eval3 
   - ...
 - noncore 
   - ...
   - A01F0576
     - A01F0576.sdb (not used in this script)
     - A01F0576-{transcript_mode}.txt
     - A01F0576-segments (not used in this script)
     - A01F0576-wav.list 
   - ...
   - D03M0038
     - D03M0038.sdb (not used in this script)
     - D03M0038-L-{transcript_mode}.txt
     - D03M0038-L-segments (not used in this script)
     - D03M0038-L-wav.list
     - D03M0038-R-{transcript_mode}.txt
     - D03M0038-R-segments (not used in this script)
     - D03M0038-R-wav.list

This script follows the espnet method of splitting the remaining core+noncore utterances into valid and train cutsets at index 4000.

In other words, after moving the eval1, eval2, eval3, and excluded utterances out, the remaining utterances within the core and noncore 
utterances are shuffled. The first 4000 utterances of the shuffled set go to the `valid` cutset and are not subjected to speed 
perturbation. The remaining utterances become the `train` cutset and are speed-perturbed (0.9x, 1.0x, 1.1x) before saved to disk.  

These cutsets serve as blueprints only. Fbank extraction is left for compute_fbank_csj.py.

"""

ORI_DATA_PARTS = (
    "eval1",
    "eval2",
    "eval3",
    "core",
    "noncore",
)

SPLIT_DATA_PARTS = (
    "train",
    "valid",
    "eval1",
    "eval2",
    "eval3"
)

RNG_SEED = 42

def parse_transcript_header(line : str):
    sgid, start, end, line = line.split(' ', maxsplit=3)
    return (sgid, float(start), float(end), line)

def parse_one_recording(
    template : str,
    wavlist_path : Path, 
    recording_id : str
) -> Tuple[Recording, List[SupervisionSegment]]:
    transcripts = []
    
    for trans in glob(template + '*.txt'):
        trans_type = trans.replace(template + '-', '').replace(".txt", '')
        transcripts.append([(trans_type, t) for t in Path(trans).read_text().split('\n')])
    
    assert all(len(c) == len(transcripts[0]) for c in transcripts), transcripts
    wav = wavlist_path.read_text()
        
    recording = Recording.from_file(wav, recording_id=recording_id)
    
    supervision_segments = []
    
    for texts in zip(*transcripts):
        customs = {}
        for trans_type, line in texts:
            sgid, start, end, customs[trans_type] = parse_transcript_header(line)
            
        text = texts[0][1] if len(customs) == 1 else ""
            
        supervision_segments.append(
            SupervisionSegment(
                id=sgid,
                recording_id=recording_id,
                start=start,
                duration=(end-start),
                channel=0,
                language="Japanese",
                speaker=recording_id,
                text=text,
                custom=customs
            )
        )
    
    return recording, supervision_segments

def prepare_csj(
    transcript_dir : Pathlike, 
    dataset_parts : Union[str, Sequence[str]] = ORI_DATA_PARTS, 
    output_dir : Pathlike = None,
    num_jobs : int = 1,
    split : int = 4000
) -> Dict[str, Dict[str, CutSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param transcript_dir: Pathlike, the path to the transcripts. Assumes that that the transcripts were processed by csj_make_transcript.py. 
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param split: int, the index for the core+noncore CutSet, before which goes to the validation set, and after which goes to the training set. 
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'cuts'.
    """
    
    transcript_dir = Path(transcript_dir)
    assert transcript_dir.is_dir(), f"No such directory for transcript_dir: {transcript_dir}"

    if not dataset_parts:
        dataset_parts = ORI_DATA_PARTS

    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]
    
    if output_dir:  
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exit: we can read them and save a bit of preparation time. 
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir, types=["cuts"],
        )
    
    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            logging.info(f"Processing CSJ subset: {part}")
            if manifests_exist(types=["cuts"], part=part, output_dir=output_dir):
                logging.info(f"CSJ subset: {part} already prepared - skipping.")
                continue
            
            recordings = []
            supervisions = []
            part_path = transcript_dir / part
            futures = []
            
            for wavlist in part_path.glob("*/*-wav.list"):
                template = wavlist.as_posix().rstrip('-wav.list')
                spk = wavlist.name.rstrip('-wav.list')
                futures.append(
                    ex.submit(parse_one_recording, template, wavlist, spk)
                )

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                assert result
                recording, segments = result
                recordings.append(recording)
                supervisions.extend(segments)
            
            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)
            validate_recordings_and_supervisions(recording_set, supervision_set)
            
            if output_dir:
                supervision_set.to_file(output_dir / f"supervisions_{part}.jsonl.gz")
                recording_set.to_file(output_dir / f"recordings_{part}.jsonl.gz")
            
    
    # Create train and valid cuts
    logging.info(f"Loading, trimming, and shuffling the remaining core+noncore cuts.")
    recording_set = RecordingSet.from_file(output_dir / "recordings_core.jsonl.gz") \
        + RecordingSet.from_file(output_dir / "recordings_noncore.jsonl.gz")
    supervision_set = SupervisionSet.from_file(output_dir / "supervisions_core.jsonl.gz") \
        + SupervisionSet.from_file(output_dir / "supervisions_noncore.jsonl.gz")
        
    cut_set = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set
    )
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
    
    cut_set = cut_set.shuffle(random.Random(RNG_SEED))
    
    logging.info(f"Creating valid cuts, split at {split}")
    valid_set = CutSet.from_cuts(islice(cut_set, 0, split))
    valid_set.to_jsonl(output_dir / "cuts_valid.jsonl.gz")
    
    logging.info(f"Creating train cuts")
    train_set = CutSet.from_cuts(islice(cut_set, split, None))
    
    train_set = (
        train_set
        + train_set.perturb_speed(0.9)
        + train_set.perturb_speed(1.1)
    )
    train_set.to_jsonl(output_dir / "cuts_train.jsonl.gz")
    
    manifests = {
        "valid": valid_set,
        "train": train_set, 
    } 
    
    logging.info("Creating eval cuts.")
    # Create eval datasets
    for i in range(1, 4):
        cut_set = CutSet.from_manifests(
            recordings=RecordingSet.from_file(output_dir / f"recordings_eval{i}.jsonl.gz"),
            supervisions=SupervisionSet.from_file(output_dir / f"supervisions_eval{i}.jsonl.gz")
        )
        cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
        cut_set.to_jsonl(output_dir / f"cuts_eval{i}.jsonl.gz")
        manifests[f"eval{i}"] = cut_set
    
    return manifests