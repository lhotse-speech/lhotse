"""
The KeSpeech is an open source speech dataset, KeSpeech, which involves 1,542 hours of speech 
signals recorded by 27,237 speakers in 34 cities in China, and the pronunciation includes 
standard Mandarin and its 8 subdialects. The new dataset possesses several properties. 
The dataset provides multiple labels including content transcription, speaker identity and 
subdialect, hence supporting a variety of speech processing tasks, such as speech recognition, 
speaker recognition, and subdialect identification, as well as other advanced techniques 
like multi-task learning and conditional learning.

Full paper: https://openreview.net/forum?id=b3Zoeq2sCLq
"""

import logging
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import AudioSource, Recording, RecordingSet, info
from lhotse.qa import validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.serialization import load_jsonl
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, compute_num_samples

KE_SPEECH_PARTS = ("train", "dev", "test")


def prepare_kespeech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike],
    dataset_parts: Union[str, Sequence[str]] = "auto",
    num_jobs: int = 1,
):
    pass
