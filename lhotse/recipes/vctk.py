"""
This CSTR VCTK Corpus includes speech data uttered by 110 English
speakers with various accents. Each speaker reads out about 400
sentences, which were selected from a newspaper, the rainbow passage
and an elicitation paragraph used for the speech accent archive.

The newspaper texts were taken from Herald Glasgow, with permission
from Herald & Times Group. Each speaker has a different set of the
newspaper texts selected based a greedy algorithm that increases the
contextual and phonetic coverage. The details of the text selection
algorithms are described in the following paper:

C. Veaux, J. Yamagishi and S. King,
"The voice bank corpus: Design, collection and data analysis of
a large regional accent speech database,"
https://doi.org/10.1109/ICSDA.2013.6709856

The rainbow passage and elicitation paragraph are the same for all
speakers. The rainbow passage can be found at International Dialects
of English Archive:
(http://web.ku.edu/~idea/readings/rainbow.htm). The elicitation
paragraph is identical to the one used for the speech accent archive
(http://accent.gmu.edu). The details of the the speech accent archive
can be found at
http://www.ualberta.ca/~aacl2009/PDFs/WeinbergerKunath2009AACL.pdf

All speech data was recorded using an identical recording setup: an
omni-directional microphone (DPA 4035) and a small diaphragm condenser
microphone with very wide bandwidth (Sennheiser MKH 800), 96kHz
sampling frequency at 24 bits and in a hemi-anechoic chamber of
the University of Edinburgh. (However, two speakers, p280 and p315
had technical issues of the audio recordings using MKH 800).
All recordings were converted into 16 bits, were downsampled to
48 kHz, and were manually end-pointed.

This corpus was originally aimed for HMM-based text-to-speech synthesis
systems, especially for speaker-adaptive HMM-based speech synthesis
that uses average voice models trained on multiple speakers and speaker
adaptation technologies. This corpus is also suitable for DNN-based
multi-speaker text-to-speech synthesis systems and waveform modeling.

COPYING

This corpus is licensed under the Creative Commons License: Attribution 4.0 International
http://creativecommons.org/licenses/by/4.0/legalcode

VCTK VARIANTS
There are several variants of the VCTK corpus:
Speech enhancement
- Noisy speech database for training speech enhancement algorithms and TTS models where we added various types of noises to VCTK artificially: http://dx.doi.org/10.7488/ds/2117
- Reverberant speech database for training speech dereverberation algorithms and TTS models where we added various types of reverberantion to VCTK artificially http://dx.doi.org/10.7488/ds/1425
- Noisy reverberant speech database for training speech enhancement algorithms and TTS models http://dx.doi.org/10.7488/ds/2139
- Device Recorded VCTK where speech signals of the VCTK corpus were played back and re-recorded in office environments using relatively inexpensive consumer devices http://dx.doi.org/10.7488/ds/2316
- The Microsoft Scalable Noisy Speech Dataset (MS-SNSD) https://github.com/microsoft/MS-SNSD

ASV and anti-spoofing
- Spoofing and Anti-Spoofing (SAS) corpus, which is a collection of synthetic speech signals produced by nine techniques, two of which are speech synthesis, and seven are voice conversion. All of them were built using the VCTK corpus. http://dx.doi.org/10.7488/ds/252
- Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof 2015) Database. This database consists of synthetic speech signals produced by ten techniques and this has been used in the first Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof 2015) http://dx.doi.org/10.7488/ds/298
- ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database. This database has been used in the 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof 2019) https://doi.org/10.7488/ds/2555


ACKNOWLEDGEMENTS

The CSTR VCTK Corpus was constructed by:

        Christophe Veaux   (University of Edinburgh)
        Junichi Yamagishi  (University of Edinburgh)
        Kirsten MacDonald

The research leading to these results was partly funded from EPSRC
grants EP/I031022/1 (NST) and EP/J002526/1 (CAF), from the RSE-NSFC
grant (61111130120), and from the JST CREST (uDialogue).

Please cite this corpus as follows:
Christophe Veaux,  Junichi Yamagishi, Kirsten MacDonald,
"CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit",
The Centre for Speech Technology Research (CSTR),
University of Edinburgh
"""
import logging
import shutil
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.qa import remove_missing_recordings_and_supervisions
from lhotse.utils import Pathlike, urlretrieve_progress

EDINBURGH_VCTK_URL = (
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
)
CREST_VCTK_URL = "http://www.udialogue.org/download/VCTK-Corpus.tar.gz"


def download_vctk(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    url: Optional[str] = CREST_VCTK_URL,
) -> Path:
    """
    Download and untar/unzip the VCTK dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param url: str, the url of tarred/zipped VCTK corpus.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_name = url.split("/")[-1]
    archive_path = target_dir / archive_name
    part_dir = target_dir / archive_name.replace(".zip", "").replace(".tar.gz", "")
    completed_detector = part_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {archive_name} because {completed_detector} exists.")
        return part_dir
    if force_download or not archive_path.is_file():
        urlretrieve_progress(
            url, filename=archive_path, desc=f"Downloading {archive_name}"
        )
    shutil.rmtree(part_dir, ignore_errors=True)
    opener = zipfile.ZipFile if archive_name.endswith(".zip") else tarfile.open
    with opener(archive_path) as archive:
        archive.extractall(path=target_dir)
    completed_detector.touch()
    return part_dir


def prepare_vctk(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepares and returns the L2 Arctic manifests which consist of Recordings and Supervisions.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a dict with keys "read" and "spontaneous".
        Each hold another dict of {'recordings': ..., 'supervisions': ...}
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    speaker_meta = _parse_speaker_description(corpus_dir)

    recordings = RecordingSet.from_recordings(
        Recording.from_file(wav) for wav in (corpus_dir / "wav48").rglob("*.wav")
    )
    supervisions = []
    for path in (corpus_dir / "txt").rglob("*.txt"):
        # One utterance (line) per file
        text = path.read_text().strip()
        speaker = path.name.split("_")[0]  # p226_001.txt -> p226
        seg_id = path.stem
        meta = speaker_meta.get(speaker, defaultdict(lambda: None))
        if meta is None:
            logging.warning(f"Cannot find metadata for speaker {speaker}.")
        supervisions.append(
            SupervisionSegment(
                id=seg_id,
                recording_id=seg_id,
                start=0,
                duration=recordings[seg_id].duration,
                text=text,
                language="English",
                speaker=speaker,
                gender=meta["gender"],
                custom={
                    "accent": meta["accent"],
                    "age": meta["age"],
                    "region": meta["region"],
                },
            )
        )
    supervisions = SupervisionSet.from_segments(supervisions)

    # note(pzelasko): There were 172 recordings without supervisions when I ran it.
    #                 I am just removing them.
    recordings, supervisions = remove_missing_recordings_and_supervisions(
        recordings, supervisions
    )
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_json(output_dir / "recordings.json")
        supervisions.to_json(output_dir / "supervisions.json")

    return {"recordings": recordings, "supervisions": supervisions}


def _parse_speaker_description(corpus_dir: Pathlike):
    meta = {}
    lines = [
        line.split()
        for line in (corpus_dir / "speaker-info.txt").read_text().splitlines()
    ]
    header = lines[0]
    assert header == ["ID", "AGE", "GENDER", "ACCENTS", "REGION"]
    for spk, age, gender, accent, *region in lines[1:]:
        meta[f"p{spk}"] = {
            "age": int(age),
            "gender": gender,
            "accent": accent,
            "region": " ".join(region) if region is not None else None,
        }
    return meta
