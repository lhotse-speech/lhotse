"""
(1) The Voice Conversion Challenge 2018: database and results

AND

(2) Listening test results of the Voice Conversion Challenge 2018

Prepared for MOS prediction


(1) The Voice Conversion Challenge 2018: database and results

Citation
Lorenzo-Trueba, Jaime; Yamagishi, Junichi; Toda, Tomoki; Saito, Daisuke;
Villavicencio, Fernando; Kinnunen, Tomi;Ling, Zhenhua. (2018).
The Voice Conversion Challenge 2018: database and results, [sound].
The Centre for Speech Technology Research, The University of Edinburgh, UK.
https://doi.org/10.7488/ds/2337.

Description
Voice conversion (VC) is a technique to transform a speaker identity included in a source speech waveform
into a different one while preserving linguistic information of the source speech waveform.
In 2016, we have launched the Voice Conversion Challenge (VCC) 2016 at Interspeech 2016.
The objective of the 2016 challenge was to better understand different VC techniques built
on a freely-available common dataset to look at a common goal, and to share views
about unsolved problems and challenges faced by the current VC techniques.
The VCC 2016 focused on the most basic VC task, that is, the construction of VC models
that automatically transform the voice identity of a source speaker into that of a target speaker
using a parallel clean training database where source and target speakers read out
the same set of utterances in a professional recording studio.
17 research groups had participated in the 2016 challenge.
The challenge was successful and it established new standard evaluation methodology
and protocols for bench-marking the performance of VC systems.
In 2018, we launched the second edition of VCC, the VCC 2018. In this second edition,
we have revised three aspects of the challenge. First, we have reduced the amount of speech data
used for the construction of participant's VC systems to half. This is based on feedback
from participants in the previous challenge and this is also essential for practical applications.
Second, we introduced a more challenging task refereed to a Spoke task in addition to a similar task to the 1st edition,
which we call a Hub task. In the Spoke task, participants need to build their VC systems using a non-parallel database
in which source and target speakers read out different sets of utterances. We then evaluate both parallel
and non-parallel voice conversion systems via the same large-scale crowdsourcing listening test.
Third, we also attempted to bridge the gap between the ASV and VC communities.
Since new VC systems developed for the VCC 2018 may be strong candidates for enhancing the ASVspoof 2015 database,
we also asses spoofing performance of the VC systems based on anti-spoofing scores.
This repository contains the training and evaluation data released to participants,
submissions from participants, and the listening test results for the 2018 Voice Conversion Challenge.

COPYING: Creative Commons License: Attribution 4.0 International

Full license at: https://datashare.ed.ac.uk/bitstream/handle/10283/3061/license_text?sequence=11&isAllowed=y


(2) Listening test results of the Voice Conversion Challenge 2018

Citation
Yamagishi, Junichi; Wang, Xin. (2019). Listening test results of the Voice Conversion Challenge 2018, [dataset].
Centre for Speech Technology Research. University of Edinburgh.
https://doi.org/10.7488/ds/2496.

Description
This dataset is associated with a paper and a dataset below:
(1) Jaime Lorenzo-Trueba, Junichi Yamagishi, Tomoki Toda,
Daisuke Saito, Fernando Villavicencio, Tomi Kinnunen, Zhenhua Ling,
"The Voice Conversion Challenge 2018: Promoting Development of Parallel and Nonparallel Methods",
Proc Speaker Odyssey 2018, June 2018.
https://doi.org/10.21437/Odyssey.2018-28
(2) Lorenzo-Trueba, Jaime; Yamagishi, Junichi; Toda, Tomoki;
Saito, Daisuke; Villavicencio, Fernando; Kinnunen, Tomi; Ling, Zhenhua. (2018).
The Voice Conversion Challenge 2018: database and results, [sound].
The Centre for Speech Technology Research, The University of Edinburgh, UK.
https://doi.org/10.7488/ds/2337
and includes lists of listening test raw scores given by subjects and corresponding audio files that they assessed.
This was funded by The Japan Society for the Promotion of Science (Grant number: 15H01686, 16H06302, 17H04687, 17H06101)


COPYING: Creative Commons License: Attribution 4.0 International

Full license at: https://datashare.ed.ac.uk/bitstream/handle/10283/3257/license_text?sequence=2&isAllowed=y

Links:
    (1) https://datashare.ed.ac.uk/handle/10283/3061
    (2) https://datashare.ed.ac.uk/handle/10283/3257
"""
from tqdm import tqdm
from typing import Optional, Union, Dict
import logging
import shutil
import zipfile
import tarfile
from pathlib import Path

from lhotse.utils import Pathlike, urlretrieve_progress
from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)


VCC2018_SUBMITTED_SPEECH_URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz?sequence=10&isAllowed=y"  # noqa
VCC2018_MOS_SCORE_URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3257/vcc2018_listening_test_scores.zip?sequence=1&isAllowed=y"  # noqa
VCC2018_TARGET_REFERENCE_SPEECH_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_reference.zip?sequence=5&isAllowed=y"  # noqa
VCC2018_TARGET_SPEECH_TRN_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation_transcriptions.tar.gz?sequence=4&isAllowed=y"  # noqa


def download_vcc2018mos(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    reference_speech_url: Optional[str] = VCC2018_TARGET_REFERENCE_SPEECH_URL,
    submitted_speech_url: Optional[str] = VCC2018_SUBMITTED_SPEECH_URL,
    evaluation_results_url: Optional[str] = VCC2018_MOS_SCORE_URL,
    trn_url: Optional[str] = VCC2018_TARGET_SPEECH_TRN_URL,
):
    """
    Download and untar the speech submitted VCC2018 challange and the MOS results.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param url: str, the url of tarred/zipped VCTK corpus.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for url in [
        submitted_speech_url,
        evaluation_results_url,
        reference_speech_url,
        trn_url,
    ]:
        archive_name = url.split("/")[-1]
        archive_path = target_dir / archive_name
        part_dir = target_dir / archive_name.replace(".zip", "").replace(".tar.gz", "")
        completed_detector = part_dir / ".completed"
        if completed_detector.is_file():
            logging.info(
                f"Skipping {archive_name} because {completed_detector} exists."
            )
            return
        if force_download or not archive_path.is_file():
            urlretrieve_progress(
                url, filename=archive_path, desc=f"Downloading {archive_name}"
            )
        shutil.rmtree(part_dir, ignore_errors=True)
        opener = zipfile.ZipFile if archive_name.endswith(".zip") else tarfile.open
        with opener(archive_path) as archive:
            archive.extractall(path=target_dir)
        completed_detector.touch()


def prepare_vcc2018mos(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
    skip_smos: bool = False,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepares and returns the VCTK manifests which consist of Recordings and Supervisions.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a dict with keys "read" and "spontaneous".
        Each hold another dict of {'recordings': ..., 'supervisions': ...}
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    converted_speech_dir = corpus_dir / "mnt/sysope/test_files/testVCC2/"
    assert converted_speech_dir.is_dir(), f"No such directory: {converted_speech_dir}"

    mos_dir = corpus_dir / "vcc2018_listening_test_scores"
    assert mos_dir.is_dir(), f"No such directory: {mos_dir}"
    mos_scores = mos_dir / "vcc2018_evaluation_mos.txt"
    assert mos_scores.is_file()
    sim_scores = mos_dir / "vcc2018_evaluation_sim.txt"
    sim_scores.is_file()

    trn_dir = corpus_dir / "scripts"
    assert trn_dir.is_dir(), f"No such directory: {trn_dir}"

    reference_speech_dir = corpus_dir / "vcc2018_reference"
    assert reference_speech_dir.is_dir(), f"No such directory: {reference_speech_dir}"

    logging.info(
        f"Collecting reference target recordings for the VCC2018 challange from {reference_speech_dir}"
    )

    # TODO
    # def from_file(p: Path, prefix):
    #     # Match the format of converted recordings  B00_VCC2TF1_VCC2SF1_30001_HUB.wav
    #     return Recording.from_file(
    #         p, recording_id=f"reference_reference_{p.parent.stem}_{p.stem}_reference"
    #     )
    #
    # reference_target_recordings = RecordingSet.from_recordings(
    #     tqdm(
    #         map(from_file, reference_speech_dir.rglob("*.wav")),
    #         desc="Collecting reference recordings",
    #     )
    # )

    logging.info(
        f"Collecting converted recordings submitted to the VCC2018 challange from {converted_speech_dir}"
    )
    converted_recordings = RecordingSet.from_dir(
        converted_speech_dir, "*.wav", num_jobs=num_jobs
    )

    # # TODO distinguish based on prefix
    # # recordings = converted_recordings + reference_target_recordings
    # recordings = reference_target_recordings
    recordings = converted_recordings

    id2trn = {}
    for trnp in trn_dir.glob("*.txt"):
        with open(trnp, "rt") as r:
            id2trn[trnp.stem] = r.read().strip()

    supervisions = prepare_mos_supervisions(mos_scores, recordings, id2trn)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_jsonl(output_dir / "recordings.jsonl.gz")
        supervisions.to_jsonl(output_dir / "supervisions.jsonl.gz")

    return {"recordings": recordings, "supervisions": supervisions}


def prepare_mos_supervisions(
    mos_results_path, recordings: RecordingSet, id2trn: Dict[str, str]
) -> SupervisionSet:
    # TODO very slow -> make it faster it takes ~8min 170it/s
    # Use sort & group by instead of O(n*n) select

    mos = load_vcc_results(mos_results_path)[
        [
            "test_id",
            "user_id",
            "set_id",
            "set_idx",
            "system1_id",
            "src_spk",
            "tgt_spk",
            "MOS",
            "left_audio",
        ]
    ]
    recording_ids = set(mos["left_audio"].tolist())
    supervisions = []
    for recording_id_wav in tqdm(recording_ids, desc="Supervision creation"):
        recording_id = recording_id_wav.rstrip(".wav")
        # N17_VCC2TF1_VCC2SM3_30004_SPO -> 30004
        prompt_id = recording_id.split("_")[-2]

        rows = mos[mos["left_audio"] == recording_id_wav]
        mos_dict = {}
        for r in rows.itertuples(index=False):
            annotation_id = f"{r.user_id}_{r.set_id}_{r.set_idx}"
            mos_dict[annotation_id] = r.MOS
        # all rows should have the same properties as the first row if we use row
        row = next(rows.itertuples(index=False))
        tgt_spk = row.tgt_spk

        s = SupervisionSegment(
            id=f"{recording_id}",
            recording_id=recording_id,
            start=0,
            duration=recordings[recording_id].duration,
            text=id2trn[prompt_id],
            speaker=tgt_spk,
            gender=tgt_spk[-2],  # F/M extracted e.g. from e.g. VCC2TF2
            custom={
                "annotator": row.user_id,
                "MOS": mos_dict,
                "src_spk": row.src_spk,
                "system": row.system1_id,
                "prompt": prompt_id,
            },
        )
        supervisions.append(s)
    return SupervisionSet.from_segments(supervisions)


def load_vcc_results(path: Pathlike):
    """
    Returns pandas.DataFrame
    """
    #     """
    #     The headers are documented in vcc2018_evaluation_listening_test_results_with_sentence_ID.txt and copied here:
    #
    #     Test ID (ID of this listening test),
    #     User ID,
    #     Set ID that the current listener uses
    # (quality evaluation: G_setID.info, similarity evaluation: G2_setID.info),
    #     Sentence index within the current set,
    #     System ID in the quality evaluation task or  that of a left sample in the same/different task,
    #     System ID of a right sample in the same/different task
    # (\N means no sample, that is, the quality evaluation task),
    #     Source speaker,
    #     Target speaker,
    #     Task: hub (HUB) or spoke (SPO),
    #     Ground-truth of the same/different evaluation: S:same, D:different, \N:n/a,  (renamed verification)
    #     MOS score for naturalness: 1-5,
    #     Speaker similarity score: 1-4,
    #     Blank,
    #     Time stamp,
    #     Left audio sample,
    #     Right audio sample
    #     """
    import pandas as pd

    cols = [
        "test_id",
        "user_id",
        "set_id",
        "set_idx",
        "system1_id",
        "system2_id",
        "src_spk",
        "tgt_spk",
        "task",
        "verification",
        "MOS",
        "SSS",
        "_blank",
        "_timestamp",
        "left_audio",
        "right_audio",
    ]
    return pd.read_csv(path, header=None, usecols=list(range(len(cols))), names=cols)
