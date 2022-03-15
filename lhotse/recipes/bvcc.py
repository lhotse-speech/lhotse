import logging
from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike
from typing import Dict, Union, Optional
from pathlib import Path


def download_bvcc(target_dir) -> None:
    print(
        """
    Unfortunately you need to download the data manually due to licensing reason.

    See info and instructions how to obtain BVCC dataset used for VoiceMOS challange:
    - https://arxiv.org/abs/2105.02373
    - https://nii-yamagishilab.github.io/ecooper-demo/VoiceMOS2022/index.html
    - https://codalab.lisn.upsaclay.fr/competitions/695"""
    )


def prepare_bvcc(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)

    phase1_main = (corpus_dir / "phase1-main").resolve()
    assert phase1_main.exists(), f"Main track dir is missing {phase1_main}"

    main1_sets = phase1_main / "DATA" / "sets"
    main1_wav = phase1_main / "DATA" / "wav"
    assert (
        main1_sets.exists() and main1_wav.exists()
    ), f"Have you run data preparation in {phase1_main}?"
    main1_devp = main1_sets / "DEVSET"
    assert main1_devp.exists(), main1_devp
    main1_trainp = main1_sets / "TRAINSET"
    assert main1_trainp.exists(), main1_trainp

    phase1_ood = (corpus_dir / "phase1-ood").resolve()
    assert phase1_ood.exists(), f"Out of domain track dir is missing {phase1_ood}"
    ood1_sets = phase1_ood / "DATA" / "sets"
    ood1_wav = phase1_ood / "DATA" / "wav"
    assert (
        ood1_sets.exists() and ood1_wav.exists()
    ), f"Have you run data preparation in {phase1_ood}?"
    ood1_unlabeled = ood1_sets / "unlabeled_mos_list.txt"
    assert ood1_unlabeled.exists(), ood1_unlabeled
    ood1_devp = ood1_sets / "DEVSET"
    assert ood1_devp.exists(), ood1_devp
    ood1_trainp = ood1_sets / "TRAINSET"
    assert ood1_trainp.exists(), ood1_devp

    manifests = {}

    # ### Main track sets
    main1_recs = RecordingSet.from_dir(main1_wav, pattern="*.wav", num_jobs=num_jobs)

    logging.info("Preparing main1_dev")
    main1_dev_sup = SupervisionSet.from_segments(
        gen_supervision_per_utt(
            sorted(open(main1_devp).readlines()),
            main1_recs,
            parse_main_line,
        )
    )
    main1_dev_recs = main1_recs.filter(lambda rec: rec.id in main1_dev_sup)
    manifests["main1_dev"] = {
        "recordings": main1_dev_recs,
        "supervisions": main1_dev_sup,
    }

    logging.info("Preparing main1_train")
    main1_train_sup = SupervisionSet.from_segments(
        gen_supervision_per_utt(
            sorted(open(main1_trainp).readlines()),
            main1_recs,
            parse_main_line,
        )
    )
    main1_train_recs = main1_recs.filter(lambda rec: rec.id in main1_train_sup)
    manifests["main1_train"] = {
        "recordings": main1_train_recs,
        "supervisions": main1_train_sup,
    }

    # ### Out of Domain (OOD) track sets
    unlabeled_wavpaths = [
        ood1_wav / name.strip() for name in open(ood1_unlabeled).readlines()
    ]
    manifests["ood1_unlabeled"] = {
        "recordings": RecordingSet.from_recordings(
            Recording.from_file(p) for p in unlabeled_wavpaths
        )
    }

    ood1_recs = RecordingSet.from_dir(ood1_wav, pattern="*.wav", num_jobs=num_jobs)

    logging.info("Preparing ood1_dev")
    ood1_dev_sup = SupervisionSet.from_segments(
        gen_supervision_per_utt(
            sorted(open(ood1_devp).readlines()),
            ood1_recs,
            parse_ood_line,
        )
    )
    ood1_dev_recs = ood1_recs.filter(lambda rec: rec.id in ood1_dev_sup)
    manifests["ood1_dev"] = {
        "recordings": ood1_dev_recs,
        "supervisions": ood1_dev_sup,
    }

    logging.info("Preparing ood1_train")
    ood1_train_sup = SupervisionSet.from_segments(
        gen_supervision_per_utt(
            sorted(open(ood1_trainp).readlines()),
            ood1_recs,
            parse_ood_line,
        )
    )
    ood1_train_recs = ood1_recs.filter(lambda rec: rec.id in ood1_train_sup)
    manifests["ood1_train"] = {
        "recordings": ood1_train_recs,
        "supervisions": ood1_train_sup,
    }

    # Optionally serializing to disc
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for part, d in manifests.items():
            d["recordings"].to_file(output_dir / f"recordings_{part}.jsonl.gz")
            if "supervisions" in d:
                d["supervisions"].to_file(output_dir / f"supervisions_{part}.jsonl.gz")

    return manifests


def parse_main_line(line):
    """
    For context see phase1-main/README:

    TRAINSET and DEVSET contain the individual ratings from each rater, along with
    some demographic information for the rater.
    The format is as follows:

      sysID,uttID,rating,ignore,listenerinfo

    The listener info is as follows:

      {}_AGERANGE_LISTENERID_GENDER_[ignore]_[ignore]_HEARINGIMPAIRMENT

    """
    sysid, uttid, rating, _ignore, listenerinfo = line.split(",")
    _, agerange, listenerid, listener_mf, _, _, haveimpairment = listenerinfo.split("_")

    assert listener_mf in ["Male", "Female", "Others"], listener_mf
    if listener_mf == "Male":
        listener_mf = "M"
    elif listener_mf == "Female":
        listener_mf = "F"
    elif listener_mf == "Others":
        listener_mf = "O"
    else:
        ValueError(f"Unsupported value {listener_mf}")
    assert haveimpairment in ["Yes", "No"], haveimpairment
    haveimpairment = haveimpairment == "Yes"

    return (
        uttid,
        sysid,
        rating,
        {
            "id": listenerid,
            "M_F": listener_mf,
            "impairment": haveimpairment,
            "age": agerange,
        },
    )


def parse_ood_line(line):
    """
    For context see phase1-ood/README:

    TRAINSET and DEVSET contain the individual ratings from each rater, along with
    some demographic information for the rater.  (TRAINSET only contains information
    about the labeled training data, not for the unlabeled samples.)
    The format is as follows:

      sysID,uttID,rating,ignore,listenerinfo

    The listener info is as follows:

      {}_na_LISTENERID_na_na_na_LISTENERTYPE

    LISTENERTYPE may take the following values:
      EE: speech experts
      EP: paid listeners, native speakers of Chinese (any dialect)
      ER: voluntary listeners

    """
    sysid, uttid, rating, _ignore, listenerinfo = line.split(",")
    _, _, listenerid, _, _, _, listenertype = listenerinfo.split("_")

    assert listenertype in ["EE", "EP", "ER"]

    return (
        uttid,
        sysid,
        rating,
        {"id": listenerid, "type": listenertype},
    )


def gen_supervision_per_utt(lines, recordings, parse_line):
    prev_uttid, prev_sups = None, []
    for line in lines:
        line = line.strip()
        info = parse_line(line)
        uttid = info[0]
        if uttid != prev_uttid:
            yield from segment_from_run(prev_sups, recordings)
            prev_uttid, prev_sups = uttid, [info]
        else:
            prev_sups.append(info)
    if len(prev_sups) > 0:
        yield from segment_from_run(prev_sups, recordings)


def segment_from_run(infos, recordings):

    MOSd = {}
    LISTENERsd = {}
    uttidA, sysidA = None, None

    for uttid, sysid, rating, listenerd in infos:
        listenerid = listenerd.pop("id")
        MOSd[listenerid] = int(rating)
        LISTENERsd[listenerid] = listenerid

        if uttidA is None:
            uttidA = uttid
        else:
            assert uttid == uttidA, f"{uttid} vs {uttidA}"
        if sysidA is None:
            sysidA = sysid
        else:
            assert sysid == sysidA, f"{sysid} vs {sysidA}"
    if uttidA is not None:
        assert sysidA is not None and len(MOSd) > 0 and len(LISTENERsd) > 0
        if uttidA.endswith(".wav"):
            uttidA = uttidA[:-4]
        duration = recordings[uttidA].duration

        yield SupervisionSegment(
            id=uttidA,
            recording_id=uttidA,
            start=0,
            duration=duration,
            text=None,
            language=None,  # cloud be
            speaker=None,
            custom={"MOS": MOSd, "listeners": LISTENERsd},
        )
