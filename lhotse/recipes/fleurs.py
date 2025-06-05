"""
This recipe provides functionality for downloading and preparing the fleurs
corpus. The data is hosted on huggingface and to enable more control of the
download format, we use the streaming download interface and save each audio
file as it is streamed. The download can take quite some time.

The fleurs corpus consist of data in 102 languages spoken by multiple speakers.
There is about 10 hrs of trainign data in each language with smaller
accompanying dev and test sets. Full details can be found in

@inproceedings{conneau2023fleurs,
  title={Fleurs: Few-shot learning evaluation of universal representations of speech},
  author={Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},
  pages={798--805},
  year={2023},
  organization={IEEE}
}
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    audio,
    fix_manifests,
    get_ffmpeg_torchaudio_info_enabled,
    set_ffmpeg_torchaudio_info_enabled,
)
from lhotse.parallel import parallel_map
from lhotse.utils import Pathlike, is_module_available

# The FLEURS languages are indicated by 2-letter ISO-codes followed by a
# country code, i.e.,
#
#  en_us, fr_fr, ml_in
#
# for American English, French French and Indian Malayalam respectively.

DEFAULT_LANGUAGES = [
    "af_za",
    "am_et",
    "ar_eg",
    "as_in",
    "ast_es",
    "az_az",
    "be_by",
    "bg_bg",
    "bn_in",
    "bs_ba",
    "ca_es",
    "ceb_ph",
    "ckb_iq",
    "cmn_hans_cn",
    "cs_cz",
    "cy_gb",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fa_ir",
    "ff_sn",
    "fi_fi",
    "fil_ph",
    "fr_fr",
    "ga_ie",
    "gl_es",
    "gu_in",
    "ha_ng",
    "he_il",
    "hi_in",
    "hr_hr",
    "hu_hu",
    "hy_am",
    "id_id",
    "ig_ng",
    "is_is",
    "it_it",
    "ja_jp",
    "jv_id",
    "ka_ge",
    "kam_ke",
    "kea_cv",
    "kk_kz",
    "km_kh",
    "kn_in",
    "ko_kr",
    "ky_kg",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "lo_la",
    "lt_lt",
    "luo_ke",
    "lv_lv",
    "mi_nz",
    "mk_mk",
    "ml_in",
    "mn_mn",
    "mr_in",
    "ms_my",
    "mt_mt",
    "my_mm",
    "nb_no",
    "ne_np",
    "nl_nl",
    "nso_za",
    "ny_mw",
    "oc_fr",
    "om_et",
    "or_in",
    "pa_in",
    "pl_pl",
    "ps_af",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sd_in",
    "sk_sk",
    "sl_si",
    "sn_zw",
    "so_so",
    "sr_rs",
    "sv_se",
    "sw_ke",
    "ta_in",
    "te_in",
    "tg_tj",
    "th_th",
    "tr_tr",
    "uk_ua",
    "umb_ao",
    "ur_pk",
    "uz_uz",
    "vi_vn",
    "wo_sn",
    "xh_za",
    "yo_ng",
    "yue_hant_hk",
    "zu_za",
]


def download_fleurs(
    target_dir: Pathlike = ".",
    languages: Optional[Union[str, Sequence[str]]] = "all",
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download the specified fleurs datasets.

    :param target_dir: The path to which the corpus will be downloaded.
    :type target_dir: Pathlike
    :param languages: Optional list of str or str specifying which
        languages to download. The str specifier for a language has the
        ISOCODE_COUNTRYCODE format, and is all lower case. By default
        this is set to "all", which will download the entire set of
        languages.
    :type languages: Optional[Union[str, Sequence[str]]]
    :param force_download: Specifies whether to overwrite an existing
        archive.
    :type force_download: bool
    :return: The root path of the downloaded data
    :rtype: Path
    """
    target_dir = Path(target_dir)
    corpus_dir = target_dir / "fleurs"
    metadata_dir = corpus_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(languages, str) and languages == "all" or languages[0] == "all":
        languages = DEFAULT_LANGUAGES

    if isinstance(languages, str):
        languages = [languages]

    for lang in tqdm(languages):
        # Download one language at a time
        lang_dir = corpus_dir / lang
        download_single_fleurs_language(
            lang_dir,
            lang,
            force_download,
        )
    return corpus_dir


def download_single_fleurs_language(
    target_dir: Pathlike,
    language: str,
    force_download: bool = False,
) -> Path:
    """
    Download a single fleurs language

    :param target_dir: The path to which one langauge will be downloaded
    :type target_dir: Pathlike
    :param language: The code for the specified language
    :type language: str
    :param force_download: Specifies whether to overwrite an existing
        archive.
    :type force_download: bool
    :return: The path to the downloaded data for the specified language
    :rtype: Path
    """
    if not is_module_available("datasets"):
        raise ImportError(
            "The huggingface datasets package is not installed. Please install"
            " ...(pip install datasets)"
        )
    else:
        from datasets import load_dataset

    def _identity(x):
        return x

    target_dir = Path(target_dir)
    metadata_dir = target_dir.parents[0] / "metadata" / language
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = target_dir / f".{language}_completed"
    if completed_detector.is_file() and not force_download:
        logging.info("Skipping dowload because {completed_detector} exists.")
        return target_dir

    for split in tqdm(["train", "validation", "test"]):
        fleurs = load_dataset(
            "google/fleurs",
            language,
            cache_dir=target_dir,
            streaming=True,
            split=split,
        )
        metadata = []
        osplit = "dev" if split == "validation" else split
        split_dir = target_dir / osplit
        split_dir.mkdir(parents=True, exist_ok=True)
        for data in tqdm(fleurs, desc=f"Downloading data from {language}-{osplit}"):
            audio.save_audio(
                f"{split_dir}/{Path(data['audio']['path']).name}",
                data["audio"]["array"],
                data["audio"]["sampling_rate"],
            )
            metadata_ = [
                str(data["id"]),  # ID
                Path(data["audio"]["path"]).name,  # filename
                data["raw_transcription"],  # raw transcript
                data["transcription"],  # transcript
                " ".join("|".join(data["transcription"].split())) + " |",  # chars
                str(data["num_samples"]),  # number of audio samples
                "FEMALE" if data["gender"] == 1 else "MALE",  # gender
            ]
            metadata.append(metadata_)
        with open(metadata_dir / f"{osplit}.tsv", "w") as f:
            for md in metadata:
                print("\t".join(md), file=f)

    completed_detector.touch()
    return target_dir


def prepare_fleurs(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    languages: Optional[Union[str, Sequence[str]]] = "all",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Prepares the manifest for all of the FLEURS languages requested.

    :param corpus_dir: Path to the root where the FLEURS data are stored.
    :type corpus_dir: Pathlike,
    :param output_dir: The directory where the .jsonl.gz manifests will be written.
    :type output_dir: Pathlike,
    :param langauges: str or str sequence specifying the languages to prepare.
        The str 'all' prepares all 102 languages.
    :return: The manifest
    :rtype: Dict[str, Dict[str, Union[RecordingSet, Supervisions]]]]
    """

    if isinstance(corpus_dir, str):
        corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(mode=511, parents=True, exist_ok=True)

    langs_list = DEFAULT_LANGUAGES
    if isinstance(languages, str) and languages != "all":
        langs_list = [languages]
    elif isinstance(languages, list) or isinstance(languages, tuple):
        if languages[0] != "all":
            langs_list = languages

    # Start buildings the recordings and supervisions
    manifests = {}
    for lang in langs_list:
        corpus_dir_lang = corpus_dir / f"{lang}"
        if not corpus_dir_lang.is_dir():
            logging.info(f"Skipping {lang}. No directory {corpus_dir_lang} found.")
            continue
        output_dir_lang = output_dir / f"{lang}"
        output_dir_lang.mkdir(mode=511, parents=True, exist_ok=True)
        manifests[lang] = prepare_single_fleurs_language(
            corpus_dir_lang,
            output_dir_lang,
            language=lang,
            num_jobs=num_jobs,
        )

    if output_dir is not None:
        for l in manifests:
            for dset in ("train", "dev", "test"):
                manifests[l][dset]["supervisions"].to_file(
                    output_dir / f"{l}" / f"fleurs-{l}_supervisions_{dset}.jsonl.gz"
                )
                manifests[l][dset]["recordings"].to_file(
                    output_dir / f"{l}" / f"fleurs-{l}_recordings_{dset}.jsonl.gz"
                )
    return manifests


def _make_recording(path):
    return Recording.from_file(path, recording_id=Path(path).stem)


def prepare_single_fleurs_language(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    language: str = "language",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares manifests using a single FLEURS language.

    :param corpus_dir: Path to the root where the FLEURS data are stored.
    :type corpus_dir: Pathlike,
    :param output_dir: The directory where the .jsonl.gz manifests will be written.
    :type output_dir: Pathlike,
    :param langauge: str specifying the language to prepare.

    :return: The manifest
    :rtype: Dict[str, Dict[str, Union[RecordingSet, Supervisions]]]]
    """

    if isinstance(corpus_dir, str):
        corpus_dir = Path(corpus_dir)

    recordings = {"train": [], "dev": [], "test": []}
    supervisions = {"train": [], "dev": [], "test": []}

    # First prepare the supervisions
    for dset in ("train", "dev", "test"):
        print(f"Preparing {dset} ...")
        prompt_ids = {}
        with open(
            corpus_dir.parents[0] / "metadata" / corpus_dir.stem / f"{dset}.tsv"
        ) as f:
            for l in f:
                vals = l.strip().split("\t")
                prompt_id, fname, raw_text, text, _, nsamples, gender = vals
                if prompt_id not in prompt_ids:
                    prompt_ids[prompt_id] = 0
                prompt_ids[prompt_id] += 1
                fname = Path(fname).stem
                supervisions[dset].append(
                    SupervisionSegment(
                        id=f"{prompt_id}_{prompt_ids[prompt_id]}_{fname}",
                        recording_id=fname,
                        start=0.0,
                        duration=round(int(nsamples) / 16000, 4),
                        channel=0,
                        text=text,
                        language=language,
                        speaker=f"{prompt_id}_{prompt_ids[prompt_id]}",
                        gender=gender,
                        custom={"raw_text": raw_text},
                    )
                )
    for dset in ("train", "dev", "test"):
        for reco in tqdm(
            parallel_map(
                _make_recording,
                (
                    corpus_dir / f"{dset}/{s.recording_id}.wav"
                    for s in supervisions[dset]
                ),
                num_jobs=num_jobs,
            ),
            desc=f"Making recordings from {language} {dset}",
        ):
            recordings[dset].append(reco)
    manifests = {}
    for dset in ("train", "dev", "test"):
        sups = SupervisionSet.from_segments(supervisions[dset])
        recos = RecordingSet.from_recordings(recordings[dset])
        recos, sups = fix_manifests(recos, sups)
        manifests[dset] = {"supervisions": sups, "recordings": recos}
    return manifests
