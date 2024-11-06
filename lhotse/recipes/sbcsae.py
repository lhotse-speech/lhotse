"""
This script downloads and prepares the data directory for the Santa Barbara
Corpus of Spoken American English.

The Santa Barbara Corpus of Spoken American English is based on a large body of
recordings of naturally occurring spoken interaction from all over the United
States. The Santa Barbara Corpus represents a wide variety of people of
different regional origins, ages, occupations, genders, and ethnic and social
backgrounds. The predominant form of language use represented is face-to-face
conversation, but the corpus also documents many other ways that that people use
language in their everyday lives: telephone conversations, card games, food
preparation, on-the-job talk, classroom lectures, sermons, story-telling, town
hall meetings, tour-guide spiels, and more.

The Santa Barbara Corpus was compiled by researchers in the Linguistics
Department of the University of California, Santa Barbara. The Director of the
Santa Barbara Corpus is John W. Du Bois, working with Associate Editors Wallace
L. Chafe and Sandra A. Thompson (all of UC Santa Barbara), and Charles Meyer
(UMass, Boston). For the publication of Parts 3 and 4, the authors are John W.
Du Bois and Robert Englebretson.

If you use the corpus or our data preparation scripts, please cite the following:
@misc{dubois_2005,
  author={Du Bois, John W. and Chafe, Wallace L. and Meyer, Charles and Thompson, Sandra A. and Englebretson, Robert and Martey, Nii},
  year={2000--2005},
  title={{S}anta {B}arbara corpus of spoken {A}merican {E}nglish, {P}arts 1--4},
  address={Philadelphia},
  organization={Linguistic Data Consortium},
}
@inproceedings{maciejewski24_interspeech,
  author={Matthew Maciejewski and Dominik Klement and Ruizhe Huang and Matthew Wiesner and Sanjeev Khudanpur},
  title={Evaluating the {Santa Barbara} Corpus: Challenges of the Breadth of Conversational Spoken Language},
  year=2024,
  booktitle={Proc. Interspeech 2024}
}
"""
import logging
import re
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
)
from lhotse.utils import (
    Pathlike,
    fastcopy,
    is_module_available,
    resumable_download,
    safe_extract,
)

SBCSAE_TAR_URL = "https://www.openslr.org/resources/155/SBCSAE.tar.gz"


lang_iterators = {
    "SBC004": iter(["Spanish"] * 17),
    "SBC006": iter(["French"] * 2),
    "SBC010": iter(["Spanish"]),
    "SBC012": iter(["Greek"] * 2),
    "SBC015": iter(["Spanish"] * 10),
    "SBC025": iter(["German"] * 2 + ["Latin"]),
    "SBC027": iter(["Spanish"] * 6 + ["French"] * 2),
    "SBC031": iter(["French"] * 2),
    "SBC033": iter(["French"]),
    "SBC034": iter(["French"] * 3),
    "SBC036": iter(["Spanish"] * 36),
    "SBC037": iter(["Spanish"] * 60),
    "SBC047": iter(["Spanish"]),
    "SBC057": iter(["Japanese"] * 62),
    "SBC058": iter(["Spanish"] + ["Italian"] * 2),
}


# These corrections to the participant metadata were needed to get geolocations
# from the geopy package.
annotation_corrections = {
    "metro St.L. IL": "Saint Louis MO",  # Use the MO side of the city
    "middle Wes MO": "Missouri",  # Just use the state location
    "S.E.Texas TX": "South East Texas",  # The geo package seems to parse this
    "South Alabama mostly AL": "Andalusia Alabama",  # Arbitrarily chosen nearby town
    "South FL": "South Bay Florida",  # Arbitrarily chosen nearby town
    "Walnut Cre CA": "Walnut Creek CA",  # Spelling error
    "San Leandr CA": "San Leandro CA",
    "Boston/Santa Fe MA/NM": "Boston/Santa Fe\tMA/NM",  # Handle this specially
    "Boston/New Mexico MA/NM": "Boston/Santa Fe\tMA/NM",
    "Millstad IL": "Millstadt IL",  # Spelling error
    "Cleveland/San Francisco OH/CA": "Cleveland/San Fransisco\tOH/CA",  # Handle specially
    "Jamesville WI": "Janesville WI",  # Spelling error
    "Falls Church/Albuquerque VA/NM": "Falls Church/Albuquerque\tVA/NM",  # Handle specially
    "Southern Florida": "South Bay Florida",  # Arbitarily chosen nearby town
    "Massachusetts MA": "Massachusetts",
    "New Zealand n/a": "New Zealand",
    "French n/a": "France",
}


bad_stereo = ["SBC020", "SBC021", "SBC027", "SBC028"]


class Dummy_Spk_Iterator:
    def __init__(self):
        self.ind = 213

    def next(self, spk="SBCXXX_X"):
        self.ind = self.ind + 1
        name = "_".join(spk.split("_")[1:])
        if name.startswith("X") or name.startswith("AUD"):
            name = "UNK"
        return f"{self.ind:04d}_{name}"


dummy_spk_iterator = Dummy_Spk_Iterator()


def download_sbcsae(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and untar the dataset.

    :param: target_dir: Pathlike, the path of the directory where the SBCSAE
        dataset will be downloaded.
    :param force_download: bool, if True, download the archive even if it already exists.
    :return: The path to the directory with the data.
    """
    target_dir = Path(target_dir)
    corpus_dir = target_dir / "SBCSAE"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    tar_path = target_dir / "SBCSAE.tar.gz"

    completed_detector = target_dir / ".sbcsae_completed"
    if completed_detector.is_file():
        logging.info(f"Skipping download because {completed_detector} exists.")
        return corpus_dir

    resumable_download(SBCSAE_TAR_URL, filename=tar_path, force_download=force_download)
    with tarfile.open(tar_path) as tar:
        safe_extract(tar, path=corpus_dir)
        completed_detector.touch()

    return corpus_dir


def prepare_sbcsae(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    geolocation: Optional[bool] = False,
    omit_realignments: Optional[bool] = False,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepares manifest for SBCSAE dataset.

    :param: corpus_dir: Path to the root where SBCSAE data was downloaded. It
        should be called SBCSAE. There is no consistent formatting between
        releases of the data. Check script comments for details if using an
        existing corpus download rather than Lhotse's download script.
    :param: output_dir: Root directory where .json manifests are stored.
    :param: geolocation: Include geographic coordinates of speakers' hometowns
        in the manifests.
    :param: omit_realignments: Only output original corpus segmentation.
    :return: The manifests.
    """
    # Resolve corpus_dir type
    if isinstance(corpus_dir, str):
        corpus_dir = Path(corpus_dir)

    # Resolve output_dir type
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    audio_dir = corpus_dir / "WAV"
    recordings = RecordingSet.from_recordings(
        Recording.from_file(p) for p in audio_dir.glob("*.wav")
    )
    if len(recordings) == 0:
        logging.warning(f"No .wav files found in {audio_dir}")

    doc_dir = corpus_dir / "docs"
    spk2gen_dict, spk2glob_dict = generate_speaker_map_dicts(doc_dir)

    spk_coords = {}
    if geolocation:
        spk_coords = generate_geolocations(corpus_dir, spk2glob_dict)

    supervisions = []
    trn_dir = corpus_dir / "TRN"
    for p in tqdm(
        list(trn_dir.glob("*.trn")), "Collecting and normalizing transcripts ..."
    ):
        for supervision in _filename_to_supervisions(p, spk2gen_dict, spk2glob_dict):
            supervisions.append(supervision)

    if len(supervisions) == 0:
        logging.warning(f"No supervisions found in {trn_dir}")

    supervisions_ = []
    for s in supervisions:
        if s.duration < 0.02:
            # Just pad with a minimum 0.02 duration
            s_reco = recordings[s.recording_id]
            new_start = max(0, s.start - 0.01)
            s_ = fastcopy(
                s,
                start=new_start,
                duration=min(new_start + 0.02, s_reco.duration),
            )
        else:
            s_ = s

        if s_.speaker in spk_coords:
            s_.custom = {
                "lat": spk_coords[s.speaker][0][0],
                "lon": spk_coords[s.speaker][0][1],
            }

        if (
            not isinstance(recordings[s.recording_id].channel_ids, list)
            or len(recordings[s.recording_id].channel_ids) < 2
            or s.recording_id in bad_stereo
        ):
            s_.channel = recordings[s.recording_id].channel_ids[0]
        supervisions_.append(s_)

    supervisions = SupervisionSet.from_segments(supervisions_)
    recordings, supervisions = fix_manifests(recordings, supervisions)

    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_file(output_dir / "sbcsae_recordings.jsonl.gz")
        supervisions.to_file(output_dir / "sbcsae_supervisions.jsonl.gz")

    manifests = {"recordings": recordings, "supervisions": supervisions}

    if not omit_realignments:
        asr_supervisions, diar_supervisions = apply_aligned_stms(
            list(recordings.ids), supervisions
        )
        _, asr_supervisions = fix_manifests(recordings, asr_supervisions)
        _, diar_supervisions = fix_manifests(recordings, diar_supervisions)

        asr_supervisions.to_file(
            output_dir / "sbcsae_supervisions_asr_aligned.jsonl.gz"
        )
        diar_supervisions.to_file(
            output_dir / "sbcsae_supervisions_diar_aligned.jsonl.gz"
        )

        manifests = {
            "asr_supervisions": asr_supervisions,
            "diar_supervisions": diar_supervisions,
            **manifests,
        }

    return manifests


def generate_geolocations(corpus: Path, spk2glob_dict: dict):
    if not is_module_available("geopy"):
        raise ImportError(
            "geopy package not found. Please install..." " (pip install geopy)"
        )
    else:
        from geopy import geocoders
        from geopy.geocoders import Nominatim

    speakers = corpus.rglob("docs/Part_*/speaker.tbl")
    # This geolocator object is repsonsible for generating a
    # latitiude and longitude from a textual description of a location, i.e.,
    # CHICAGO IL --> (41,-87)
    geolocator = Nominatim(user_agent="myapplication")
    spk_coords = {}
    for spk in tqdm(list(speakers), "Generating speaker geolocations..."):
        with open(spk) as f:
            for l in f:
                vals = l.strip().split(",")
                if len(vals) < 5:
                    continue
                # Check non-empty
                empty_hometown = vals[4] in ("", "?")
                empty_state = vals[5] in ("", "?")
                if empty_hometown and not empty_state:
                    loc = vals[5] + ", United States"
                elif not empty_hometown:
                    orig_loc = vals[4] + " " + vals[5]
                    loc = annotation_corrections.get(orig_loc, orig_loc)
                else:
                    continue
                if "/" in loc:
                    try:
                        hometowns, states = loc.split("\t", 1)
                        hometowns = hometowns.split("/")
                        states = states.split("/")
                        coords = []
                        for h, s in zip(hometowns, states):
                            coords.append(
                                geolocator.geocode(f"{h} {s}", timeout=None)[1]
                            )
                    except ValueError:
                        states, country = loc.split(",", 1)
                        coords = []
                        for s in states.split("/"):
                            coords.append(
                                geolocator.geocode(f"{s}, {country}", timeout=None)[1]
                            )
                else:
                    coords = [geolocator.geocode(loc, timeout=None)[1]]
                spk_coords[vals[0]] = coords
    spknum2spk_name = {n.split("_")[0]: n for s, n in spk2glob_dict.items()}
    spk_coords_ = {}
    for s in spk_coords:
        if s in spknum2spk_name:
            spk_coords_[spknum2spk_name[s]] = spk_coords[s]
    return spk_coords_


def generate_speaker_map_dicts(doc_dir: Path):
    spk2gen_dict = dict()
    spk2glob_dict = dict()

    spk_num_to_reco_ids = dict()
    for part in ["Part_1", "Part_2", "Part_4"]:
        filename = doc_dir / part / "segment.tbl"
        for line in filename.read_text().split("\n"):
            if "speaker:" in line:
                line = line.replace(" 0", "\t0")
                reco_id = re.sub(r"sbc0?([0-9]{3})\s.*", r"SBC\1", line)
                spk_num = line.split("\t")[-1][:4]
                if spk_num not in spk_num_to_reco_ids:
                    spk_num_to_reco_ids[spk_num] = []
                if reco_id not in spk_num_to_reco_ids[spk_num]:
                    spk_num_to_reco_ids[spk_num].append(reco_id)

    for part in ["Part_1", "Part_2", "Part_4"]:
        filename = doc_dir / part / "speaker.tbl"
        for line in filename.read_text().split("\n"):
            if "," not in line:
                continue
            line = line.replace("0163,Dan,m", "0166,Dan,M")
            spk_num, name, gen = line.split(",")[:3]
            name = (
                name.replace(" (extra-corpus)", "").upper().split(" ")[-1].split("/")[0]
            )
            gen = gen.upper()
            if not gen:
                gen = None

            if spk_num in ["0069", "0091", "0092", "0097"]:
                continue
            for reco in spk_num_to_reco_ids[spk_num]:
                spk2gen_dict[reco + "_" + name] = gen
                spk2glob_dict[reco + "_" + name] = spk_num + "_" + name

    for part in ["Part_3"]:
        seg_list = []
        filename = doc_dir / part / "segment.tbl"
        for line in filename.read_text().split("\n"):
            if "speaker:" in line:
                reco_id = re.sub(r"sbc0?([0-9]{3})\s.*", r"SBC\1", line)
                name = line.split(" ")[-1].upper().split("/")[0]
                seg_list.append([name, reco_id])

        spk_list = []
        filename = doc_dir / part / "speaker.tbl"
        for line in filename.read_text().split("\n"):
            if "," not in line:
                continue
            spk_num, name, gen = line.split(",")[:3]
            name = name.upper().split("/")[0]
            spk_list.append([name, spk_num, gen])

        for seg_info, spk_info in zip(seg_list, spk_list):
            assert seg_info[0] == spk_info[0], f"{seg_info[0]} != {spk_info[0]}"
            spk2gen_dict[seg_info[1] + "_" + seg_info[0]] = spk_info[2]
            spk2glob_dict[seg_info[1] + "_" + seg_info[0]] = (
                spk_info[1] + "_" + spk_info[0]
            )

    for spk_key in [
        "SBC006_ALL",
        "SBC008_ALL",
        "SBC012_MANY",
        "SBC020_AUD",
        "SBC021_MANY",
        "SBC023_MANY",
        "SBC025_AUD",
        "SBC026_AUD",
        "SBC027_MANY",
        "SBC027_AUD",
        "SBC028_BOTH",
        "SBC030_AUD",
        "SBC038_AUD",
        "SBC053_RADIO",
        "SBC054_AUD",
        "SBC054_MANY",
        "SBC055_AUD",
    ]:
        spk2gen_dict[spk_key] = None
        spk2glob_dict[spk_key] = spk_key

    return spk2gen_dict, spk2glob_dict


def _filename_to_supervisions(filename: Path, spk2gen_dict: dict, spk2glob_dict: dict):
    reco_id = filename.stem.split(".")[0]
    lines = filename.read_text(encoding="latin1")
    supervisions = []

    #### Transcript fix
    lines = lines.replace("\x92", "'")
    lines = lines.replace("\u007f", "")
    lines = lines.replace("\u0000", "c")

    if reco_id == "SBC002":
        lines = lines.replace("(TSK ", "(TSK) ")
    elif reco_id == "SBC004":
        lines = lines.replace("KATE", "KATHY")
        lines = lines.replace("sen~orita", "se\xf1orita")
    elif reco_id == "SBC005":
        lines = lines.replace("good_/god/", "good")
        lines = lines.replace("(H)@>", "(H) @>")
        lines = lines.replace("[@@ <@Mm@>]", "[@@ <@ Mm @>]")
    elif reco_id == "SBC006":
        lines = lines.replace("/pub/", "pub")
        lines = lines.replace("<WH@@@@ (H) @@WH>", "<WH @@@@ (H) @@ WH>")
        lines = lines.replace("[2(H)2]1", "[2(H)2]")
    elif reco_id == "SBC007":
        lines = lines.replace(
            "\\000000000 000000000 MARY: 1182.90 1186.92\t        ",
            "\n1182.90 1186.92\tMARY:   ",
        )
        lines = lines.replace("(YAWN0", "(YAWN)")
    elif reco_id == "SBC008":
        lines = lines.replace("[<X Go]=dX>", "[<X Go]=d X>")
    elif reco_id == "SBC010":
        lines = lines.replace("366.87 366.87", "366.16 366.87")
    elif reco_id == "SBC012":
        lines = lines.replace(
            "\n".join(["807.02 807.92\tFRANK:  \t.. Mhm."] * 2),
            "807.02 807.92\tFRANK:  \t.. Mhm.",
        )
        lines = lines.replace("MONTOYA", "MONTOYO")
    elif reco_id == "SBC013":
        lines = lines.replace("[8<@She8]", "[8<@ She8]")
        lines = lines.replace("[2(H) cou_ couch@>2]", "[2(H) cou_ couch @>2]")
        lines = lines.replace("[4<@No=4]", "[4<@ No=4]")
        lines = lines.replace("VOX2]", "VOX>2]")
    elif reco_id == "SBC014":
        lines = lines.replace("\\000000000 000000000 ", "\n")
        lines = lines.replace("<@he thought", "<@ he thought")
    elif reco_id == "SBC015":
        lines = lines.replace(
            "243.055\t244.080\tKEN:\t(H)] the little,",
            "243.465\t244.670\tKEN:\t(H)] the little,",
        )
        lines = lines.replace("\u0000urch things.", "church things.")
        lines = lines.replace("2(H]=2", "2(H)=2")
        lines = lines.replace(" 0.000000e+00", "e")
        lines = lines.replace("0m=,", "um=,")
        lines = lines.replace("0eople", "people")
        lines = lines.replace("0id", "did")
        lines = lines.replace("X 0ne %tho", "X uh line %tho")
        lines = lines.replace("and 0t [was]", "and it [was]")
        lines = lines.replace("0t was like", "it was like")
    elif reco_id == "SBC016":
        lines = lines.replace("/sed ai/", "sed ai")
    elif reco_id == "SBC017":
        lines = lines.replace("a\tand names the] na=me,", "and names the] na=me,")
        lines = lines.replace(" 0.000000e+00", "e")
        lines = lines.replace("[2I mean2", "[2I mean2]")
        lines = lines.replace("no2.", "no.")
        lines = lines.replace("0rganisms", "organisms")
        lines = lines.replace("0ttle", "little")
    elif reco_id == "SBC018":
        lines = lines.replace("0f", "if")
        lines = lines.replace(
            "129.916\t130.324\tLINDSEY:\tYeah.\n129.915\t130.325\t\t[Mhm.]\n",
            "129.915\t130.325\tLINDSEY:\t[Mhm.] Yeah.\n",
        )
    elif reco_id == "SBC019":
        lines = lines.replace("cello_(/cheller/)", "cheller")
        lines = lines.replace("(sigh)", "(SIGH)")
        lines = lines.replace("<F<VOX> Mo=m", "<F<VOX Mo=m")
        lines = lines.replace("@@[3@=3", "@@[3@=3]")
        lines = lines.replace("[#5Jason", "[5#Jason")
        lines = lines.replace("[20nh2]", "[2Unh2]")
        lines = lines.replace("Draw 0n", "Draw on")
        lines = lines.replace("0oes", "Does")
        lines = lines.replace("0=kay", "O=kay")
    elif reco_id == "SBC020":
        lines = lines.replace("(COUGh)", "(COUGH)")
        lines = lines.replace("(throat)", "(THROAT)")
        lines = lines.replace("S-  0emon", "S- demon")
        lines = lines.replace(" 0.000000E+00", "E")
        lines = lines.replace("now 0m", "now um")
        lines = lines.replace("uh  0s", "uh is")
        lines = lines.replace("but  0n", "but uh in")
        lines = lines.replace("i- % 0t's", "i- uh it's")
        lines = lines.replace("0retty", "pretty")
        lines = lines.replace("AUD:\tY", "X:\tY")
    elif reco_id == "SBC022":
        lines = lines.replace("(h)", "(H)")
        lines = lines.replace("0.000000e+00", "e-")
        lines = lines.replace("0ttle", "little")
        lines = lines.replace("0ne thing", "uh one thing")
    elif reco_id == "SBC023":
        lines = lines.replace("JANICD", "JANICE")
        lines = lines.replace("NORA?", "NORA")
        lines = lines.replace("SUE?", "SUE")
        lines = lines.replace("(throat)", "(THROAT)")
        lines = lines.replace("2(SNIFF2", "2(SNIFF)2")
        lines = lines.replace("[<Xbu=tX>]", "[<X bu=t X>]")
        lines = lines.replace("<or did it", "<Q or did it")
        lines = lines.replace("x>5]", "X>5]")
        lines = lines.replace("0nly", "uh only")
        lines = lines.replace("[50r5]", "[5Or5]")
    elif reco_id == "SBC024":
        lines = lines.replace(" >ENV: ", ">ENV:\t")
        lines = lines.replace(" 0.000000irst", "First")
        lines = lines.replace("2[cause", "[2cause")
        lines = lines.replace(" 0oes", "does")
        lines = lines.replace("0id]", "did]")
    elif reco_id == "SBC025":
        lines = lines.replace("<ot,", "<% not,")
        lines = lines.replace(" 0.000000e+00", "e")
        lines = lines.replace("0mself", "himself")
    elif reco_id == "SBC026":
        lines = lines.replace("does_(/uz/)", "does")
        lines = lines.replace(" 0.000000e+00", "e")
        lines = lines.replace("0ngoing", "ongoing")
        lines = lines.replace("AUD:\t<X", "X_2:\t<X")
    elif reco_id == "SBC027":
        lines = lines.replace("142.870\t144.790 :", "142.870\t144.790")
        lines = lines.replace("451.510\t452.130 :", "451.510\t452.130")
        lines = lines.replace(" 0oing", "doing")
        lines = lines.replace("AUD:\t.. [We", "X:\t.. [We")
        lines = lines.replace("AUD:\t... Liquid", "X_1:\t... Liquid")
        lines = lines.replace("AUD:\tAdd", "X_2:\tAdd")
        lines = lines.replace("AUD:\t     [", "X_3:\t     [")
        lines = lines.replace("AUD1:\t... One", "X_4:\t... One")
        lines = lines.replace("AUD2:\t[One", "X_5:\t[One")
        lines = lines.replace("AUD:\t...X [X", "X_6:\tX [X")
        lines = lines.replace("AUD1:\tEight", "X_7:\tEight")
        lines = lines.replace("AUD2:\t... [@", "AUD:\t... [@")
        lines = lines.replace("AUD3:\t    [Four", "X_8:\t    [Four")
        lines = lines.replace("AUD:\t... Seven", "X_9:\t... Seven")
        lines = lines.replace("AUD1:\t.. <L2", "X_10:\t.. <L2")
        lines = lines.replace("AUD2:\t        [", "X_11:\t       [")
        lines = lines.replace("AUD:\t... <L2", "X_12:\t... <L2")
        lines = lines.replace("AUD1:\t... [E", "X_13:\t... [E")
        lines = lines.replace("AUD2:\t    [<L2", "X_14:\t    [<L2")
        lines = lines.replace("AUD1:\t     ", "X_15:\t     ")
        lines = lines.replace("AUD2:\t... There", "X_16:\t... There")
        lines = lines.replace("AUD1:\t[Pull", "X_17:\t[Pull")
        lines = lines.replace("AUD2:\tYou", "X_18:\tYou")
        lines = lines.replace("AUD:\t[<X", "X_19:\t[<X")
        lines = lines.replace("AUD:\t... Solid", "X_20:\t... Solid")
        lines = lines.replace("AUD:\t.. Hydrogen", "X_21:\t.. Hydrogen")
        lines = lines.replace("AUD:\t.. Oxygen", "X_22:\t.. Oxygen")
        lines = lines.replace("AUD:\t.. [<", "X_23:\t.. [<")
        lines = lines.replace("AUD:\t       ", "X_24:\t       ")
        lines = lines.replace("AUD:\tThey're", "X_25:\tThey're")
        lines = lines.replace("AUD:\t XXX", "X_26:\t XXX")
        lines = lines.replace("AUD:\t... No", "X_27:\t... No")
        lines = lines.replace("AUD:\t<X", "X_28:\t<X")
        lines = lines.replace("AUD:\tThrow", "X_29:\tThrow")
        lines = lines.replace("AUD:\tHotter", "X_30:\tHotter")
        lines = lines.replace("AUD:\t.. Liquid", "X_31:\t.. Liquid")
        lines = lines.replace("AUD:\t Did", "X_32:\t Did")
        lines = lines.replace("AUD:\tX", "X_33:\tX")
    elif reco_id == "SBC028":
        lines = lines.replace(
            "482.610\t484.010\tJILL_S: ", "482.610\t484.010\tJILL_S:\t"
        )
        lines = lines.replace("<@Oh[2=@>", "<@ Oh[2= @>")
        lines = lines.replace(" 0.000000", " ")
        lines = lines.replace("i 0f", "i- if")
        lines = lines.replace("0f we", "if we")
        lines = lines.replace("th- 0t's", "th- that's")
        lines = lines.replace("0t's", "it's")
        lines = lines.replace("0f", "if")
    elif reco_id == "SBC029":
        lines = lines.replace("96.230\t98.240\t>ENV: ", "96.230\t98.240\t>ENV:\t")
        lines = lines.replace("(H )", "(H)")
        lines = lines.replace("<0h=,", "<% Oh=,")
        lines = lines.replace("knowX>]", "know X>]")
        lines = lines.replace("0verheating", "overheating")
    elif reco_id == "SBC030":
        lines = lines.replace("DANNY", "BRADLEY")
        lines = lines.replace("AUD:\tYes", "X:\tYes")
    elif reco_id == "SBC034":
        lines = lines.replace("13548.02 ", "1354.802")
    elif reco_id == "SBC036":
        lines = lines.replace(
            "1558.463\t1558.906\t\t[thought he was,",
            "1558.906\t1558.923\t\t[thought he was,",
        )
    elif reco_id == "SBC038":
        lines = lines.replace("AUD:\t... What's", "X_2:\t... What's")
        lines = lines.replace("AUD:\t... U", "X_3:\t... U")
        lines = lines.replace("AUD:\t... How far", "X_2:\t... How far")
        lines = lines.replace("AUD:\t<X Quite", "X_4:\t<X Quite")
        lines = lines.replace("AUD:\tYeah", "X_5:\tYeah")
        lines = lines.replace("AUD:\tAbout", "X_6:\tAbout")
        lines = lines.replace("AUD:\t... That", "X_7:\t... That")
        lines = lines.replace("AUD:\t.. <X Oh", "X_8:\t.. <X Oh")
        lines = lines.replace("AUD:\t... How long", "X_3:\t... How long")
        lines = lines.replace("AUD:\t<X @", "X_3:\t<X @")
        lines = lines.replace("AUD:\tEach", "X_2:\tEach")
        lines = lines.replace("AUD:\tThe water", "X_2:\tThe water")
        lines = lines.replace("AUD:\t[Right", "X_9:\t[Right")
        lines = lines.replace("AUD:\t... It's", "X_9:\t... It's")
        lines = lines.replace("AUD:\t[Perp", "X_9:\t[Perp")
        lines = lines.replace("AUD:\t[2perp", "X_9:\t[2perp")
        lines = lines.replace("AUD:\t[3The", "X_9:\t[3The")
        lines = lines.replace("AUD:\t[4Right", "X_9:\t[4Right")
        lines = lines.replace("AUD:\tOh yeah", "X_9:\tOh yeah")
        lines = lines.replace("AUD:\t[6Now", "X_9:\t[6Now")
        lines = lines.replace("AUD:\twith the", "X_9:\twith the")
        lines = lines.replace("AUD:\t[That-", "X_9:\t[That-")
        lines = lines.replace("AUD:\t[Spinning", "X_9:\t[Spinning")
        lines = lines.replace("AUD:\t[2Yeah", "X_9:\t[2Yeah")
        lines = lines.replace("AUD:\t[3X", "X_9:\t[3X")
        lines = lines.replace("AUD:\t[4<X", "X_9:\t[4<X")
        lines = lines.replace("AUD:\tAnd that's", "X_9:\tAnd that's")
        lines = lines.replace("AUD:\t[So", "X_9:\t[So")
        lines = lines.replace("AUD:\t[2that's", "X_9:\t[2that's")
        lines = lines.replace("AUD:\tthat's3", "X_9:\tthat's3")
        lines = lines.replace("AUD:\tWe", "X_9:\tWe")
        lines = lines.replace("AUD:\t.. All", "X_9:\t.. All")
        lines = lines.replace("AUD:\t.. What's", "X_10:\t.. What's")
        lines = lines.replace("AUD:\t... Are", "X_3:\t... Are")
        lines = lines.replace("AUD:\tThe rest", "X_11:\tThe rest")
        lines = lines.replace("AUD:\t... Y'all", "X_12:\t... Y'all")
        lines = lines.replace("AUD:\t... Is", "X_13:\t... Is")
        lines = lines.replace("AUD:\t[<X", "X_13:\t[<X")
        lines = lines.replace("AUD:\t[Yeah", "X_13:\t[Yeah")
        lines = lines.replace("AUD:\t... What are", "X_13:\t... What are")
        lines = lines.replace("AUD_2", "AUD")
        lines = lines.replace("AUD:\t[What are", "X_13:\t[What are")
        lines = lines.replace("AUD:\t... Say", "X_14:\t... Say")
        lines = lines.replace("AUD:\t[what's", "X_14:\t[what's")
        lines = lines.replace("AUD:\t.. Hmm", "X_14:\t.. Hmm")
        lines = lines.replace("AUD:\t[3When", "X_14:\t[3When")
        lines = lines.replace("AUD:\t[It's", "X_15:\t[It's")
        lines = lines.replace("AUD:\t... Have", "X_16:\t... Have")
        lines = lines.replace("AUD:\tThanks", "X_17:\tThanks")
        lines = lines.replace("AUD:\t... Wow", "X_13:\t... Wow")
    elif reco_id == "SBC040":
        lines = lines.replace("AUD:\t... What's", "X:\t... What's")
        lines = lines.replace("AUD:\t... He", "X_2:\t... He")
        lines = lines.replace("AUD:\t[What", "X_3:\t[What")
        lines = lines.replace("AUD:\t.. Isn't", "X_4:\t.. Isn't")
        lines = lines.replace("AUD:\tClaiborne", "X_4:\tClaiborne")
        lines = lines.replace("AUD:\t... How", "X_4:\t... How")
        lines = lines.replace("AUD:\t.. How", "X_4:\t.. How")
        lines = lines.replace("AUD:\t.. The", "X_5:\t.. The")
        lines = lines.replace("AUD:\t... Yes", "X_6:\t... Yes")
    elif reco_id == "SBC043":
        lines = lines.replace("< HI any nights HI>", "<HI any nights HI>")
        lines = lines.replace("ANNETTE", "ANETTE")
    elif reco_id == "SBC048":
        lines = lines.replace("<@in San[2ta", "<@ in San[2ta")
    elif reco_id == "SBC052":
        lines = lines.replace("~Janine\t said", "~Janine said")
    elif reco_id == "SBC054":
        lines = lines.replace("<VOX Ugh VOX >", "<VOX Ugh VOX>")
        lines = lines.replace("AUD:\tX", "X:\tX")
        lines = lines.replace("AUD:\t<X", "X_2:\t<X")
        lines = lines.replace("AUD_2:\t[Tha-]", "X_3:\t[Tha-]")
        lines = lines.replace("AUD_3:\t[Tha-]", "X_4:\t[Tha-]")
        lines = lines.replace("AUD:\t[@rhino", "X_5:\t[@rhino")
        lines = lines.replace("AUD_2", "AUD")
    elif reco_id == "SBC055":
        lines = lines.replace("in spite ..\tof having", "in spite .. of having")
        lines = lines.replace("AUD:\t... Beatrice", "X:\t... Beatrice")
        lines = lines.replace("AUD:\tHow was", "X_2:\tHow was")
        lines = lines.replace("AUD:\tCan", "X_3:\tCan")
        lines = lines.replace("AUD_2:", "X_4:")
    elif reco_id == "SBC056":
        lines = lines.replace("@@@2]\t[3@@@@3]", "@@@2] [3@@@@3]")
        lines = lines.replace("(sniff)", "(SNIFF)")
    elif reco_id == "SBC057":
        lines = lines.replace("Hane-makikomi", "<L2 Hane-makikomi L2>")
        lines = lines.replace("sensei", "<L2 sensei L2>")
        lines = lines.replace("ippon", "Ippon")
        lines = lines.replace("Ippon", "<L2 Ippon L2>")
        lines = re.sub(r"gi([^a-z])", r"<L2 gi L2>\1", lines)
        lines = re.sub(r"Makikomi([^-])", r"<L2 Makikomi L2>\1", lines)
        lines = lines.replace("Hane-goshi", "<L2 Hane-goshi L2>")
        lines = lines.replace("Sode-makikomi", "<L2 Sode-makikomi L2>")
        lines = lines.replace("shiai", "<L2 shiai L2>")
        lines = lines.replace("randori", "<L2 randori L2>")
        lines = re.sub(r"Sode([^-])", r"<L2 Sode L2>\1", lines)
        lines = lines.replace("Ukemi", "<L2 Ukemi L2>")
        lines = lines.replace("Ha-jime", "<L2 Ha-jime L2>")
        lines = lines.replace("Ude-garami", "<L2 Ude-garami L2>")
        lines = lines.replace("Hane-uchi-mata", "<L2 Hane-uchi-mata L2>")
        lines = lines.replace("Uchi-<X mother X>", "Uchi-mata")
        lines = lines.replace("Uchi-mata", "<L2 Uchi-mata L2>")
        lines = lines.replace("Hande-maki- <L2 ", "<L2 Hande-maki- ")
        lines = re.sub(r"Hane([^-])", r"<L2 Hane L2>\1", lines)
        lines = lines.replace("%Sode-maki[komi]", "<L2 %Sode-maki[komi] L2>")
        lines = lines.replace("Tsuri-komi", "<L2 Tsuri-komi L2>")
        lines = lines.replace("Uchi-komi", "<L2 Uchi-komi L2>")
        lines = lines.replace("O-uchi", "<L2 O-uchi L2>")
        lines = lines.replace("Goshi", "<L2 Goshi L2>")
        lines = lines.replace("Uchi]-mata", "<L2 Uchi]-mata L2>")
        lines = lines.replace("Komi", "<L2 Komi L2>")
        lines = lines.replace("Tani-otoshi", "<L2 Tani-otoshi L2>")
        lines = lines.replace("Hane-maki][2komi=", "<L2 Hane-maki][2komi= L2>")
        lines = lines.replace("Makikomi-waza", "<L2 Makikomi-waza L2>")
        lines = lines.replace("Seoi", "<L2 Seoi L2>")
        lines = lines.replace("uke", "<L2 uke L2>")
    elif reco_id == "SBC059":
        lines = lines.replace("[<F 3And you", "<F [3And you")
        lines = lines.replace("hour[6=6 F>]", "hour[6=6] F>")

    spk_buffer = ""
    lang_buffer = "English"
    for line in lines.split("\n"):
        #### Transcript fixes
        if line == "77.200\t77.540 :\t(H)":
            continue
        if line.startswith("000000000 000000000 ") or line.startswith("0.00 0.00"):
            continue
        if line.startswith("\t"):
            line.lstrip("\t")
        if "and in his pamphlet the Liber Arbetrio" in line:
            continue

        line = line.strip()
        line = re.sub(r" +", " ", line)
        line = re.sub(r"\t+", "\t", line)
        fields = line.strip().split("\t")
        if len(fields) == 4:
            spk_field, raw_trans = fields[2:]
            start, end = [float(time.rstrip()) for time in fields[:2]]
        elif len(fields) == 3:
            if len(fields[0].rstrip().split(" ")) > 1:
                spk_field, raw_trans = fields[1:]
                start, end = [float(time) for time in fields[0].split(" ")[:2]]
                raw_trans = fields[-1]
            else:
                start, end = [float(time.rstrip()) for time in fields[:2]]
                spk_field_candidate = fields[2].split(" ")[0]
                if re.fullmatch(r"[A-Z]+:", spk_field_candidate):
                    spk_field = spk_field_candidate
                    raw_trans = " ".join(fields[2].split(" ")[1:])
                else:
                    spk_field = ""
                    raw_trans = fields[2]
        elif len(fields) == 2:
            timesish = fields[0].rstrip().split(" ")
            if len(timesish) == 1:
                continue
            start, end = [float(time) for time in timesish[:2]]
            if len(timesish) > 2:
                spk_field = timesish[2]
                raw_trans = fields[1]
            else:
                spk_field_candidate = fields[1].split(" ")[0]
                if re.fullmatch(r"[A-Z]+:", spk_field_candidate):
                    spk_field = spk_field_candidate
                    raw_trans = " ".join(fields[1].split(" ")[1:])
                else:
                    spk_field = ""
                    raw_trans = fields[1]
        else:
            split = line.split(" ")
            if re.fullmatch(r"[0-9]+\.[0-9]+", split[0]) and re.fullmatch(
                r"[0-9]+\.[0-9]+", split[1]
            ):
                start, end = [float(time.rstrip()) for time in split[:2]]
                if re.fullmatch(r"[A-Z]+:", split[2]):
                    spk_field = split[2]
                    raw_trans = " ".join(split[3:])
                else:
                    spk_field = ""
                    raw_trans = " ".join(split[2:])
            else:
                continue

        #### Transcript fixes
        if raw_trans == "[2<L2 Zocalo.":
            raw_trans = "[2<L2 Zocalo L2>2]."
        elif raw_trans == "[You're <L2 outre mer L2].":
            raw_trans = "[You're <L2 outre mer L2>]."

        if " $ " in raw_trans:
            continue

        spk_field = spk_field.strip().rstrip(":").rstrip().upper()
        if spk_field in [">ENV", "ENV", ">MAC", ">DOG", ">HORSE", ">CAT", ">BABY"]:
            continue
        elif spk_field == "#READ":
            spk_field = "WALT"

        if spk_field:
            spk_field = re.sub(r"^[^A-Z]", "", spk_field)
            spk_buffer = spk_field

        utt_id = f"{reco_id}_{int(start*1000):07}_{int(end*1000):07}_{spk_buffer}"

        text, lang_tag = _parse_raw_transcript(raw_trans)

        if "l" in lang_tag:
            for _ in range(lang_tag.count("l")):
                new_lang = next(lang_iterators[reco_id])
            if "c" in lang_tag:
                lang_buffer = f"English-{new_lang}"
            else:
                lang_buffer = new_lang
        elif "c" in lang_tag:
            lang_buffer = f"English-{lang_buffer.split('-')[-1]}"

        spk_key = reco_id + "_" + spk_buffer
        if spk_key not in spk2glob_dict and reco_id != "SBC021":
            spk2gen_dict[spk_key] = None
            spk2glob_dict[spk_key] = dummy_spk_iterator.next(spk_key)

        if spk_key in spk2glob_dict:
            speaker = spk2glob_dict[spk_key]
            gender = spk2gen_dict[spk_key]
        else:
            speaker = dummy_spk_iterator.next(spk_key)
            gender = None

        if re.search(r"[A-Za-z]", text):
            supervisions.append(
                SupervisionSegment(
                    id=utt_id,
                    recording_id=reco_id,
                    start=start,
                    duration=end - start,
                    channel=[0, 1],
                    text=text,
                    language=lang_buffer,
                    speaker=speaker,
                    gender=gender,
                )
            )

        if lang_tag:
            if lang_tag[-1] == "r":
                lang_buffer = "English"
            if lang_tag[-1] == "l":
                lang_buffer = lang_buffer.split("-")[-1]

    return supervisions


def _parse_raw_transcript(transcript: str):

    transcript = transcript.replace("0h", "oh")
    transcript = transcript.replace("s@so", "s- so")
    transcript = transcript.replace("la@ter", "later")
    transcript = transcript.replace("you@.", "you @.")
    transcript = transcript.replace("[N=]", "N")
    transcript = transcript.replace("[2C2]=", "C")
    transcript = transcript.replace("[MM=]", "MM")
    transcript = transcript.replace("[I=]", "I")

    transcript = transcript.replace("(YELL)", "<yell>")

    transcript = transcript.replace("_", "-")

    transcript = transcript.replace("=", "")
    transcript = transcript.replace("%", "")

    # Process overlapped UNKs before they get removed by the following step
    transcript = re.sub(r"\[([2-9]?)([A-Z])+\1\]", r"\2", transcript)

    # Paired parenthetical/bracket annotation remover
    paren_matches = re.findall(r"\([^a-z@ ]*\)", transcript)
    for paren_match in paren_matches:
        transcript = transcript.replace(
            paren_match, re.sub(r"[^\[\]]", "", paren_match)
        )
    brack_matches = re.findall(r"\[[^a-z@ ]+\]", transcript)
    for brack_match in brack_matches:
        transcript = transcript.replace(
            brack_match, re.sub(r"[^\(\)]", "", brack_match)
        )

    transcript = re.sub(r"<<[^a-z@ ]+>>", "", transcript)
    transcript = re.sub(r"<<[^a-z@ ]+", "", transcript)
    transcript = re.sub(r"[^a-z@ ]+>>", "", transcript)

    transcript = re.sub(r"<[^a-z@ ]+>", "", transcript)
    transcript = re.sub(r"<[^a-z2 ]*[^2 ]([ <])", r"\1", transcript)
    transcript = re.sub(r"([ >])[^a-z2 ]*[^a-z 2]>", r"\1", transcript)

    transcript = re.sub(r"\[[2-9]?", "", transcript)
    transcript = re.sub(r"[2-9]?\]", "", transcript)

    transcript = transcript.replace("(Hx)", " ")
    transcript = transcript.replace("(hx)", " ")
    transcript = transcript.replace("(@Hx)", "@")

    transcript = transcript.replace("(COUGH COUGH)", " ")
    transcript = transcript.replace("(SNIFF", "")

    transcript = transcript.replace("(", "")
    transcript = transcript.replace(")", "")

    transcript = transcript.replace("< ", " ")
    transcript = transcript.replace(" >", " ")

    transcript = re.sub(r"[^A-Za-z-]-+", "", transcript)
    transcript = re.sub(r"\.\.+", "", transcript)

    transcript = transcript.replace("+", "")
    transcript = transcript.replace("&", "")
    transcript = transcript.replace("#", "")
    transcript = transcript.replace("*", "")

    transcript = re.sub(r"!([A-Za-z])", r"\1", transcript)

    # Deal with extra white space
    transcript = re.sub(r" +", " ", transcript)

    # Merge X's
    transcript = re.sub(r"X+", "X", transcript)

    # Parse laughter
    transcript = transcript.replace("on@,", "on @,")
    transcript = re.sub(r"([a-z-])@([a-z])", r"\1\2", transcript)
    transcript = re.sub(r"@+", "@", transcript)
    transcript = re.sub(r"(^| )@([^ ])", r" @ \2", transcript)
    transcript = re.sub(r"([^ ])@( |$)", r"\1 @ ", transcript)
    transcript = transcript.replace("@ @", "@").replace("@ @", "@")

    transcript = re.sub(r"(^| )X([ ,.?']|$)", r"\1<UNK>\2", transcript)
    transcript = re.sub(r"(^| )X([ ,.?']|$)", r"\1<UNK>\2", transcript)
    transcript = re.sub(r"X-($| )", r"<UNK>\1", transcript)

    transcript = re.sub(r"^ ", "", transcript)
    transcript = re.sub(r" $", "", transcript)

    transcript = transcript.replace(" .", ".")
    transcript = transcript.replace(" ,", ",")
    transcript = transcript.replace(" ?", "?")

    transcript = re.sub(r"^\. ", "", transcript)
    transcript = re.sub(r"^\.$", "", transcript)

    if (
        len(transcript.split("<L2")) > 1
        and re.search(r"[A-Za-z]", transcript.split("<L2")[0])
    ) or (
        len(transcript.split("L2>")) > 1
        and re.search(r"[A-Za-z]", transcript.split("L2>")[-1])
    ):
        lang_tag = "c"
    else:
        lang_tag = ""

    transcript = transcript.replace("@", "<LAUGH>")
    transcript = transcript.replace("<yell>", "<YELL>")

    if "L2" in transcript:
        lang_tag = lang_tag + re.sub(
            r"(<L2|L2>)(?!.*(<L2|L2>)).*$",
            r"\1",
            re.sub(r".*?(<L2|L2>)", r"\1", transcript),
        )
        lang_tag = lang_tag.replace("<L2", "l").replace("L2>", "r")

    # We choose to leave the language tags in, but uncommenting this would remove them.
    #    transcript = transcript.replace("<L2 ", "")
    #    transcript = transcript.replace(" L2>", "")

    return transcript, lang_tag


@dataclass
class StmSegment:
    recording_id: str
    speaker: str
    start: float
    end: float
    text: str
    channel: str = "1"


def parse_stm_file(data: str) -> List[StmSegment]:
    lines = data.split("\n")
    stm_segments = []

    for line in lines:
        if not line:
            continue

        fields = line.strip().split()
        reco_id, channel, speaker = fields[:3]
        start, end = [float(time) for time in fields[3:5]]
        text = " ".join(fields[5:])

        stm_segments.append(
            StmSegment(
                recording_id=reco_id,
                speaker=speaker,
                start=start,
                end=end,
                text=text,
                channel=channel,
            )
        )

    return stm_segments


def retrieve_stm_file(url) -> List[StmSegment]:
    import urllib.request

    response = urllib.request.urlopen(url)
    data = response.read().decode("utf-8")

    return parse_stm_file(data)


def norm_txt(text: str):
    text = text.strip()
    text = text.lower()
    return text


def compute_iou(seg1: SupervisionSegment, seg2: StmSegment) -> float:
    start = max(seg1.start, seg2.start)
    end = min(seg1.end, seg2.end)

    intersection = max(0.0, end - start)
    union = (seg1.end - seg1.start) + (seg2.end - seg2.start) - intersection

    return intersection / union


def apply_stm(
    recording_ids: List[str],
    supervisions: SupervisionSet,
    aligned_stm_segs: List[StmSegment],
) -> SupervisionSet:

    if not is_module_available("intervaltree"):
        raise ImportError(
            "intervaltree package not found. Please install..."
            " (pip install intervaltree)"
        )
    else:
        from intervaltree import IntervalTree

    if not is_module_available("jiwer"):
        raise ImportError(
            "jiwer package not found. Please install..." " (pip install jiwer==3.0.4)"
        )
    else:
        from jiwer import cer

    sset = deepcopy(supervisions)

    per_rec_its = {}
    for rid in recording_ids:
        per_rec_its[rid] = IntervalTree()
    for stm_seg in tqdm(aligned_stm_segs, desc="Building interval tree..."):
        per_rec_its[stm_seg.recording_id][stm_seg.start : stm_seg.end] = stm_seg

    for s in tqdm(sset, desc="Applying STM..."):
        # We need to find the closest and best-matching segment.
        # Some labeled segments were misplaced a lot and fixed by manual post-processing.
        # Hence, in order to find a good match, we tuned collar value to find all matches.
        # Example: 451 seconds, SBC027 recording.
        collar = 2.0
        matching_segments = list(
            filter(
                lambda x: x.data.speaker == s.speaker,
                per_rec_its[s.recording_id][s.start - collar : s.end + collar],
            )
        )
        # Alignments used slightly different speaker IDs for UNK speakers, so we relax the speaker ID matching.
        if not matching_segments:
            matching_segments = per_rec_its[s.recording_id][
                s.start - collar : s.end + collar
            ]

        best_cer = inf
        best_cer_res = None
        best_matching_seg = None
        best_iou = 0.0

        for matching_seg in matching_segments:
            cer_res = cer(
                norm_txt(s.text), norm_txt(matching_seg.data.text), return_dict=True
            )
            cer_val = cer_res["cer"]

            if cer_val < best_cer:
                best_cer = cer_val
                best_cer_res = cer_res
                best_matching_seg = matching_seg
                best_iou = compute_iou(s, matching_seg.data)

            # There's been an update between the alignments and the lhotse recipe, so some UNK speakers have shifted IDs.
            # It's enough to match the speaker names (or UNK).
            if (
                cer_val == best_cer
                and matching_seg.data.speaker.split("_")[1] == s.speaker.split("_")[1]
            ):
                current_iou = compute_iou(s, matching_seg.data)
                if current_iou >= best_iou:
                    best_matching_seg = matching_seg
                    best_cer_res = cer_res
                    best_iou = current_iou

        if (
            s.speaker.split("_")[1] == best_matching_seg.data.speaker.split("_")[1]
            and best_cer_res["substitutions"] == best_cer_res["deletions"] == 0
            and (best_cer < 0.5 or len(s.text) < 3)
        ):
            s.start = best_matching_seg.data.start
            s.duration = best_matching_seg.data.end - best_matching_seg.data.start
            s.text = best_matching_seg.data.text

            per_rec_its[s.recording_id].remove(best_matching_seg)

    return sset


def apply_aligned_stms(
    recording_ids: List[str], processed_supervisions: SupervisionSet
) -> Tuple[SupervisionSet, SupervisionSet]:
    aligned_for_asr_stm = retrieve_stm_file(
        "https://raw.githubusercontent.com/domklement/SBCSAE_alignments/main/alignments/stm/aligned_for_asr.stm"
    )
    aligned_for_diar_stm = retrieve_stm_file(
        "https://raw.githubusercontent.com/domklement/SBCSAE_alignments/main/alignments/stm/aligned_for_diar.stm"
    )

    asr_sup = apply_stm(recording_ids, processed_supervisions, aligned_for_asr_stm)
    diar_sup = apply_stm(recording_ids, processed_supervisions, aligned_for_diar_stm)

    return asr_sup, diar_sup
