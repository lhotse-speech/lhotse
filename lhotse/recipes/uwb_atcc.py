"""
University of West Bohemia Air Traffic Control Communication (UWB-ATCC)

Šmídl, Luboš, 2011, Air Traffic Control Communication, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11858/00-097C-0000-0001-CCA1-0.

Corpus contains recordings of communication between air traffic controllers and pilots. The speech is manually transcribed and labeled with the information about the speaker (pilot/controller, not the full identity of the person). The corpus is currently small (20 hours) but we plan to search for additional data next year. The audio data format is: 8kHz, 16bit PCM, mono.
"""

import hashlib
import logging
import re
import shutil
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
    is_module_available,
    resumable_download,
    safe_extract_rar,
)


def download_uwb_atcc(
    target_dir: Pathlike = ".", force_download: Optional[bool] = False
) -> Path:
    if not is_module_available("rarfile"):
        raise ImportError("Please 'pip install rarfile' first.")
    import rarfile

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "ZCU_CZ_ATC"
    rar_path = target_dir / f"{dataset_name}.rar"
    corpus_dir = target_dir / dataset_name
    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {dataset_name} because {completed_detector} exists.")
        return corpus_dir
    resumable_download(
        f"https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0001-CCA1-0/{dataset_name}.rar",
        filename=rar_path,
        completed_file_size=584245376,
        force_download=force_download,
    )
    if (
        hashlib.md5(open(rar_path, "rb").read()).hexdigest()
        != "44b4ea6ffe0ac0bf8fd29f14a735d23a"
    ):
        raise RuntimeError("MD5 checksum does not match")
    shutil.rmtree(corpus_dir, ignore_errors=True)
    with rarfile.RarFile(rar_path) as rar:
        safe_extract_rar(rar, path=corpus_dir)
    completed_detector.touch()
    return corpus_dir


def strip_accents(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


ABBREVIATIONS = {
    ".": "DECIMAL",
    "..": "DECIMAL DECIMAL",
    "FL": "FLIGHT LEVEL",
    "Fl": "FLIGHT LEVEL",
    "LLZ": "LOCALIZER",
    "PR": "PAPA ROMEO",
    "RWY": "RUNWAY",
}

UNKNOWN_ABBREVIATIONS = (
    "HDO",
    "VOZ",
    "VLM",
)

PHONETIC_ALPHABET = {
    "A": "ALFA",
    "B": "BRAVO",
    "C": "CHARLIE",
    "D": "DELTA",
    "E": "ECHO",
    "F": "FOXTROT",
    "G": "GOLF",
    "H": "HOTEL",
    "I": "INDIA",
    "J": "JULIETT",
    "K": "KILO",
    "L": "LIMA",
    "M": "MIKE",
    "N": "NOVEMBER",
    "O": "OSCAR",
    "P": "PAPA",
    "Q": "QUEBEC",
    "R": "ROMEO",
    "S": "SIERRA",
    "T": "TANGO",
    "U": "UNIFORM",
    "V": "VICTOR",
    "W": "WHISKEY",
    "X": "XRAY",
    "Y": "YANKEE",
    "Z": "ZULU",
}

INDIVIDUALLY_PRONOUNCED = (
    "ATR",
    "CRJ",
    "CSA",
    "CTO",
    "DEM",
    "DME",
    "EFC",
    "IFR",
    "ILS",
    "KLM",
    "QNH",
    "TMA",
    "UPS",
    "VFR",
    "VMC",
    "VOR",
)

FIX_TYPOS = {
    "ACCELARATING": "ACCELERATING",
    "ACCPET": "ACCEPT",
    "ACTUALY": "ACTUALLY",
    "AFETRNOON": "AFTERNOON",
    "AFFRIM": "AFFIRM",
    "AFTENOON": "AFTERNOON",
    "AFTERNON": "AFTERNOON",
    "AIRBORN": "AIRBORNE",
    "ALLRIGHT": "ALL RIGHT",
    "ALTITUED": "ALTITUDE",
    "APPORACH": "APPROACH",
    "APPORACHING": "APPROACHING",
    "APPRAOCH": "APPROACH",
    "APPROCHING": "APPROACHING",
    "APPRON": "APRON",
    "APROVED": "APPROVED",
    "APROXIMATELY": "APPROXIMATELY",
    "APROXIMETLY": "APPROXIMATELY",
    "AUSRTIAN": "AUSTRIAN",
    "AUSTRAIN": "AUSTRIAN",
    "AVAILBALE": "AVAILABLE",
    "AVALIABLE": "AVAILABLE",
    "AVIALABLE": "AVAILABLE",
    "BOARDLINE": "BROAD LINE",
    "BRUSSELES": "BRUSSELS",
    "BRUSSELS": "BRUSSELS",
    "CANCELED": "CANCELLED",
    "CANCELING": "CANCELLING",
    "CHALENGER": "CHALLENGER",
    "CHECH": "CZECH",
    "CIMB": "CLIMB",
    "CIMBING": "CLIMBING",
    "CLEARD": "CLEARED",
    "CLEARENCE": "CLEARANCE",
    "CLIBM": "CLIMB",
    "CLIMBIN": "CLIMBING",
    "CLMBING": "CLIMBING",
    "COMMING": "COMING",
    "CONACT": "CONTACT",
    "CONATACT": "CONTACT",
    "CONNTINUE": "CONTINUE",
    "CONTAC": "CONTACT",
    "CONTACE": "CONTACT",
    "CONTATC": "CONTACT",
    "CONTROLE": "CONTROL",
    "CONTROLO": "CONTROL",
    "COORECTION": "CORRECTION",
    "COPPIED": "COPIED",
    "CORECTION": "CORRECTION",
    "COTACT": "CONTACT",
    "COTINUE": "CONTINUE",
    "COTNACT": "CONTACT",
    "CURCUIT": "CIRCUIT",
    "DEAPARTURE": "DEPARTURE",
    "DEAPRTURE": "DEPARTURE",
    "DECEND": "DESCEND",
    "DEGEES": "DEGREES",
    "DEGRES": "DEGREES",
    "DENCENDING": "DESCENDING",
    "DEPARURE": "DEPARTURE",
    "DESCEDING": "DESCENDING",
    "DESCEN": "DESCEND",
    "DESCENG": "DESCEND",
    "DESCENIDNG": "DESCENDING",
    "DESCNED": "DESCEND",
    "DESECEND": "DESCEND",
    "DESEND": "DESCEND",
    "DESSCEND": "DESCEND",
    "DIREC": "DIRECT",
    "DISCRTION": "DISCRETION",
    "EADING": "HEADING",
    "ESTABLSIH": "ESTABLISH",
    "ESTALBISHED": "ESTABLISHED",
    "ETABLISHED": "ESTABLISHED",
    "ETIOPIAN": "ETHIOPIAN",
    "EVNING": "EVENING",
    "EXEPECT": "EXPECT",
    "EXPERIANCING": "EXPERIENCING",
    "EXTANSION": "EXTENSION",
    "FAVOUR": "FAVOR",
    "FINNARI": "FINNAIR",
    "FLIGTH": "FLIGHT",
    "FOLOW": "FOLLOW",
    "FOURTY": "FORTY",
    "GERMANWING": "GERMANWINGS",
    "GOAHEAD": "GO AHEAD",
    "GODD": "GOOD",
    "GOODBYE": "GOOD BYE",
    "GROSJET": "GROSSJET",
    "GROUDN": "GROUND",
    "HALLO": "HELLO",
    "HEADINT": "HEADING",
    "HEADNIG": "HEADING",
    "HEDING": "HEADING",
    "HODLING": "HOLDING",
    "HUDRED": "HUNDRED",
    "IFORMATION": "INFORMATION",
    "INBOUD": "INBOUND",
    "INBOUDN": "INBOUND",
    "INFOMRATION": "INFORMATION",
    "INITIALY": "INITIALLY",
    "INTERESCTION": "INTERSECTION",
    "KDNOTS": "KNOTS",
    "KNTOS": "KNOTS",
    "LANDA": "LAND",
    "LCIMB": "CLIMB",
    "LENGHT": "LENGTH",
    "LENGT": "LENGTH",
    "LEVELED": "LEVEL",
    "LEVLE": "LEVEL",
    "LIGHER": "LIGHTER",
    "LOUND": "LOUD",
    "LUFHANSA": "LUFTHANSA",
    "LUFHTANSA": "LUFTHANSA",
    "LUFTAHNSA": "LUFTHANSA",
    "LUFTHASNA": "LUFTHANSA",
    "MAINATINANING": "MAINTAINING",
    "MAINTAING": "MAINTAINING",
    "MAINTANING": "MAINTAINING",
    "MAITAIN": "MAINTAIN",
    "MINTUES": "MINUTES",
    "MOLDAVA": "MOLDOVA",
    "MOORNING": "MORNING",
    "NEAGATIVE": "NEGATIVE",
    "NINTEEN": "NINETEEN",
    "NINTY": "NINETY",
    "NOICE": "NOISE",
    "NORTHSHUTTLE": "NORSHUTTLE",
    "NORTHSTHUTTEL": "NORSHUTTLE",
    "NORTHSTHUTTLE": "NORSHUTTLE",
    "NOSIG": "NOSING",
    "NOSRHUTLE": "NORSHUTTLE",
    "OPOSITE": "OPPOSITE",
    "OT": "TO",
    "PASSINF": "PASSING",
    "PASSIN": "PASSING",
    "PLESE": "PLEASE",
    "POSSBILE": "POSSIBLE",
    "PREFERED": "PREFERRED",
    "PROCCEDING": "PROCEEDING",
    "PROCEEDTO": "PROCEED TO",
    "PSSING": "PASSING",
    "QHN": "QNH",
    "QUANTAS": "QANTAS",
    "QUATARI": "QATARI",
    "RADR": "RADAR",
    "READBACK": "READ BACK",
    "RECOMEND": "RECOMMEND",
    "RECOMEND": "RECOMMEND",
    "REQEUSTED": "REQUESTED",
    "REQEUST": "REQUEST",
    "REQUESTE": "REQUEST",
    "REQUSTED": "REQUESTED",
    "REQUSTING": "REQUESTING",
    "RESETING": "RESETTING",
    "RESRTICTION": "RESTRICTION",
    "RESTRCTIONS": "RESTRICTIONS",
    "RESTRISCTION": "RESTRICTION",
    "RIGH": "RIGHT",
    "ROGGER": "ROGER",
    "ROGRE": "ROGER",
    "SESION": "DECISION",
    "SHOTRCUT": "SHORTCUT",
    "SINAGAPORE": "SINGAPORE",
    "SINGAPOOR": "SINGAPORE",
    "SKYRAVEL": "SKYTRAVEL",
    "SKYTAVEL": "SKYTRAVEL",
    "SMARTWING": "SMARTWINGS",
    "SPEEDBIRG": "SPEEDBIRD",
    "SQUAKING": "SQUAWKING",
    "SQUAK": "SQUAWK",
    "SQUWAK": "SQUAWK",
    "STANDAR": "STANDARD",
    "STANDART": "STANDARD",
    "STARTUP": "START UP",
    "SUFFICIAN": "SUFFICIENT",
    "SWTICHING": "SWITCHING",
    "TAHNK": "THANK",
    "TECHNICAN": "TECHNICIAN",
    "TELAVIV": "TEL AVIV",
    "THAT'T": "THAT'S",
    "THIRDY": "THIRTY",
    "THOSUAND": "THOUSAND",
    "THOUASAND": "THOUSAND",
    "TIMECHECK": "TIME CHECK",
    "TRAFIC": "TRAFFIC",
    "TRESHOLD": "THRESHOLD",
    "TUBULENCE": "TURBULENCE",
    "TURBOLENCE": "TURBULENCE",
    "TURUBLENCE": "TURBULENCE",
    "UNREADEBLE": "UNREADABLE",
    "UNTILL": "UNTIL",
    "UTNIL": "UNTIL",
    "VACAT": "VACATE",
    "VECTORIN": "VECTOR IN",
    "WCHICH": "WHICH",
    "WIHT": "WITH",
    "WINE": "WIEN",
    "WIZZIAR": "WIZZAIR",
    "WONDREFUL": "WONDERFUL",
}

COLLAPSE_WORDS = (
    ("AIR SPACE", "AIRSPACE"),
    ("CLEAR FOR", "CLEARED FOR"),
    ("DESCENT TO", "DESCEND TO"),
    ("DESCENT FLIGHT", "DESCEND FLIGHT"),
    ("DESCEND RATE", "DESCENT RATE"),
    ("STAND BYE", "STANDBY"),
)


def text_normalize(
    text: str,
    silence_sym: str,
    breath_sym: str,
    noise_sym: str,
    foreign_sym: str,  # When None, will output foreign words
    unintelligble_sym: str,  # When None, will output unintellible words
    partial_sym: str,  # When None, will output partial words
    unknown_sym: str,
):

    assert is_module_available(
        "num2words"
    ), "Please run 'pip install num2words' for number to word normalization."
    from num2words import num2words

    # regex patterns
    BRACKET_PADDING_PATTERN1 = re.compile(r"([\w\.\+])(\[|\()")
    BRACKET_PADDING_PATTERN2 = re.compile(r"(\]|\))([\w\+])")
    COMMENT_PATTERN = re.compile(r"\[comment_\|].*?\[\|_comment]")
    BACKGROUND_SPEECH_PATTERN = re.compile(
        r"\[background_speech_\|](.*?)\[\|_background_speech]"
    )
    NOISE_PATTERN = re.compile(r"\[noise_\|](.*?)\[\|_noise]")
    SPEAKER_PATTERN = re.compile(r"\[speaker_\|](.*?)\[\|_speaker]")
    DECIMAL_NUMBER_PATTERN = re.compile(r"\.([0-9])")
    NUMBER_DECIMAL_PATTERN = re.compile(r"([0-9])\.")
    PHONETIC_INTERRUPTED_PATTERN1 = re.compile(r"([A-Z]+\+)")
    PHONETIC_INTERRUPTED_PATTERN2 = re.compile(r"(\+[A-Z]+)")
    INTERRUPTED_PATTERN1 = re.compile(r"(\w+\+)")
    INTERRUPTED_PATTERN2 = re.compile(r"(\+\w+)")
    ABBREVIATION_PATTERN = re.compile(r"\(((\w*|\s*|\+)*)\(((\w*|\s*)*)\)\)")
    SPLIT_NUMERIC_ALPHA = re.compile(r"([0-9])([A-Za-z])")
    SPLIT_ALPHA_NUMERIC = re.compile(r"([A-Za-z])([0-9])")
    NO_ENG_PATTERN = re.compile(r"\[NO_ENG_\|](.*?)\[\|_NO_ENG]")
    CZECH_PATTERN = re.compile(r"\[CZECH_\|](.*?)\[\|_CZECH]")
    UNINTELLIGIBLE_PATTERN = re.compile(
        r"\[UNINTELLIGIBLE_\|](.*?)\[\|_UNINTELLIGIBLE]"
    )
    WHITESPACE_PATTERN = re.compile(r"  +")

    text = BRACKET_PADDING_PATTERN1.sub(r"\1 \2", text)
    text = BRACKET_PADDING_PATTERN2.sub(r"\1 \2", text)
    text = text.replace("](", "] (")

    text = text.replace("°", "")
    text = text.replace("?", "")
    text = text.replace("¨", "")
    text = text.replace("´", "'")

    text = COMMENT_PATTERN.sub("", text)

    text = BACKGROUND_SPEECH_PATTERN.sub(r"\1", text)
    text = NOISE_PATTERN.sub(r"\1", text)
    text = SPEAKER_PATTERN.sub(r"\1", text)
    text = DECIMAL_NUMBER_PATTERN.sub(r". \1", text)
    text = NUMBER_DECIMAL_PATTERN.sub(r"\1 .", text)

    text = PHONETIC_INTERRUPTED_PATTERN1.sub(lambda m: m.group(1).lower(), text)
    text = PHONETIC_INTERRUPTED_PATTERN2.sub(lambda m: m.group(1).lower(), text)

    text = ABBREVIATION_PATTERN.sub(r"\1", text)

    # apply this word fix before applying split numeric-alpha patterns
    text = text.replace("6raha", "praha")

    text = SPLIT_NUMERIC_ALPHA.sub(r"\1 \2", text)
    text = SPLIT_ALPHA_NUMERIC.sub(r"\1 \2", text)

    text = strip_accents(text)

    simple_replace = {
        "[ehm_]": breath_sym,
        "[noise]": noise_sym,
        "[unintelligible]": unknown_sym,
        "[background_speech]": noise_sym,
        "[speaker]": breath_sym,
    }

    result = []
    for w in text.split():
        if w in simple_replace:
            result.append(simple_replace[w])
        elif w in UNKNOWN_ABBREVIATIONS:
            result.append(unknown_sym)
        elif w in ABBREVIATIONS:
            result.append(ABBREVIATIONS[w])
        elif w in INDIVIDUALLY_PRONOUNCED:
            result.append(" ".join([*w]).upper())
        elif w in PHONETIC_ALPHABET:
            result.append(PHONETIC_ALPHABET[w])
        elif w.isdigit():
            result.append(num2words(w).replace("-", " ").replace(",", "").upper())
        else:
            result.append(w.upper())
    text = " ".join(result)

    # from here on, text, with the exception of inserted symbols, is all uppercase,
    # therfore [no_eng], [czech] and [unintelligible] pattern matching must be done in uppercase.

    if foreign_sym == None:
        text = text.replace("[NO_ENG]", unknown_sym)
        text = NO_ENG_PATTERN.sub(r"\1", text)
        text = CZECH_PATTERN.sub(r"\1", text)
    else:
        text = text.replace("[NO_ENG]", foreign_sym)
        text = NO_ENG_PATTERN.sub(foreign_sym, text)
        text = CZECH_PATTERN.sub(foreign_sym, text)

    if unintelligble_sym == None:
        text = UNINTELLIGIBLE_PATTERN.sub(r"\1", text)
    else:
        text = UNINTELLIGIBLE_PATTERN.sub(unintelligble_sym, text)

    if partial_sym != None:
        text = INTERRUPTED_PATTERN1.sub(partial_sym, text)
        text = INTERRUPTED_PATTERN2.sub(partial_sym, text)

    text = text.replace("+", "")

    text = WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()

    text = " ".join([FIX_TYPOS[w] if w in FIX_TYPOS else w for w in text.split()])

    for pair in COLLAPSE_WORDS:
        text = text.replace(pair[0], pair[1])

    return text


SPEAKER_TO_ID_SUFFIX = {"air_ground": "PIAT", "ground": "AT", "air": "PI"}


def finish_segment(supervisions, segment, end_time):
    segment.duration = end_time - segment.start
    segment.id += "_%06d_%s" % (end_time * 100, SPEAKER_TO_ID_SUFFIX[segment.speaker])
    supervisions.append(segment)


def prepare_uwb_atcc(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    silence_sym: Optional[str] = "",
    breath_sym: Optional[str] = "",
    noise_sym: Optional[str] = "",
    foreign_sym: Optional[str] = "<unk>",
    partial_sym: Optional[str] = "<unk>",
    unintelligble_sym: Optional[str] = "<unk>",
    unknown_sym: Optional[str] = "<unk>",
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param silence_sym: str, silence symbol
    :param breath_sym: str, breath symbol
    :param noise_sym: str, noise symbol
    :param foreign_sym: str, foreign symbol. when set to None, will output foreign words
    :param partial_sym: str, partial symbol. When set to None, will output partial words
    :param unintelligble_sym: str, unintellible symbol. When set to None, will output unintelligble words
    :param unknown_sym: str, unknown symbol
    :return: The RecordingSet and SupervisionSet with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    trs_files = sorted(corpus_dir.glob("*.trs"), key=lambda p: p.name)
    assert len(trs_files) == 2657

    recordings = []
    supervisions = []

    # regex pattern for multiple whitespaces
    WHITESPACE_PATTERN = re.compile(r"  +")

    from tqdm.auto import tqdm

    for t in tqdm(trs_files, desc="Preparing"):
        # repair broken xml files
        if t.stem in (
            "ACCU-80UXVV",
            "ACCU-7NqzYv",
            "ACCU-PhR5Oj",
            "ACCU-JaeNLH",
            "TWR-XgqNSk",
        ):
            with open(t, encoding="cp1250") as f:
                root = ET.fromstring(f.read() + "</Turn></Section></Episode></Trans>")
        else:
            root = ET.parse(t).getroot()

        audio_path = corpus_dir / root.attrib["audio_filename"][len("e2_") :]
        if not audio_path.is_file():
            logging.warning(f"No such file: {audio_path}")
            continue

        recording = Recording.from_file(audio_path)
        recordings.append(recording)

        last_segment = None

        for section in root.findall(".//Section"):
            for turn in section:
                if turn.tag != "Turn":
                    logging.warning(f"Unexpected tag: {turn.tag}")
                    continue

                end_time = float(turn.attrib["endTime"])

                for sync in turn:
                    if sync.tag != "Sync":
                        logging.warning(f"Unexpected tag: {sync.tag}")
                        continue

                    time = float(sync.attrib["time"])

                    if last_segment:
                        finish_segment(supervisions, last_segment, time)
                        last_segment = None

                    text = sync.tail.strip()
                    if text == "":
                        continue

                    orig_text = text

                    # extract speaker id
                    if "[air_|]" in text or "[ground_|]" in text:
                        speaker = "air_ground"
                    elif "[air]" in text:
                        speaker = "air"
                    elif "[ground]" in text:
                        speaker = "ground"
                    else:
                        continue

                    # remove speaker id tags
                    text = text.replace("][", "] [")
                    for label in (
                        "[air_|]",
                        "[|_air]",
                        "[ground_|]",
                        "[|_ground]",
                        "[air]",
                        "[ground]",
                    ):
                        text = text.replace(label, "")

                    text = text_normalize(
                        text,
                        silence_sym=silence_sym,
                        breath_sym=breath_sym,
                        noise_sym=noise_sym,
                        foreign_sym=foreign_sym,
                        partial_sym=partial_sym,
                        unintelligble_sym=unintelligble_sym,
                        unknown_sym=unknown_sym,
                    )

                    if text == "":
                        continue

                    last_segment = SupervisionSegment(
                        id="uwb-atcc_%s_%06d" % (audio_path.stem, time * 100),
                        recording_id=recording.id,
                        start=time,
                        duration=0,  # populated by finish_segment()
                        channel=0,
                        language="English",
                        text=text,
                        speaker=speaker,
                        custom={
                            "type": section.attrib["type"],
                            "orig_text": WHITESPACE_PATTERN.sub(" ", orig_text.strip()),
                        },
                    )

                if last_segment:
                    finish_segment(supervisions, last_segment, end_time)
                    last_segment = None

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        supervision_set.to_file(output_dir / "uwb_atcc_supervisions_all.jsonl.gz")
        recording_set.to_file(output_dir / "uwb_atcc_recordings_all.jsonl.gz")

    return {"recordings": recording_set, "supervisions": supervision_set}
