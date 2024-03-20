"""
The SEAME corpora of Singaporean Codeswitched English and Mandarin.

This corpus comes defined with a training split and two development splits:

train -- A mix of codeswitched, Mandarin and Singaporean English
dev_sge -- A set of primarily Singaporean English though there is codeswitching  
dev_man -- A set of primarily Mandarin though there is also some codeswitching

From these dev sets we separate the sentences with purely English, purely
Mandarin sentences, and mixes of the two to form new sets called:

dev_eng -- English only sentences (at least more so than dev_sge)
dev_cmn -- Mandarin only sentences (at least more so than dev_man)
dev_csw -- Codeswitched only sentences (at least in theory)

All audio files (found in audio in the directory shown in the directory tree
below), are sampled at 16kHz and stored in the .flac format.
 
The directory structure of the corpus is

/LDC2015S04/
├── data
│   ├── conversation
│   │   ├── audio
│   │   └── transcript
│   │       ├── phaseI
│   │       └── phaseII
│   └── interview
│       ├── audio
│       └── transcript
│           ├── phaseI
│           └── phaseII
├── docs
├── original
│   ├── data
│   │   ├── conversation
│   │   │   ├── audio
│   │   │   └── transcript
│   │   └── interview
│   │       ├── audio
│   │       └── transcript
│   └── docs
└── partitions
    ├── dev_man
    ├── dev_sge
    └── train
"""

import logging
import os
import shutil
import tarfile
import collections
from pathlib import Path
from typing import Dict, Optional, Union
import soundfile as sf

from lhotse import AudioSource, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.qa import (
    remove_missing_recordings_and_supervisions,
    trim_supervisions_to_recordings,
)
from lhotse.utils import Pathlike
from lhotse.manipulation import combine
import random as rd
import re
import itertools
import sys

rd.seed(531)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
remove_punc = '()[]{}.,?·@，。、「」＃"~-—#%_`｀×*（）［］&【】～ｌ\\'
pattern = str.maketrans(remove_punc, " " * len(remove_punc))

translate_char_source = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺé"
translate_char_target = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyze"
pattern2 = str.maketrans(translate_char_source, translate_char_target)

all_chars = (chr(i) for i in range(sys.maxunicode))
categories = {"Cc"}
control_chars = "".join(map(chr, itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0))))
control_char_re = re.compile("[%s]" % re.escape(control_chars))


def prepare_seame(
    corpus_dir: Pathlike,
    split_dir: Pathlike,
    clean_text: bool = True,
    delimiter: str = "|",
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests of Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param split_dir: Path to splits from https://github.com/zengzp0912/SEAME-dev-set.git
    :param output_dir: Pathlike, the path where to write the manifests.
    :param delimiter: used to split the code switching text
    :return: a Dict whose key is the dataset part, and whose values are Dicts
        with keys 'recordings', and 'supervisions'
    """
    corpus_dir = Path(corpus_dir)
    split_dir = Path(split_dir)
    recos = id2recos(corpus_dir)
    dataset_parts = ["valid", "train", "dev_man", "dev_sge"]
    manifests = {}

    splits = load_splits(split_dir, corpus_dir, output_dir)

    for part in dataset_parts:
        logging.info(f"processing {part} ...")

        #     lang = 'Mandarin' if part == 'dev_man' else 'English'
        segments, recs = [], {}
        for uttid, values in splits[part].items():

            text, start, end, spk, _, _ = values

            # map to sec, original ms, start, end here has / 10
            start, end = (
                float(start) / 100,
                float(end) / 100,
            )
            recoid = uttid.split("-")[0]
            duration = round(float(end) - float(start), ndigits=8)
            if clean_text:
                normalized_text = normalize_text(text)
                no_noise_text = normalized_text.replace("<noise>", "").replace(
                    "<unk>", ""
                )
                no_noise_text = remove_redundant_whitespaces(no_noise_text)
                # remove short utterances
                if len(no_noise_text) == 0:
                    continue

            # split code switched text with "|": 然 然 service 罢 了 -> 然 然  | service| 罢 了
            # and get lid labels for each chunk: lid = ['<zh>', '<en>', '<zh>']
            lid, no_noise_text, cs = add_lid(no_noise_text, delimiter)
            segments.append(
                SupervisionSegment(
                    id=uttid,
                    recording_id=recoid,
                    start=float(start),
                    duration=duration,
                    channel=0,
                    text=no_noise_text,
                    language=lid,
                    speaker=spk,
                    custom={"cs": cs},
                )
            )
            if recoid not in recs:
                audio_sf = sf.SoundFile(str(recos[recoid]))
                recs[recoid] = Recording(
                    id=recoid,
                    sources=[
                        AudioSource(
                            type="file",
                            channels=[0],
                            source=str(recos[recoid]),
                        ),
                    ],
                    sampling_rate=audio_sf.samplerate,
                    num_samples=audio_sf.frames,
                    duration=audio_sf.frames / audio_sf.samplerate,
                )
        supervisions = SupervisionSet.from_segments(segments).filter(
            lambda s: s.duration > 0.0
        )
        recordings = RecordingSet.from_recordings(recs.values())
        recordings, supervisions = remove_missing_recordings_and_supervisions(
            recordings,
            supervisions,
        )
        supervisions = trim_supervisions_to_recordings(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)
        manifests[part] = {
            'reclogging.info(f"Creating {part} supervisions")ordings': recordings,
            "supervisions": supervisions,
        }

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recordings.to_file(output_dir / f"recordings_{part}.jsonl.gz")
            supervisions.to_file(output_dir / f"supervisions_{part}.jsonl.gz")
    return manifests


def get_lid(c):
    if len(c) == 0:
        return ""
    if is_english(c[0]):
        return "<en>"
    else:
        return "<zh>"


def add_lid(txt, delimiter="|"):
    txt = txt.split()
    new_txt = []
    lid = []
    prev = ""
    cs = False
    for i, word in enumerate(txt):
        curr_lid = get_lid(word)
        if word != "<noise>" and curr_lid != prev:
            if i > 0:
                new_txt.append(delimiter)
                cs = True
            new_txt.append(word)

            lid.append(curr_lid)
            prev = curr_lid
        new_txt.append(word)

    return lid, " ".join(new_txt), cs


def is_english(c):
    """check character is in English"""
    return ord(c.lower()) >= ord("a") and ord(c.lower()) <= ord("z")


def is_mandarin(c):
    """check character is Mandarin"""
    return (
        not is_english(c)
        and not c.isdigit()
        and c != " "
        and c != "<"
        and c != ">"
        and c != "'"
    )


def remove_control_chars(text):
    """remove unprintable characters"""
    return control_char_re.sub("", text)


def remove_redundant_whitespaces(text):
    """remove redundant whitespaces"""
    return re.sub(" +", " ", text).strip()


def insert_space_between_mandarin(text):
    """insert space between Mandarin characters"""

    if len(text) <= 1:
        return text
    out_text = text[0]
    for i in range(1, len(text)):
        if is_mandarin(text[i]):
            out_text += " "
        out_text += text[i]
        if is_mandarin(text[i]):
            out_text += " "

    return out_text


def remove_repeated_noise(text, pattern="<noise>"):
    """remove repeated <noise>"""

    if len(re.findall(pattern, text)) <= 1:
        return text

    out_text = ""
    text_split = text.split()
    out_text = [text_split[0]]
    for i in range(1, len(text_split)):
        if text_split[i] == pattern and text_split[i - 1] == pattern:
            continue
        else:
            out_text.append(text_split[i])

    return " ".join(out_text)


def normalize_text(text):
    """normalize a text sequence"""

    rmtext = re.sub(
        r"\(((pp)(\w)+)\)",
        "<noise>",
        text.lower(),
    )
    rmtext = re.sub(
        r"\<((pp)(\w)+)\>",
        "<noise>",
        rmtext,
    )
    rmtext = rmtext.translate(pattern)
    rmtext = remove_control_chars(rmtext)
    output_text = ""
    for wrd in rmtext.split():
        if wrd in {
            "ppl",
            "ppc",
            "ppb",
            "ppo",
            "<v-noise>",
        }:
            wrd = "<noise>"
        output_text += f"{wrd} "

    output_text = output_text.strip()
    output_text = output_text.translate(pattern2)
    output_text = output_text.replace("<unl>", "<unk>")
    output_text = output_text.replace("< unk >", "<unk>")
    output_text = re.sub(r"\<((unk)[a-z ]+)\>", "<unk>", output_text)
    output_text = insert_space_between_mandarin(output_text)
    output_text = remove_redundant_whitespaces(output_text)
    output_text = remove_repeated_noise(output_text, "<noise>")

    return output_text


def fit_format(digit):
    """fit file name format"""
    str_digit = str(float(digit) / 10.0)
    if int(str_digit[-1]) >= 5:
        return float(digit) + 1
    else:
        return float(digit)


def read_trans(data_dict, pth, phs):
    """read transcriptions (SEAME/{type}/transcript/phaseII/??.txt)"""

    with open(pth, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            if phs.lower() == "phasei":
                lang = None
                if len(line.split("\t")) == 4:
                    idx, start, end, text = line.split("\t")
                else:
                    idx, cont = line.split("\t", 1)
                    print(f"Skip {idx} with {cont}... (no transcript)")
                    continue
            elif phs.lower() == "phaseii":
                idx, start, end, lang, text = line.split("\t")
            else:
                print("folder error! not PhaseI or PhaseII")
                raise
            # start: start time in msec
            # end: end time in msec

            start_ms = start
            end_ms = end

            # fit the devset format
            s_len, e_len = len(start), len(end)
            if s_len < 5:
                start = int(round(fit_format(start) / 10, 0))
                start = str(start).zfill(5)
            else:
                start = int(round(float(start) / 10, 0))
            if e_len < 5:
                end = int(round(fit_format(end) / 10, 0))
                end = str(end).zfill(5)
            else:
                end = int(round(float(end) / 10, 0))

            name = f"{idx}-{start}-{end}"
            if name not in data_dict:
                if idx.split("_")[0][0].isdigit():
                    spkr = idx.split("_")[0][2:-2].lower()
                else:
                    spkr = idx.split("_")[0][:5].lower()

                data_dict[name.lower()] = [text, start, end, spkr, start_ms, end_ms]
            else:
                print("Repeated idx!")
                raise


def read_list(pth):
    """read data list (data/SEAME-dev-set/train/wav_file.txt)"""

    idxs = []
    with open(pth, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            _, idx = line.split("/")[-3], line.split("/")[-2]
            idxs.append(idx)
        return idxs


def split_val(data_dict, num_val=None):
    """split train/val sets"""
    tr_list = list(data_dict.keys())
    rd.shuffle(tr_list)
    val_len = num_val if num_val else int(len(tr_list) * 0.05)
    tr_list, val_list = tr_list[:-val_len], tr_list[-val_len:]
    valid = {key: data_dict[key] for key in val_list}
    train = {key: data_dict[key] for key in tr_list}
    return train, valid


def load_trans(corpus_dir):
    mapping = {}
    audio_type = ["conversation", "interview"]
    for atp in audio_type:
        for phs in ["phaseII"]:
            for txt in os.listdir(
                os.path.join(corpus_dir / "data", atp, "transcript", phs)
            ):
                trans_pth = os.path.join(
                    corpus_dir / "data", atp, "transcript", phs, txt
                )
                read_trans(mapping, trans_pth, phs)
    return mapping


def test_split(test, data_dict):
    """find testing data in data_dict"""
    dev_dict = {}
    data = list(data_dict.keys())
    count = 0
    space = {}
    idx_space = {}
    for key in data:
        idx, start, end = key.split("-")
        idx_space[idx] = idx_space.get(idx, []) + [[str(start), str(end)]]
        space[idx] = space.get(idx, []) + [[float(start), float(end)]]
    for key in test:
        idx, start, end = key.split("-")
        start, end = float(start), float(end)
        for list_idx, time in enumerate(space[idx]):
            if abs(start - time[0]) < 3 and abs(end - time[1]) < 3:
                count += 1
                time1, time2 = idx_space.get(idx)[list_idx]
                dev_dict[(f"{idx}-{time1}-{time2}")] = data_dict[
                    (f"{idx}-{time1}-{time2}")
                ]

    logging.info(f"Test set = {count}/{len(test)}")
    return dev_dict


def read_dev_text(pth, rmspk=False):
    """read dev set text data (data/SEAME-dev-set/{devset}/text)"""
    idxs = []
    with open(pth, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            line = line.split()[0]
            if rmspk:
                line = line.split("-", 1)[-1]
            idxs.append(line.lower())
        return idxs


def extract_mandarin_only(text):
    """remove other symbols except for Mandarin characters in a string"""
    return "".join([c for c in text if is_mandarin(c)])


def write_mandarin_only_text(data_dict, char_file1, char_file2):
    """write Mandarin text data"""

    counter = collections.Counter()

    for idx, content in data_dict.items():
        text = content[0]
        text = normalize_text(text)
        text = text.replace("<noise>", "")
        text = text.replace("<unk>", "")
        text = remove_redundant_whitespaces(text)
        text = extract_mandarin_only(text)
        counter.update(text)

    vocab_list = sorted(counter.keys())
    logging.info(f"Mandarin vocab size = {len(vocab_list)}")

    with open(char_file1, "w") as fp:
        fp.write("\n".join(vocab_list))
    with open(char_file2, "w") as fp:
        fp.write('bpe_nlsyms="<noise>,▁' + ",▁".join(vocab_list) + '"\n')
        fp.write(f"man_chars={len(vocab_list)}")


def load_splits(split_dir, corpus_dir, out_dir):
    splits = {}
    # load all transcripts
    data_dict = load_trans(corpus_dir)
    dataset_parts = ["train", "dev_man", "dev_sge"]
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # extract mandarin vocab to be used as user predefined bpe
        write_mandarin_only_text(
            data_dict,
            os.path.join(out_dir, "text.man"),
            os.path.join(out_dir, "token.man.1"),
        )

    for split in dataset_parts:
        logging.info(f"Reading {split} set indices...")
        splits_path = split_dir / split
        if split == "train":
            # load train idx
            train_audio_pth = splits_path / "wav_file.txt"
            train_audio_idx = read_list(train_audio_pth)
            train_dict = {
                key: val
                for key, val in data_dict.items()
                if key.split("-")[0] in train_audio_idx
            }
            assert (
                len(train_dict) == 97294
            ), f"Train: {len(train_dict)}, should be exactly 97294"
            train, valid = split_val(train_dict)
            splits["valid"] = valid
            splits["train"] = train
        else:
            rmspk = True
            dev_path = os.path.join(splits_path, "text")
            dev_idx = read_dev_text(dev_path, rmspk)
            dev_dict = test_split(dev_idx, data_dict)
            splits[split] = dev_dict
            if split == "dev_man":
                assert (
                    len(dev_dict) == 6531
                ), f"Train: {len(train_dict)}, should be exactly 6531"
            elif split == "dev_sge":
                assert (
                    len(dev_dict) == 5321
                ), f"Train: {len(train_dict)}, should be exactly 97294"
    return splits


def id2recos(path):
    recos = {}
    for p in path.glob("data/*/audio/*.flac"):
        recoid = p.stem.lower()
        recos[recoid] = p.resolve(strict=False)
    return recos
