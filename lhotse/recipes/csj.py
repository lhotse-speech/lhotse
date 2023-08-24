#!/usr/bin/env python3
# Copyright    2023  The University of Electro-Communications  (Author: Teo Wen Shen)  # noqa
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Corpus owner: https://clrd.ninjal.ac.jp/csj/en/index.html
Corpus description:
- http://www.lrec-conf.org/proceedings/lrec2000/pdf/262.pdf
- https://isca-speech.org/archive_open/archive_papers/sspr2003/sspr_mmo2.pdf

This script accesses the CSJ/SDB/{core,noncore} directories, and generates
transcripts where each word is formatted as '{surface}+{morph}+{pron}'.

NOTE: Besides the eval1, eval2, eval3, and excluded datasets introduced in
kaldi, this script also creates a separate validation dataset. Recording
sessions chosen for validation are listed in this script.

This script does the following in sequence:-

**MOVE**
0. This stage is skipped if `--transcript-dir` is not provided.
1. Copies each .sdb files from /SDB into its own directory in the designated
  `transcript_dir`, i.e. {transcript_dir}/{spk_id}/{spk_id}.sdb
2. Verifies that the corresponding wav file exists in the /WAV directory, and
   outputs that absolute path into {spk_id}-wav.list
3. Moves the predefined datasets for eval1, eval2, eval3, and excluded, into
   its own dataset directory
4. Touches a .done_mv in `transcript_dir`.
NOTE: If a .done_mv exists already in `transcript_dir`, then this stage is skipped.

**PREPARE MANIFESTS**
1. Parses all .sdb files it can find within `transcript_dir` in disfluent mode.
2. Generates supervisions and recordings manifests for each dataset part.

Differences to kaldi include:-
1. The evaluation datasets do not follow `trans_dir`/eval/eval{i}, but are
   instead saved in the same level as core, noncore, and excluded.
2. A validation dataset is explicitly specified.
3. Utterances with "×" are not removed. You will need to remove them in a
   later stage.
4. Segments are not concatenated, unless an F-tag, D-tag, L-tag, or A-tag
   spans between two segments.
5. Multi-segment M-tags, R-tags, and O-tags are removed, while M-tags,
   R-tags, and O-tags within a segment are retained.

Example structure of SupervisionSegment.custom:-
{
  "raw": (
      "(F_えー)+感動詞+(F_エー) (M_(F_うーん)+感動詞+(M_(F_(W_ウー;ウーン)) "
      "それ+代名詞+ソレ だっ+助動詞/促音便+ダッ たら)+助動詞+タラ) と+助詞/格助詞+ト "
      "いう+動詞/ワア行五段+ユー の+助詞/準体助詞+ノ は+助詞/係助詞+ワ "
      "(F_えー)+感動詞+(F_エー)"
      ),
  "disfluent": "えーうーんそれだったらというのはえー",
  "disfluent_tag": "F,F,M/F,M/F,M/F,M,M,M,M,M,M,,,,,,F,F"
}
NOTE:
1. XX_tag is guaranteed to be the same length as XX. It labels the tag to which each
   character belongs. It is useful for evaluation.
2. The SupervisionSegment.text field is populated with 'disfluent', i.e.
   SupervisionSegment.text == SupervisionSegment.custom['dislfuent'], so that this
   supervision is compatible with other recipes.

The transcript directory, if generated, has this structure:-
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
     - A01F0576.sdb
     - A01F0576-trans.txt
     - A01F0576-wav.list
   - ...
   - D03M0038
     - D03M0038.sdb
     - D03M0038-L-trans.txt
     - D03M0038-L-wav.list
     - D03M0038-R-trans.txt
     - D03M0038-R-wav.list
 - valid
     - ...

"""

import copy
import logging
import re
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

# ---------------------------------- #
#    Prepare transcripts from SDB    #


_FULL_DATA_PARTS = ["eval1", "eval2", "eval3", "excluded", "valid", "core", "noncore"]

# Exclude speaker ID
_A01M0056 = [
    "S05M0613",
    "R00M0187",
    "D01M0019",
    "D04M0056",
    "D02M0028",
    "D03M0017",
]

_VALID = [
    "A01M0264",
    "A01M0377",
    "A01M0776",
    "A01M0891",
    "A03F0109",
    "A04M0899",
    "A05M0420",
    "A07M0318",
    "A07M0912",
    "A11M0795",
    "A12M0983",
    "D03F0058",
    "R00M0415",
    "R01F0101",
    "R01F0125",
    "R02M0073",
    "R03F0108",
    "R03F0157",
    "S00F0014",
    "S00M0793",
    "S01F0507",
    "S02F0122",
    "S02F0362",
    "S02M1351",
    "S02M1372",
    "S03F1199",
    "S04F1020",
    "S05F0443",
    "S07F0853",
    "S07F1333",
    "S07M0827",
    "S08F0717",
    "S08F1340",
    "S09M0619",
    "S10M1090",
    "S10M1275",
    "S11F0578",
    "S11M0864",
    "S11M1174",
]

# Evaluation set ID
_EVAL = [
    # eval1
    [
        "A01M0110",
        "A01M0137",
        "A01M0097",
        "A04M0123",
        "A04M0121",
        "A04M0051",
        "A03M0156",
        "A03M0112",
        "A03M0106",
        "A05M0011",
    ],
    # eval2
    [
        "A01M0056",
        "A03F0072",
        "A02M0012",
        "A03M0016",
        "A06M0064",
        "A06F0135",
        "A01F0034",
        "A01F0063",
        "A01F0001",
        "A01M0141",
    ],
    # eval3
    [
        "S00M0112",
        "S00F0066",
        "S00M0213",
        "S00F0019",
        "S00M0079",
        "S01F0105",
        "S00F0152",
        "S00M0070",
        "S00M0008",
        "S00F0148",
    ],
]


def _create_trans_dir(corpus_dir: Path, trans_dir: Path):

    if (trans_dir / ".done_mv").exists():
        logging.info(
            f"{trans_dir} already created. "
            f"Delete {trans_dir / '.done_mv'} to create again."
        )
        return

    for ori_files in (corpus_dir / "MORPH/SDB").glob("*/*.sdb"):
        vol = ori_files.parts[-2]
        spk_id = ori_files.name[:-4]
        new_dir = trans_dir / vol / spk_id
        new_dir.mkdir(parents=True, exist_ok=True)
        wav_dir = corpus_dir / "WAV" / vol

        if spk_id[0] == "D":
            l_wav = wav_dir / f"{spk_id}-L.wav"
            r_wav = wav_dir / f"{spk_id}-R.wav"
            assert l_wav.is_file(), f"{spk_id}-L.wav cannot be found"
            assert r_wav.is_file(), f"{spk_id}-R.wav cannot be found"
            (new_dir / f"{spk_id}-L-wav.list").write_text(
                l_wav.as_posix(), encoding="utf8"
            )
            (new_dir / f"{spk_id}-R-wav.list").write_text(
                r_wav.as_posix(), encoding="utf8"
            )
            L_sdb = []
            R_sdb = []
            for line in ori_files.read_text(encoding="shift_jis").split("\n"):
                if not line:
                    L_sdb.append(line)
                    R_sdb.append(line)
                elif "L:" in line.split("\t")[3]:
                    L_sdb.append(line)
                else:
                    assert "R:" in line, line
                    R_sdb.append(line)
            (new_dir / f"{spk_id}-R.sdb").write_text(
                "\n".join(R_sdb), encoding="shift_jis"
            )
            (new_dir / f"{spk_id}-L.sdb").write_text(
                "\n".join(L_sdb), encoding="shift_jis"
            )
        else:
            (new_dir / f"{spk_id}.sdb").write_bytes(ori_files.read_bytes())
            wav = wav_dir / f"{spk_id}.wav"
            assert wav.is_file(), f"{spk_id}.wav cannot be found"
            (new_dir / f"{spk_id}-wav.list").write_text(wav.as_posix(), encoding="utf8")

    for ori_files in _A01M0056:
        ori_files = list(trans_dir.glob(f"*/{ori_files}/{ori_files}*"))

        for ori_file in ori_files:
            *same_part, vol, spk_id, filename = ori_file.as_posix().split("/")
            new_dir = Path("/".join(same_part + ["excluded", spk_id]))
            new_dir.mkdir(parents=True, exist_ok=True)
            ori_file.rename(new_dir / filename)
        ori_files[0].parent.rmdir()

    for i, eval_list in enumerate(_EVAL, start=1):
        for ori_files in eval_list:
            ori_files = list(trans_dir.glob(f"*/{ori_files}/{ori_files}*"))

            for ori_file in ori_files:
                *same_part, vol, spk_id, filename = ori_file.as_posix().split("/")
                new_dir = Path("/".join(same_part + [f"eval{i}", spk_id]))
                new_dir.mkdir(parents=True, exist_ok=True)
                ori_file.rename(new_dir / filename)
            ori_files[0].parent.rmdir()

    for ori_files in _VALID:
        ori_files = list(trans_dir.glob(f"*/{ori_files}/{ori_files}*"))

        for ori_file in ori_files:
            *same_part, vol, spk_id, filename = ori_file.as_posix().split("/")
            new_dir = Path("/".join(same_part + ["valid", spk_id]))
            new_dir.mkdir(parents=True, exist_ok=True)
            ori_file.rename(new_dir / filename)
        ori_files[0].parent.rmdir()

    (trans_dir / ".done_mv").touch()
    logging.info("Transcripts have been moved.")


# ----------------------------

INTERNAL_SEP = " "
_FIELDS = {
    "time": 3,
    "surface": 5,
    "notag": 9,
    "pos1": 11,
    "cForm": 12,
    "cType1": 13,
    "pos2": 14,
    "cType2": 15,
    "other": 16,
    "pron": 10,
    "spkid": 2,
}

_MORPH = ["pos1", "cForm", "cType2", "pos2"]

_REPLACEMENTS = [
    "<FV>",
    "<VN>",
    "<H>",
    "<Q>",
    "<笑>",
    "<咳>",
    "<息>",
    "<泣>",
    "<フロア発話>",
    "<フロア笑>",
    "<拍手>",
    "<デモ>",
    "<ベル>",
    "<朗読間違い>",
    "<雑音>",
]

DECISIONS = {
    "F": 0,
    "D": 0,
    "D2": 0,
    "?": 0,
    "?,": 0,
    "M": 0,
    "O": 0,
    "R": 0,
    "X": 0,
    "A": 1,
    "A_num": 0,
    "K": 1,
    "W": 1,
    "B": 0,
    "笑": 0,
    "泣": 0,
    "咳": 0,
    "L": 0,
}


class _CSJSDBWord:
    time = ""
    surface = ""
    notag = ""
    pos1 = ""
    cForm = ""
    cType1 = ""
    pos2 = ""
    cType2 = ""
    other = ""
    pron = ""
    spkid = ""
    sgid = 0
    start = -1.0
    end = -1.0
    morph = ""

    @staticmethod
    def from_line(line=""):
        word = _CSJSDBWord()
        line = line.strip().split("\t")

        for f, i in _FIELDS.items():
            try:
                attr = line[i]
            except IndexError:
                attr = ""
            setattr(word, f, attr)

        for _ in range(2):
            # Do twice in case of "んーー"
            for c, r in zip(["んー", "ンー"], ["ん", "ン"]):
                word.pron = word.pron.replace(c, r)
                word.surface = word.surface.replace(c, r)

        for r in _REPLACEMENTS:
            word.pron = word.pron.replace(r, "")
            word.surface = word.surface.replace(r, "")
        word.pron = word.pron.replace(INTERNAL_SEP, "_")
        word.surface = word.surface.replace(INTERNAL_SEP, "_")
        # This is for pauses <P:00453.373-00454.013>
        word.pron = re.sub(r"<PL.+>", "", word.pron)

        # Occurs for example in noncore/A01F0063:
        # 0099 00280.998-00284.221 L:-001-001	一・	一・	イチ	一	イチ
        word.surface = word.surface.rstrip("・")

        # Make morph
        morph = [getattr(word, s) for s in _MORPH]
        word.morph = "/".join(m for m in morph if m)
        for c in ["Ａ", "１", "２", "３", "４"]:
            word.morph = word.morph.replace(c, "")
        word.morph = word.morph.replace("　", "＿")

        # Parse time
        word.sgid, start_end, channel = word.time.split(" ")
        word.start, word.end = [float(s) for s in start_end.split("-")]
        if word.spkid[0] == "D":
            word.spkid = word.spkid + "-" + channel.split(":")[0]

        return word

    def __repr__(self):
        return f"{self.surface}+{self.morph}+{self.pron}"

    def __bool__(self):
        return bool(self.surface or self.pron)


class _Transcript:
    text: str = ""
    shape0: List[int]
    shape1: List[int]
    shape2: List[int]
    tag_end: Dict

    def __init__(self, segments: List, text_type: str):
        self.shape0 = []
        self.shape1 = []
        self.shape2 = []
        self.tag_end = {}
        self.right_offset = defaultdict(list)
        for i, s in enumerate(segments):
            for j, w in enumerate(s):
                w = getattr(w, text_type)
                self.text += w
                for k in range(len(w)):
                    self.shape0.append(i)
                    self.shape1.append(j)
                    self.shape2.append(k)

        open_brackets = []
        for i, s in enumerate(self.text):
            if s == "(":
                open_brackets.append(i)
            elif s == ")":
                self.tag_end[open_brackets.pop()] = i

    def use_index(self, pos: int, right=False) -> Tuple[int]:
        if not right:
            return (self.shape0[pos], self.shape1[pos], self.shape2[pos])

        adjust = 0
        for coords in self.right_offset[(self.shape0[pos], self.shape1[pos])]:
            if coords < self.shape2[pos]:
                adjust += 1

        self.right_offset[(self.shape0[pos], self.shape1[pos])].append(self.shape2[pos])

        return (self.shape0[pos], self.shape1[pos], self.shape2[pos] - adjust)


class _CSJSDBSegment:
    text: str
    start: float
    end: float
    sgid: str

    @staticmethod
    def from_words(words: List[_CSJSDBWord]):
        ret = _CSJSDBSegment()
        ret.text = INTERNAL_SEP.join(str(w) for w in words)
        ret.start = words[0].start
        ret.end = words[-1].end
        ret.sgid = f"{words[0].spkid}_{words[0].sgid}"
        return ret

    def __repr__(self):
        return self.text

    def to_line(self):
        return f"{self.sgid}\t" + f"{self.start:09.3f}\t{self.end:09.3f}\t" + self.text

    def verify_line(self) -> bool:
        return self.text.count("(") == self.text.count(")")

    @staticmethod
    def from_line(line: str):
        ret = _CSJSDBSegment()
        ret.sgid, ret.start, ret.end, text = line.strip().split("\t")
        ret.start = float(ret.start)
        ret.end = float(ret.end)
        ret.text = text
        return ret


class CSJSDBParser:

    tag_regex = re.compile(r"( )|([\x00-\x7F])")
    JPN_NUM = [
        "ゼロ",
        "０",
        "零",
        "一",
        "二",
        "三",
        "四",
        "五",
        "六",
        "七",
        "八",
        "九",
        "十",
        "百",
        "千",
        "．",
    ]

    def __default_preprocess(self, x: str):
        ret = []
        for a in x.split(INTERNAL_SEP):
            a = a.split("+")[0]
            if a:
                ret.append(a)
        return INTERNAL_SEP.join(ret)

    def __init__(self, decisions: Dict = DECISIONS, preprocess: Callable = None):
        self.decisions = decisions
        if not preprocess:
            self.preprocess: Callable[[str], str] = self.__default_preprocess
        else:
            self.preprocess = preprocess

    def parse(
        self, text: str, sep="", with_tags=False
    ) -> Union[str, List[Tuple[str, ...]]]:
        processed_text = self.preprocess(text)
        ret = self._parse(processed_text, -1)

        assert len(ret["string"]) == len(ret["tag"]), text
        if not with_tags:
            return ret["string"].replace(INTERNAL_SEP, sep)

        if not sep:
            return [
                (w, t) for w, t in zip(ret["string"], ret["tag"]) if w != INTERNAL_SEP
            ]
        return [
            (w, t) if w != INTERNAL_SEP else (sep, t)
            for w, t in zip(ret["string"], ret["tag"])
        ]

    def _parse(self, text: str, open_bracket):
        i = open_bracket + 1
        tag = ""
        choices = [""]
        choices_tag = [[]]

        while i < len(text):
            c = text[i]
            t = [tag]

            if c == "(":
                ret = self._parse(text, i)
                c = ret["string"]
                i = ret["end"]
                if not tag:
                    t = ret["tag"]
                else:
                    t = [tag + f"/{ta}" for ta in ret["tag"]]

            matches = self.tag_regex.search(c)

            if c == ")" and not tag:
                print(f"{open_bracket}, {i}, {choices}, {tag}, {text}")
                return {"string": choices[-1], "end": i, "tag": choices_tag[-1]}
            elif c == ")":
                if tag == "A" and choices[0] and choices[0][0] in self.JPN_NUM:
                    tag = "A_num"

                result, result_tag = self._decide(
                    tag, choices + [""], choices_tag + [[]]
                )
                return {"string": result, "end": i, "tag": result_tag}

            elif c == ";":
                choices.append("")
                choices_tag.append([])
            elif c == ",":
                choices.append("")
                choices_tag.append([])
                if "," not in tag:
                    tag += ","
            elif c == "_":
                pass
            elif matches and matches.group(2):
                tag += c
            elif not tag and open_bracket > -1 and c in ["笑", "泣", "咳"]:
                tag = c
            else:
                choices[-1] += c
                choices_tag[-1].extend(t)
            i += 1
        return {
            "string": choices[-1],
            "end": i,
            "tag": choices_tag[-1] if choices[-1] else [],
        }

    def _decide(self, tag, choices, choices_tag) -> Tuple[str, List[str]]:
        assert len(choices) > 1

        for t, decision in self.decisions.items():
            if t != tag:
                continue
            ret = -1
            if isinstance(decision, int):
                ret = choices[decision], choices_tag[decision]
            elif decision[:5] == "eval:":
                ret = eval(decision[5:])
            elif decision[:5] == "exec:":
                exec(decision[5:])

            if ret != -1:
                return ret
            else:
                raise Exception(
                    f"Decision for {tag} cannot be resolved. Got {decision}"
                )

        raise NotImplementedError(f"Unknown tag {tag} encountered.")


class _CSJSDBTagSegment:
    segments: List[List[_CSJSDBWord]]
    surface_open_brackets: Dict
    pron_open_brackets: Dict

    def __init__(self):
        self.segments = []
        self.surface_open_brackets = {}
        self.pron_open_brackets = {}

    def append(self, word: _CSJSDBWord):
        if self.segments:
            self.segments[-1].append(word)
        else:
            self.segments = [[word]]

    def flatten(self) -> _CSJSDBSegment:
        return _CSJSDBSegment.from_words(words=[w for s in self.segments for w in s])

    def split(self) -> List[_CSJSDBSegment]:
        return [_CSJSDBSegment.from_words(words=s) for s in self.segments if s]

    def __getitem__(self, pos):
        return self.segments[pos]

    @property
    def is_complete(self):
        surface = "".join(w.surface for s in self.segments for w in s)
        surface_open_brackets = self._get_open_brackets(surface)
        pron = "".join(w.pron for s in self.segments for w in s)
        pron_open_brackets = self._get_open_brackets(pron)

        if not surface_open_brackets and not pron_open_brackets:
            return True

        self.surface_open_brackets.update(
            {ii: surface[ii + 1] for ii in surface_open_brackets[::-1]}
        )

        self.pron_open_brackets.update(
            {ii: pron[ii + 1] for ii in pron_open_brackets[::-1]}
        )
        return False

    def _get_open_brackets(self, text) -> List[int]:
        brackets = []
        for i, s in enumerate(text):
            if s == "(":
                brackets.append(i)
            elif s == ")":
                brackets.pop()

        return brackets

    def __bool__(self):
        return bool(self.segments and self.segments[0])


def _read_one_sdb(sdb: Path) -> List[_CSJSDBSegment]:
    lines = sdb.read_text(encoding="shift_jis").split("\n")
    sgid = lines[0].split("\t")[3].split(" ")[0]

    words_until_now = _CSJSDBTagSegment()
    segments: List[_CSJSDBSegment] = []

    for line in lines:
        if not line:
            # Last line is always empty
            word = _CSJSDBWord()
        else:
            word = _CSJSDBWord.from_line(line)

        if not word and line:
            continue

        if word.sgid == sgid:
            words_until_now.append(word)
            continue

        sgid = word.sgid

        if not words_until_now.is_complete:
            words_until_now.segments.append([])
            pass
        elif not words_until_now:
            pass
        elif len(words_until_now.segments) > 1:
            surface = _Transcript(words_until_now, "surface")
            pron = _Transcript(words_until_now, "pron")

            for pos, linking_tag in words_until_now.pron_open_brackets.items():
                if linking_tag in ["R", "M", "O"]:
                    l0, l1, l2 = pron.use_index(pos)
                    r0, r1, r2 = pron.use_index(pron.tag_end[pos], True)
                    # r2 -= i
                    l1_p = words_until_now[l0][l1].pron
                    r1_p = words_until_now[r0][r1].pron
                    words_until_now[l0][l1].pron = l1_p[:l2] + l1_p[l2 + 3 :]
                    words_until_now[r0][r1].pron = r1_p[:r2] + r1_p[r2 + 1 :]

            split = True
            for pos, linking_tag in words_until_now.surface_open_brackets.items():
                if linking_tag in ["R", "M", "O"]:
                    l0, l1, l2 = surface.use_index(pos)
                    r0, r1, r2 = surface.use_index(surface.tag_end[pos], True)
                    # r2 -= i
                    l1_s = words_until_now[l0][l1].surface
                    r1_s = words_until_now[r0][r1].surface
                    words_until_now[l0][l1].surface = l1_s[:l2] + l1_s[l2 + 3 :]
                    words_until_now[r0][r1].surface = r1_s[:r2] + r1_s[r2 + 1 :]
                else:
                    split = False

            if split:
                segments.extend(words_until_now.split())
            else:
                segments.append(words_until_now.flatten())
            words_until_now = _CSJSDBTagSegment()
        else:
            segments.append(words_until_now.flatten())
            words_until_now = _CSJSDBTagSegment()

        words_until_now.append(word)

    return segments


def _process_one_recording(
    segments: List[_CSJSDBSegment],
    wav: Path,
    recording_id: str,
    parser: CSJSDBParser,
) -> Tuple[Recording, List[SupervisionSegment]]:
    recording = Recording.from_file(wav, recording_id=recording_id)

    supervision_segments = []

    for segment in segments:
        text = parser.parse(segment.text, sep="", with_tags=True)
        if not text:
            continue
        text, tag = list(zip(*text))
        text = "".join(text)
        tag = ",".join(tag)
        supervision_segments.append(
            SupervisionSegment(
                id=segment.sgid,
                recording_id=recording_id,
                start=segment.start,
                duration=(segment.end - segment.start),
                channel=0,
                language="Japanese",
                speaker=recording_id,
                gender=("Male" if recording_id[3] == "M" else "Female"),
                text=text,
                custom={"raw": segment.text, "disfluent": text, "disfluent_tag": tag},
            )
        )
    return recording, supervision_segments


def _process_one(sdb: Path, parser: CSJSDBParser):
    segments = _read_one_sdb(sdb)
    spk = sdb.stem
    try:
        wavfile = Path((sdb.parent / (spk + "-wav.list")).read_text())
        (sdb.parent / f"{sdb.stem}-trans.txt").write_text(
            "\n".join(s.to_line() for s in segments)
        )
    except FileNotFoundError:
        part = sdb.parent.name
        wavfile = sdb.parents[3] / (f"WAV/{part}/{spk}.wav")
        assert wavfile.exists()
    return _process_one_recording(segments, wavfile, spk, parser)


def prepare_manifests(
    transcript_dir: Path,
    dataset_parts: Union[str, Sequence[str]] = None,
    manifest_dir: Pathlike = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will
    simply read and return them.

    :param transcript_dir: Path, the path to the .sdb transcripts.
    :param dataset_parts: string or sequence of strings representing
        dataset part names, e.g. 'eval1', 'core', 'eval2'. This defaults to the
        full dataset - core, noncore, eval1, eval2, and eval3.
    :param manifest_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: int, to be passed to ThreadPoolExecutor.
    :return: a Dict whose key is the dataset part, and the value is Dicts
        with the keys 'recordings' and 'supervisions'.
    """
    assert (
        transcript_dir.is_dir()
    ), f"No such directory for transcript_dir: {transcript_dir}"
    if not dataset_parts:
        dataset_parts = _FULL_DATA_PARTS
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    glob_pattern = "*.sdb" if transcript_dir.name == "SDB" else "*/*.sdb"

    manifests = {}

    if manifest_dir:
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exit: we can read them and
        # save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts,
            output_dir=manifest_dir,
            prefix="csj",
        )
    parser = CSJSDBParser(DECISIONS)
    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            futures = []
            logging.info(f"Processing CSJ subset: {part}")
            if manifests_exist(part=part, output_dir=manifest_dir, prefix="csj"):
                logging.info(f"CSJ subset: {part} already prepared - skipping.")
                continue
            for sdb in transcript_dir.glob(f"{part}/{glob_pattern}"):
                futures.append(ex.submit(_process_one, sdb, parser))

            recordings = []
            supervisions = []
            for future in tqdm(futures, desc="Processing", leave=False):
                result: Tuple[Recording, List[SupervisionSegment]] = future.result()
                assert result
                recording, segments = result
                recordings.append(recording)
                supervisions.extend(segments)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            recording_set, supervision_set = fix_manifests(
                recording_set, supervision_set
            )
            validate_recordings_and_supervisions(recording_set, supervision_set)

            if manifest_dir:
                supervision_set.to_file(
                    manifest_dir / f"csj_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(manifest_dir / f"csj_recordings_{part}.jsonl.gz")

            manifests[part] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }
    return manifests


def prepare_csj(
    corpus_dir: Pathlike,
    transcript_dir: Pathlike = None,
    manifest_dir: Pathlike = None,
    dataset_parts: Union[str, Sequence[str]] = None,
    nj: int = 16,
):
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir()
    if transcript_dir:
        transcript_dir = Path(transcript_dir)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Creating transcript directories now.")
        _create_trans_dir(corpus_dir, transcript_dir)
    else:
        transcript_dir = corpus_dir / "MORPH" / "SDB"
        logging.info(
            "Preparing manifests without saving transcripts. Only core and "
            "noncore can be created. "
        )
        if not dataset_parts:
            dataset_parts = ["core", "noncore"]
    return prepare_manifests(
        transcript_dir=transcript_dir,
        dataset_parts=dataset_parts,
        manifest_dir=manifest_dir,
        num_jobs=nj,
    )


def concat_csj_supervisions(
    supervisions: SupervisionSet,
    gap: float,
    maxlen: float,
    max_extend_right=0.0,
) -> SupervisionSet:
    """Concatenates supervisions according to the permissible gap and maximum length
    of an utterance. This function is not called in this script, but is provided as
    a utility function for users to concatenate supervisions themselves.

    Args:
        supervisions (SupervisionSet): The list of `SupervisionSegments` to concatenate.
        gap (float): Maximum length of permissible silence in an utterance.
        maxlen (float): Maximum length of one utterance.
        max_extend_right (float, optional): Since the immediate right context is silence,
        optionally extends the supervision to include some silence on the right. Included
        for endpointing training. Defaults to 0.0.

    Returns:
        SupervisionSet: Concatenated supervisions
    """
    grouped_supervisions = []
    supervisions = copy.deepcopy(supervisions)
    tmp_sp = []
    for long_sp0 in supervisions:
        if "×" in long_sp0.custom["raw"]:
            if tmp_sp:
                grouped_supervisions.append(tmp_sp)
                tmp_sp = []
        elif not tmp_sp:
            tmp_sp.append(long_sp0)
        elif long_sp0.speaker != tmp_sp[0].speaker:
            grouped_supervisions.append(tmp_sp)
            tmp_sp = [long_sp0]
        elif (long_sp0.end - tmp_sp[0].start) >= maxlen:
            grouped_supervisions.append(tmp_sp)
            tmp_sp = [long_sp0]
        elif (long_sp0.start - tmp_sp[-1].end) >= gap:
            tmp_sp[-1].duration += min(
                max_extend_right, long_sp0.start - tmp_sp[-1].end
            )
            grouped_supervisions.append(tmp_sp)
            tmp_sp = [long_sp0]
        else:
            tmp_sp.append(long_sp0)
    if tmp_sp:
        grouped_supervisions.append(tmp_sp)
    ret = []
    for long_sp in grouped_supervisions:
        long_sp0 = long_sp[0]
        long_sp0.duration = long_sp[-1].end - long_sp0.start
        for k in long_sp0.custom:
            if k == "raw":
                long_sp0.custom[k] = " ".join(sp.custom[k] for sp in long_sp)
            elif "_tag" in k:
                long_sp0.custom[k] = ",".join(sp.custom[k] for sp in long_sp)
            else:
                long_sp0.custom[k] = "".join(sp.custom[k] for sp in long_sp)
        long_sp0.text = "".join(sp.text for sp in long_sp)

        ret.append(long_sp0)

    return SupervisionSet.from_segments(ret)
