"""
Corpus owner: https://clrd.ninjal.ac.jp/csj/en/index.html
Corpus description:
- http://www.lrec-conf.org/proceedings/lrec2000/pdf/262.pdf
- https://isca-speech.org/archive_open/archive_papers/sspr2003/sspr_mmo2.pdf

This script accesses the CSJ/SDB/{core,noncore} directories and generates
transcripts in accordance to the tag decisions defined in the .ini file.
If no .ini file is provided, it reverts to default settings and produces
1 transcript per segment.

This script does the following in sequence:-

**MOVE**
1. Copies each .sdb files from /SDB into its own directory in the designated
  `trans_dir`, i.e. {trans_dir}/{spk_id}/{spk_id}.sdb
2. Verifies that the corresponding wav file exists in the /WAV directory, and
   outputs that absolute path into {spk_id}-wav.list
3. Moves the predefined datasets for eval1, eval2, eval3, and excluded, into
   its own dataset directory
4. Touches a .done_mv in `trans_dir`.
NOTE: If a .done_mv exists already in `trans_dir`, then this stage is skipped.

**PARSE**
1. Takes in an .ini file which - among others - contains the behaviour for each
   tag and the segment details.
2. Parses all .sdb files it can find within `trans_dir`, and optionally outputs
   a segment file.
3. Touches a .done in `trans_dir`.

**PREPARE MANIFESTS**
1. Globs through all transcript files and generates supervisions and
    recordings manifests for each dataset part.

Differences to kaldi include:-
1. The evaluation datasets do not follow `trans_dir`/eval/eval{i}, but are
   instead saved in the same level as core, noncore, and excluded.
2. Morphology tags are parsed but not included in the final transcript. The
   original morpheme segmentations are preserved by spacing, i.e. 分かち書き,
   so removal, if required, has to be done at a later stage.
3. Kana pronunciations are parsed but not included in the final transcript.

The transcript directory will have this structure:-
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

"""

import logging
import os
import re
from concurrent.futures.thread import ThreadPoolExecutor
from configparser import ConfigParser
from copy import copy
from io import TextIOWrapper
from multiprocessing import Queue, get_context
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

# ---------------------------------- #
#           DEFAULT_CONFIG           #

DEFAULT_CONFIG = """
[SEGMENTS]
; # Allowed period of nonverbal noise. If exceeded, a new segment is created.
gap = 0.5
; # Maximum length of segment (s).
maxlen = 10
; # Minimum length of segment (s). Segments shorter than `minlen` will be dropped silently.
minlen = 0.02
; # Use this symbol to represent a period of allowed nonverbal noise, i.e. `gap`.
; # Pass an empty string to avoid adding any symbol. It was "<sp>" in kaldi.
; # If you intend to use a multicharacter string for gap_sym, remember to register the
; # multicharacter string as part of userdef-string in prepare_lang_char.py.
gap_sym =

[CONSTANTS]
; # Name of this mode
MODE = disfluent
; # Suffixes to use after the word surface (no longer used)
MORPH = pos1 cForm cType2 pos2
; # Used to differentiate between A tag and A_num tag
JPN_NUM = ゼロ ０ 零 一 二 三 四 五 六 七 八 九 十 百 千 ．
; # Dummy character to delineate multiline words
PLUS = ＋

[DECISIONS]
; # TAG+'^'とは、タグが一つの転記単位に独立していない場合
; # The PLUS (fullwidth) sign '＋' marks line boundaries for multiline entries # noqa

; # フィラー、感情表出系感動詞
; # 0 to remain, 1 to delete
; # Example: '(F ぎょっ)'
F = 0
; # Example: '(L (F ン))', '比べ(F えー)る'
F^ = 0
; # 言い直し、いいよどみなどによる語断片
; # 0 to remain, 1 to delete
; # Example: '(D だ)(D だいが) 大学の学部の会議'
D = 0
; # Example: '(L (D ドゥ)＋(D ヒ))'
D^ = 0
; # 助詞、助動詞、接辞の言い直し
; # 0 to remain, 1 to delete
; # Example: '西洋 (D2 的)(F えー)(D ふ) 風というか'
D2 = 0
; # Example: '(X (D2 ノ))'
D2^ = 0
; # 聞き取りや語彙の判断に自信がない場合
; # 0 to remain, 1 to delete
; # Example: (? 字数) の
; # If no option: empty string is returned regardless of output
; # Example: '(?) で'
? = 0
; # Example: '(D (? すー))＋そう＋です＋よ＋ね'
?^ = 0
; # タグ?で、値は複数の候補が想定される場合
; # 0 for main guess with matching morph info, 1 for second guess
; # Example:  '(? 次数, 実数)', '(? これ,ここで)＋(? 説明＋し＋た＋方＋が＋いい＋か＋な)' # noqa
?, = 0
; # Example: '(W (? テユクー);(? ケッキョク,テユウコトデ))', '(W マシ;(? マシ＋タ,マス))' # noqa
?,^ = 0
; # 音や言葉に関するメタ的な引用
; # 0 to remain, 1 to delete
; # Example: '助詞の (M は) は (M は) と書くが発音は (M わ)'
M = 0
; # Example: '(L (M ヒ)＋(M ヒ))', '(L (M (? ヒ＋ヒ)))'
M^ = 0
; # 外国語や古語、方言など
; # 0 to remain, 1 to delete
; # Example: '(O ザッツファイン)'
O = 0
; # Example: '(笑 (O エクスキューズ＋ミー))', '(笑 メダッ＋テ＋(O ナンボ))' # noqa
O^ = 0
; # 講演者の名前、差別語、誹謗中傷など
; # 0 to remain, 1 to delete
; # Example: '国語研の (R ××) です'
R = 0
R^ = 0
; # 非朗読対象発話（朗読における言い間違い等）
; # 0 to remain, 1 to delete
; # Example: '(X 実際は) 実際には'
X = 0
; # Example: '(L (X (D2 ニ)))'
X^ = 0
; # アルファベットや算用数字、記号の表記
; # 0 to use Japanese form, 1 to use alphabet form
; # Example: '(A シーディーアール;ＣＤ－Ｒ)'
A = 1
; # Example: 'スモール(A エヌ;Ｎ)', 'ラージ(A キュー;Ｑ)',
; # '(A ティーエフ;ＴＦ)＋(A アイディーエフ;ＩＤＦ)'
; # (Strung together by pron: '(W (? ティーワイド);ティーエフ＋アイディーエフ)') # noqa
A^ = 1
; # タグAで、単語は算用数字の場合
; # 0 to use Japanese form, 1 to use Arabic numerals
; # Example: (A 二千;２０００)
A_num = eval:self.notag
A_num^ = eval:self.notag
; # 何らかの原因で漢字表記できなくなった場合
; # 0 to use broken form, 1 to use orthodox form
; # Example: '(K たち (F えー) ばな;橘)'
K = 1
; # Example: '合(K か(?)く;格)', '宮(K ま(?)え;前)'
K^ = 1
; # 転訛、発音の怠けなど、一時的な発音エラー
; # 0 to use wrong form, 1 to use orthodox form
; # Example: '(W ギーツ;ギジュツ)'
W = 1
; # Example: '(F (W エド;エト))', 'イベント(W リレーティッド;リレーテッド)'
W^ = 1
; # 語の読みに関する知識レベルのいい間違い
; # 0 to use wrong form, 1 to use orthodox form
; # Example: '(B シブタイ;ジュータイ)'
B = 0
; # Example: 'データー(B カズ;スー)'
B^ = 0
; # 笑いながら発話
; # 0 to remain, 1 to delete
; # Example: '(笑 ナニガ)', '(笑 (F エー)＋ソー＋イッ＋タ＋ヨー＋ナ)'
笑 = 0
; # Example: 'コク(笑 サイ＋(D オン))',
笑^ = 0
; # 泣きながら発話
; # 0 to remain, 1 to delete
; # Example: '(泣 ドンナニ)'
泣 = 0
泣^ = 0
; # 咳をしながら発話
; # 0 to remain, 1 to delete
; # Example: 'シャ(咳 リン) ノ'
咳 = 0
; # Example: 'イッ(咳 パン)', 'ワズ(咳 カ)'
咳^ = 0
; # ささやき声や独り言などの小さな声
; # 0 to remain, 1 to delete
; # Example: '(L アレコレナンダッケ)', '(L (W コデ;(? コレ,ココデ))＋(? セツメー＋シ＋タ＋ホー＋ガ＋イー＋カ＋ナ))'
L = 0
; # Example: 'デ(L ス)', 'ッ(L テ＋コ)ト'
L^ = 0

[REPLACEMENTS]
; # ボーカルフライなどで母音が同定できない場合
<FV> =
; # 「うん/うーん/ふーん」の音の特定が困難な場合
<VN> =
; # 非語彙的な母音の引き延ばし
<H> =
; # 非語彙的な子音の引き延ばし
<Q> =
; # 言語音と独立に講演者の笑いが生じている場合
<笑> =
; # 言語音と独立に講演者の咳が生じている場合
<咳> =
; # 言語音と独立に講演者の息が生じている場合
<息> =
; # 講演者の泣き声
<泣> =
; # 聴衆（司会者なども含む）の発話
<フロア発話> =
; # 聴衆の笑い
<フロア笑> =
; # 聴衆の拍手
<拍手> =
; # 講演者が発表中に用いたデモンストレーションの音声
<デモ> =
; # 学会講演に発表時間を知らせるためにならすベルの音
<ベル> =
; # 転記単位全体が再度読み直された場合
<朗読間違い> =
; # 上記以外の音で特に目立った音
<雑音> =
; # 0.2秒以上のポーズ
<P> =
; # Redacted information, for R
; # It is \x00D7 multiplication sign, not your normal 'x'
× = ×

[FIELDS]
; # Time information for segment
time = 3
; # Word surface
surface = 5
; # Word surface root form without CSJ tags
notag = 9
; # Part Of Speech
pos1 = 11
; # Conjugated Form
cForm = 12
; # Conjugation Type
cType1 = 13
; # Subcategory of POS
pos2 = 14
; # Euphonic Change / Subcategory of Conjugation Type
cType2 = 15
; # Other information
other = 16
; # Pronunciation for lexicon
pron = 10
; # Speaker ID
spk_id = 2

[KATAKANA2ROMAJI]
ア = 'a
イ = 'i
ウ = 'u
エ = 'e
オ = 'o
カ = ka
キ = ki
ク = ku
ケ = ke
コ = ko
ガ = ga
ギ = gi
グ = gu
ゲ = ge
ゴ = go
サ = sa
シ = si
ス = su
セ = se
ソ = so
ザ = za
ジ = zi
ズ = zu
ゼ = ze
ゾ = zo
タ = ta
チ = ti
ツ = tu
テ = te
ト = to
ダ = da
ヂ = di
ヅ = du
デ = de
ド = do
ナ = na
ニ = ni
ヌ = nu
ネ = ne
ノ = no
ハ = ha
ヒ = hi
フ = hu
ヘ = he
ホ = ho
バ = ba
ビ = bi
ブ = bu
ベ = be
ボ = bo
パ = pa
ピ = pi
プ = pu
ペ = pe
ポ = po
マ = ma
ミ = mi
ム = mu
メ = me
モ = mo
ヤ = ya
ユ = yu
ヨ = yo
ラ = ra
リ = ri
ル = ru
レ = re
ロ = ro
ワ = wa
ヰ = we
ヱ = wi
ヲ = wo
ン = ŋ
ッ = q
ー = -
キャ = kǐa
キュ = kǐu
キョ = kǐo
ギャ = gǐa
ギュ = gǐu
ギョ = gǐo
シャ = sǐa
シュ = sǐu
ショ = sǐo
ジャ = zǐa
ジュ = zǐu
ジョ = zǐo
チャ = tǐa
チュ = tǐu
チョ = tǐo
ヂャ = dǐa
ヂュ = dǐu
ヂョ = dǐo
ニャ = nǐa
ニュ = nǐu
ニョ = nǐo
ヒャ = hǐa
ヒュ = hǐu
ヒョ = hǐo
ビャ = bǐa
ビュ = bǐu
ビョ = bǐo
ピャ = pǐa
ピュ = pǐu
ピョ = pǐo
ミャ = mǐa
ミュ = mǐu
ミョ = mǐo
リャ = rǐa
リュ = rǐu
リョ = rǐo
ァ = a
ィ = i
ゥ = u
ェ = e
ォ = o
ヮ = ʍ
ヴ = vu
ャ = ǐa
ュ = ǐu
ョ = ǐo
"""

# ---------------------------------- #
# Prepare manifests from transcripts #

ORI_DATA_PARTS = (
    "eval1",
    "eval2",
    "eval3",
    "core",
    "noncore",
)


def parse_transcript_header(line: str):
    sgid, start, end, line = line.split(" ", maxsplit=3)
    return (sgid, float(start), float(end), line)


def parse_one_recording(
    template: Path, wavlist_path: Path, recording_id: str
) -> Tuple[Recording, List[SupervisionSegment]]:
    transcripts = []

    for trans in template.glob(f"{recording_id}*.txt"):
        trans_type = trans.stem.replace(recording_id + "-", "")
        transcripts.append(
            [(trans_type, t) for t in Path(trans).read_text().split("\n")]
        )

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
                duration=(end - start),
                channel=0,
                language="Japanese",
                speaker=recording_id,
                gender=("Male" if recording_id[3] == "M" else "Female"),
                text=text,
                custom=customs,
            )
        )

    return recording, supervision_segments


def prepare_manifests(
    transcript_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = None,
    manifest_dir: Pathlike = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will
    simply read and return them.

    :param transcript_dir: Pathlike, the path to the transcripts.
        Assumes that that the transcripts were processed by
        csj_make_transcript.py.
    :param dataset_parts: string or sequence of strings representing
        dataset part names, e.g. 'eval1', 'core', 'eval2'. This defaults to the
        full dataset - core, noncore, eval1, eval2, and eval3.
    :param manifest_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts
        with the keys 'recordings' and 'supervisions'.
    """

    transcript_dir = Path(transcript_dir)
    assert (
        transcript_dir.is_dir()
    ), f"No such directory for transcript_dir: {transcript_dir}"

    if not dataset_parts:
        dataset_parts = ORI_DATA_PARTS

    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

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

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            logging.info(f"Processing CSJ subset: {part}")
            if manifests_exist(part=part, output_dir=manifest_dir, prefix="csj"):
                logging.info(f"CSJ subset: {part} already prepared - skipping.")
                continue

            recordings = []
            supervisions = []
            part_path = transcript_dir / part
            futures = []

            for wavlist in part_path.glob("*/*-wav.list"):
                spk = wavlist.name.rstrip("-wav.list")
                template = wavlist.parent

                futures.append(ex.submit(parse_one_recording, template, wavlist, spk))

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                assert result
                recording, segments = result
                recordings.append(recording)
                supervisions.extend(segments)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)
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


# ---------------------------------- #
#    Prepare transcripts from SDB    #


FULL_DATA_PARTS = ["core", "noncore", "eval1", "eval2", "eval3", "excluded"]

# Exclude speaker ID
A01M0056 = [
    "S05M0613",
    "R00M0187",
    "D01M0019",
    "D04M0056",
    "D02M0028",
    "D03M0017",
]

# Evaluation set ID
EVAL = [
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

# https://stackoverflow.com/questions/23589174/regex-pattern-to-match-excluding-when-except-between # noqa
tag_regex = re.compile(r"(<|>|\+)|([\x00-\x7F])")


def kana2romaji(katakana: str) -> str:
    if not KATAKANA2ROMAJI or not katakana:
        return katakana

    tmp = []
    mem = ""

    for c in katakana[::-1]:
        if c in ["ャ", "ュ", "ョ", "ァ", "ィ", "ゥ", "ェ", "ォ", "ヮ"]:
            mem += c
            continue
        if mem:
            c += mem
            mem = ""

        try:
            tmp.append(KATAKANA2ROMAJI[c])
        except KeyError:
            for i in c[::-1]:
                try:
                    tmp.append(KATAKANA2ROMAJI[i])
                except KeyError:
                    tmp.append(i)

    if mem:
        try:
            tmp.append(KATAKANA2ROMAJI[mem])
        except KeyError:
            for i in mem[::-1]:
                try:
                    tmp.append(KATAKANA2ROMAJI[i])
                except KeyError:
                    tmp.append(i)

    return "".join(tmp[::-1])


# -------------------------- #
class CSJSDB_Word:

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
    spk_id = ""
    sgid = 0
    start = -1.0
    end = -1.0
    morph = ""
    words = []

    @staticmethod
    def from_line(line=""):
        word = CSJSDB_Word()
        line = line.strip().split("\t")

        for f, i in FIELDS.items():
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

        # Make morph
        morph = [getattr(word, s) for s in MORPH]
        word.morph = "/".join(m for m in morph if m)
        for c in ["Ａ", "１", "２", "３", "４"]:
            word.morph = word.morph.replace(c, "")
        word.morph = word.morph.replace("　", "＿")
        if word.morph:
            word.morph = "+" + word.morph

        # Parse time
        word.sgid, start_end, channel = word.time.split(" ")
        word.start, word.end = [float(s) for s in start_end.split("-")]
        if word.spk_id[0] == "D":
            word.spk_id = word.spk_id + "-" + channel.split(":")[0]

        return word

    @staticmethod
    def from_dict(other: Dict):
        word = CSJSDB_Word()
        for k, v in other.items():
            setattr(word, k, v)

    def _parse_pron(self):
        for tag, replace in REPLACEMENTS_PRON.items():
            self.pron = self.pron.replace(tag, replace)

        # This is for pauses <P:00453.373-00454.013>
        self.pron = re.sub(r"<P:.+>", REPLACEMENTS_PRON["<P>"], self.pron)
        matches = tag_regex.findall(self.pron)
        if all(not g2 for _, g2 in matches):
            return self.pron
        elif self.pron.count("(") != self.pron.count(")"):
            return None

        open_brackets = [pos for pos, c in enumerate(self.pron) if c == "("]
        close_brackets = [pos for pos, c in enumerate(self.pron) if c == ")"]

        if open_brackets[0] > close_brackets[-1]:
            return None

        pron = self._parse(-1, self.pron, "p")["string"]
        return pron

    def _parse_surface(self):
        for ori, replace in REPLACEMENTS_SURFACE.items():
            self.surface = self.surface.replace(ori, replace)
        # Occurs for example in noncore/A01F0063:
        # 0099 00280.998-00284.221 L:-001-001	一・	一・	イチ	一	イチ
        self.surface = self.surface.rstrip("・")
        # This is for pauses <P:00453.373-00454.013>
        self.surface = re.sub(r"<P:.+>", REPLACEMENTS_SURFACE["<P>"], self.surface)
        matches = tag_regex.findall(self.surface)
        if all(not g2 for _, g2 in matches):
            return self.surface
        elif self.surface.count("(") != self.surface.count(")"):
            return None

        open_brackets = [i for i, c in enumerate(self.surface) if c == "("]
        close_brackets = [i for i, c in enumerate(self.surface) if c == ")"]

        if open_brackets[0] > close_brackets[-1]:
            return None
        surface = self._parse(-1, self.surface, "s")["string"]
        return surface

    def _decide(self, tag, choices, ps) -> str:
        assert len(choices) > 1
        decisions = DECISIONS_PRON if ps == "p" else DECISIONS_SURFACE
        for t, decision in decisions.items():
            if t != tag:
                continue
            ret = -1
            if isinstance(decision, int):
                ret = choices[decision]
            else:
                if decision[:5] == "eval:":
                    ret = eval(decision[5:])
                elif decision[:5] == "exec:":
                    exec(decision[5:])
                else:
                    ret = PLUS.join(decision for _ in range(choices[0].count(PLUS) + 1))

            if ret != -1:
                return ret

        raise NotImplementedError(f"Unknown tag {tag} encountered")

    def __bool__(self):
        ret = bool(self.surface and self.pron)
        return ret

    def __eq__(self, other: "CSJSDB_Word"):
        return (
            self.surface == other.surface
            and self.pron == other.pron
            and self.morph == other.morph
        )

    def __repr__(self):
        return self.to_lexicon(" ")

    def __hash__(self):
        return hash(self.__repr__())

    def _parse_pronsurface(self) -> bool:
        new_pron = self._parse_pron()
        new_surface = self._parse_surface()

        if new_pron is not None:
            self.pron = new_pron

        if new_surface is not None:
            self.surface = new_surface
            self.notag = new_surface

        if new_pron is not None and new_surface is not None:
            return True
        else:
            return False

    def _parse(self, open_bracket: int, text: str, ps: str) -> Dict:
        assert ps in ["p", "s"]
        result = ""
        mem = ""
        i = open_bracket + 1
        tag = ""
        choices = [""]
        long_multiline = text.count(PLUS) > 5  # HARDCODE ALERT

        while i < len(text):
            c = text[i]

            if c == "(":
                ret = self._parse(i, text, ps)
                c = ret["string"]
                i = ret["end"]
            mem += c
            matches = tag_regex.search(c)

            if c == ")" and not tag:
                return {"string": mem, "end": i}

            elif c == ")":
                if tag == "A" and choices[0] and choices[0][0] in JPN_NUM:
                    tag = "A_num"

                if open_bracket and not long_multiline:
                    tag += "^"

                result += self._decide(
                    tag, choices + [PLUS * choices[0].count(PLUS)], ps
                )
                return {"string": result, "end": i}
            elif c == ";":
                choices.append("")
            elif c == ",":
                choices.append("")
                if "," not in tag:
                    tag += ","
            elif c == " ":
                pass
            elif matches and matches.group(2):
                tag += c
            elif not tag and open_bracket > -1 and c in ["笑", "泣", "咳"]:
                tag = c
            else:
                choices[-1] = choices[-1] + c
            i += 1

        return {"string": mem, "end": i}

    def to_lexicon(self, separator="\t"):
        return f"{self.surface}{self.morph}{separator}{self.pron}"

    def to_transcript(self):
        return f"{self.surface}{self.morph}"

    def convert_pron(self):
        self.pron = kana2romaji(self.pron)
        if "+" not in self.pron:
            self.pron = tuple(self.pron)
        else:
            self.pron = (self.pron,)

    @staticmethod
    def from_file(fin: TextIOWrapper) -> List["CSJSDB_Word"]:
        """Reads an SDB file and outputs a list of `CSJSDB_Word`
        nodes.

        Returns:
            List['CSJSDB_Word]: A list of CSJSDB_Word
        """

        ret: List[CSJSDB_Word] = []
        mem = None
        for line in fin:
            w = CSJSDB_Word.from_line(line)
            is_complete_word = w._parse_pronsurface()

            if mem is not None:
                mem._add_word(w)
                # assert len(mem.words) < 50  or 'R' in mem.surface
                if mem._parse_pronsurface():
                    mem = mem._resolve_multiline()

                    ret.extend(mem)
                    mem = None
            elif is_complete_word and not w:
                continue
            elif is_complete_word:
                # assert all(p not in w.pron for p in ['(', ')', 'x'])
                ret.append(w)
            else:
                mem = w

        for word in ret:
            assert all(
                p not in word.surface for p in ["(", ")", ";"]
            ), f"surface {word.surface} contains invalid character. {fin.name}"

            assert all(
                p not in word.pron for p in ["(", ")", ";"]
            ), f"pron {word.pron} contains invalid character. {fin.name}"

            word.convert_pron()

        return ret

    def _add_word(self, w: "CSJSDB_Word"):
        if not self.words:
            self.words = [copy(self)]

        self.words.append(w)
        try:
            del w.words
        except AttributeError:
            pass
        self.__dict__.update(w.__dict__)

        self.surface = PLUS.join(ww.surface for ww in self.words)
        self.notag = PLUS.join(ww.notag for ww in self.words)
        self.pron = PLUS.join(ww.pron for ww in self.words)
        self.start = self.words[0].start
        self.end = self.words[-1].end

    def _resolve_multiline(self):
        # Only called when trying to resolve a multiline CSJSDB_Word object.
        split_surface = PLUS in self.surface
        split_pron = PLUS in self.pron
        ret = []

        if split_surface and split_pron:
            assert split_pron
            surfaces = self.surface.split(PLUS)
            prons = self.pron.split(PLUS)
            len_words = len(self.words)
            surfaces = surfaces + [""] * (len_words - len(surfaces))
            prons = prons + [""] * (len_words - len(prons))

            for s, p, i in zip(surfaces, prons, range(len_words)):
                self.words[i].surface = s
                self.words[i].pron = p

            ret = [w for w in self.words if w]
        elif not self:
            pass
        elif split_surface and not split_pron:
            self.surface = re.sub(r"\+<.+>|" + PLUS, "", self.surface)
            ret = [self]
        elif not split_surface and split_pron:
            self.pron = re.sub(r"\+<.+>|" + PLUS, "", self.pron)
            ret = [self]
        else:
            self.surface = re.sub(r"\+<.+>", "", self.surface)
            self.pron = re.sub(r"\+<.+>", "", self.pron)
            ret = [self]
        del self.words
        return ret


def modify_text(
    word_list: List[CSJSDB_Word], segments: List[str], gap_sym: str, gap: float
) -> List[Dict[str, List[str]]]:
    """Takes a list of parsed CSJSDB words and a list of time boundaries for
        each segment, and outputs them in transcript format

    Args:
        word_list (List[CSJSDB_Word]): List of parsed words from CSJ SDB
        segments (List[str]): List of time boundaries for each segment
        gap (float): Permissible period of nonverbal noise. If exceeded, a new
            segment is created.
        gap_sym (str): Use this symbol to represent gap, if nonverbal noise
            does not exceed `gap`. Pass
                        an empty string to avoid adding any symbol.

    Returns:
        List[Dict[str, List[str]]]: A list of maximum two elements.
                If len == 2, first element is Left channel, and second element
                    is Right channel
                Available fields:

                'spk_id': the speaker ID, including the trailing 'L' and 'R' if
                    two-channeled
                'text': the output text

    """

    last_end = word_list[0].start

    segments_ = []
    for s in segments:
        sgid, start, end = s.split()
        start = float(start)
        end = float(end)
        segments_.append((sgid, start, end))
    segments = segments_.copy()
    line_sgid, line_start, line_end = segments_.pop(0)

    single_char_gap = "⨋"
    out = []
    line = []
    tobreak = False
    for word in word_list:

        if word.spk_id not in line_sgid:
            continue
        elif word.start < line_start:
            continue
        elif "×" in word.surface:
            continue
        elif word.end <= line_end:
            if gap_sym and gap < (word.start - last_end):
                line.append(gap_sym)
            line.append(word.surface)
        else:  # word.end > line_end
            line = " ".join(line).replace(single_char_gap, gap_sym)
            out.append(f"{line_sgid} {line_start:09.3f} {line_end:09.3f} " + line)

            try:
                line_sgid, line_start, line_end = segments_.pop(0)
            except IndexError:
                line = []
                tobreak = True
                break

            if word.spk_id not in line_sgid:
                continue

            while word.start >= line_end:
                out.append(f"{line_sgid} {line_start:09.3f} {line_end:09.3f} ")
                try:
                    line_sgid, line_start, line_end = segments_.pop(0)
                except IndexError:
                    line = []
                    tobreak = True
                    break
            if tobreak:
                break
            line = [word.surface]

        last_end = word.end

    if not tobreak:
        line = " ".join(line).replace(single_char_gap, gap_sym)
        # assert '×' not in line
        out.append(f"{line_sgid} {line_start:09.3f} {line_end:09.3f} " + line)

    while segments_:
        line_sgid, line_start, line_end = segments_.pop(0)
        out.append(f"{line_sgid} {line_start:09.3f} {line_end:09.3f} ")

    return {"text": out, "spk_id": line_sgid[:-5], "segments": segments}


def make_text(
    word_list: List[CSJSDB_Word],
    gap: float,
    maxlen: float,
    minlen: float,
    gap_sym: str,
) -> List[Dict[str, List[str]]]:
    """Takes a list of parsed CSJSDB words and outputs them as transcript

    Args:
        word_list (List[CSJSDB_Word]): List of parsed words from CSJ SDB
        gap (float): Permissible period of nonverbal noise. If exceeded, a new
            segment is created.
        maxlen (float): Maximum length of the segment.
        minlen (float): Minimum length of the segment. Segments shorter than
            this will be silently dropped.
        gap_sym (str): Use this symbol to represent gap, if nonverbal noise
            does not exceed `gap`. Pass
                        an empty string to avoid adding any symbol.

    Returns:
        List[Dict[str, List[str]]]: A list of maximum two elements.
                If len == 2, first element is Left channel, and second element
                    is Right channel
                Available fields:

                'spk_id': the speaker ID, including the trailing 'L' and 'R' if
                    two-channeled
                'text': the output text

    """

    line_sgid = word_list[0].sgid
    line_spk_id = word_list[0].spk_id
    line_start = word_list[0].start
    last_sgid = word_list[0].sgid
    last_spk_id = word_list[0].spk_id
    last_end = word_list[0].start

    out = []
    line = []
    segments = []

    single_char_gap = "⨋"

    for word in word_list:

        if last_sgid == word.sgid and last_spk_id == word.spk_id:
            line.append(word.surface)
        elif (
            gap < (word.start - last_end)
            or maxlen < (last_end - line_start)
            or line_spk_id != word.spk_id
        ):

            line = " ".join(line).replace(single_char_gap, gap_sym)
            if minlen < (last_end - line_start) and "×" not in line:
                out.append(
                    f"{line_spk_id}_{line_sgid} {line_start:09.3f} "
                    f"{last_end:09.3f} " + line
                )
                segments.append((f"{line_spk_id}_{line_sgid}", line_start, last_end))

            line_start = word.start
            line_sgid = word.sgid
            line_spk_id = word.spk_id
            line = [word.surface]
        elif gap_sym:
            line.extend([single_char_gap, word.surface])
        else:
            line.append(word.surface)

        last_sgid = word.sgid
        last_spk_id = word.spk_id
        last_end = word.end

    line = " ".join(line).replace(single_char_gap, gap_sym)
    if line and "×" not in line:
        out.append(
            f"{line_spk_id}_{line_sgid} "
            f"{line_start:09.3f} "
            f"{last_end:09.3f} " + line
        )
        segments.append((f"{line_spk_id}_{line_sgid}", line_start, last_end))

    if last_spk_id[-1] not in ["R", "L"]:
        return [{"text": out, "spk_id": last_spk_id, "segments": segments}]
    else:
        out = _tear_apart_LR(out, segments)
        spk_id = last_spk_id[:-2]
        return [
            {
                "text": out["out_L"],
                "spk_id": spk_id + "-L",
                "segments": out["segment_L"],
            },
            {
                "text": out["out_R"],
                "spk_id": spk_id + "-R",
                "segments": out["segment_R"],
            },
        ]


def _tear_apart_LR(lines: List[str], segments: List[Tuple]):
    out_R = []
    out_L = []
    segment_R = []
    segment_L = []

    for line, segment in zip(lines, segments):
        spkid = line.split("_", maxsplit=1)[0]
        if spkid[-1] == "R":
            out_R.append(line)
            segment_R.append(segment)
        else:
            out_L.append(line)
            segment_L.append(segment)

    return {
        "out_R": out_R,
        "out_L": out_L,
        "segment_R": segment_R,
        "segment_L": segment_L,
    }


def create_trans_dir(corpus_dir: Path, trans_dir: Path):

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
        (new_dir / f"{spk_id}.sdb").write_bytes(ori_files.read_bytes())
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

        else:
            wav = wav_dir / f"{spk_id}.wav"
            assert wav.is_file(), f"{spk_id}.wav cannot be found"
            (new_dir / f"{spk_id}-wav.list").write_text(wav.as_posix(), encoding="utf8")

    for ori_files in A01M0056:
        ori_files = list(trans_dir.glob(f"*/{ori_files}/{ori_files}*"))

        for ori_file in ori_files:
            *same_part, vol, spk_id, filename = ori_file.as_posix().split("/")
            new_dir = Path("/".join(same_part + ["excluded", spk_id]))
            new_dir.mkdir(parents=True, exist_ok=True)
            ori_file.rename(new_dir / filename)
        ori_files[0].parent.rmdir()

    for i, eval_list in enumerate(EVAL):
        i += 1
        for ori_files in eval_list:
            ori_files = list(trans_dir.glob(f"*/{ori_files}/{ori_files}*"))

            for ori_file in ori_files:
                *same_part, vol, spk_id, filename = ori_file.as_posix().split("/")
                new_dir = Path("/".join(same_part + [f"eval{i}", spk_id]))
                new_dir.mkdir(parents=True, exist_ok=True)
                ori_file.rename(new_dir / filename)
            ori_files[0].parent.rmdir()

    (trans_dir / ".done_mv").touch()
    logging.info("Transcripts have been moved.")


def parse_sdb_process(
    jobs_queue: Queue,
    gap: float,
    maxlen: float,
    minlen: float,
    gap_sym: str,
    trans_mode: str,
    use_segments: bool,
    write_segments: bool,
):
    def parse_one_sdb(sdb: Path):
        with sdb.open("r", encoding="shift_jis") as fin:
            result = CSJSDB_Word.from_file(fin)

        if not use_segments:
            transcripts = make_text(result, gap, maxlen, minlen, gap_sym)
        else:
            channels = (
                ["-L-segments", "-R-segments"] if sdb.name[0] == "D" else ["-segments"]
            )
            transcripts = []
            for channel in channels:
                segments = Path(sdb.as_posix()[:-4] + channel).read_text().split("\n")
                assert segments, segments
                transcripts.append(modify_text(result, segments, "", 0.5))

        for transcript in transcripts:
            spk_id = transcript.pop("spk_id")
            segments = transcript.pop("segments")
            (sdb.parent / f"{spk_id}-{trans_mode}.txt").write_text(
                "\n".join(transcript["text"]), encoding="utf8"
            )
            if write_segments:
                (sdb.parent / f"{spk_id}-segments").write_text(
                    "\n".join(f"{s[0]} {s[1]} {s[2]}" for s in segments),
                    encoding="utf8",
                )

    while True:
        job = jobs_queue.get()
        if not job:
            break
        parse_one_sdb(sdb=job)


def load_config(config_file: str):
    config = ConfigParser()
    config.optionxform = str
    config.read_string(config_file)
    # fmt: off
    global PLUS, DECISIONS_PRON, DECISIONS_SURFACE, REPLACEMENTS_PRON, \
        REPLACEMENTS_SURFACE, MORPH, FIELDS, JPN_NUM, KATAKANA2ROMAJI
    # fmt: on
    PLUS = config["CONSTANTS"]["PLUS"]
    MORPH = config["CONSTANTS"]["MORPH"].split()
    JPN_NUM = config["CONSTANTS"]["JPN_NUM"].split()
    DECISIONS_PRON = {}
    for k, v in config["DECISIONS"].items():
        try:
            DECISIONS_PRON[k] = int(v)
        except ValueError:
            DECISIONS_PRON[k] = v

    DECISIONS_SURFACE = DECISIONS_PRON.copy()
    REPLACEMENTS_PRON = {}
    for k, v in config["REPLACEMENTS"].items():
        REPLACEMENTS_PRON[k] = v
    REPLACEMENTS_SURFACE = REPLACEMENTS_PRON.copy()
    FIELDS = {}
    for k, v in config["FIELDS"].items():
        FIELDS[k] = int(v)
    KATAKANA2ROMAJI = {}
    for k, v in config["KATAKANA2ROMAJI"].items():
        KATAKANA2ROMAJI[k] = v

    return config


def prepare_transcripts(
    corpus_dir: Path,
    transcript_dir: Path,
    config: str,
    nj: int,
    write_segments: bool,
    use_segments: bool,
):
    if (transcript_dir / ".done").exists():
        logging.info(
            f"{transcript_dir} already parsed. "
            f"Delete {transcript_dir / '.done'} to parse again."
        )
        return

    config = load_config(config)
    trans_mode = config["CONSTANTS"]["MODE"]

    assert corpus_dir.is_dir()

    segment_config = config["SEGMENTS"]
    gap = float(segment_config["gap"])
    maxlen = float(segment_config["maxlen"])
    minlen = float(segment_config["minlen"])
    gap_sym = segment_config["gap_sym"]

    Process = get_context("fork").Process
    num_jobs = min(nj, os.cpu_count())
    maxsize = 10 * num_jobs

    jobs_queue = Queue(maxsize=maxsize)

    workers: List[Process] = []

    for _ in range(num_jobs):
        worker = Process(
            target=parse_sdb_process,
            args=(
                jobs_queue,
                gap,
                maxlen,
                minlen,
                gap_sym,
                trans_mode,
                use_segments,
                write_segments,
            ),
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    num_sdb = 0
    logging.info(f"Gathering sdbs to be parsed in {trans_mode} mode now.")
    for sdb in transcript_dir.glob("*/*/*.sdb"):
        jobs_queue.put(sdb)
        num_sdb += 1

    logging.info(f"Parsing found {num_sdb} sdbs now.")
    # signal termination
    for _ in workers:
        jobs_queue.put(None)

    # wait for workers to terminate
    for w in workers:
        w.join()

    logging.info("All done.")
    (transcript_dir / ".done").touch()


def prepare_csj(
    corpus_dir: Pathlike,
    transcript_dir: Pathlike,
    manifest_dir: Pathlike = None,
    dataset_parts: Union[str, Sequence[str]] = None,
    configs: Sequence[str] = None,
    nj: int = 16,
):
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir()
    transcript_dir = Path(transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # Move SDBs to transcript directory
    logging.info("Creating transcript directories now.")
    create_trans_dir(corpus_dir, transcript_dir)

    # Parse transcript
    if not configs:
        prepare_transcripts(
            corpus_dir=corpus_dir,
            transcript_dir=transcript_dir,
            config=DEFAULT_CONFIG,
            nj=nj,
            write_segments=True,
            use_segments=False,
        )
    else:
        for i, config in enumerate(configs):
            config = Path(config).read_text(encoding="utf8")
            prepare_transcripts(
                corpus_dir=corpus_dir,
                transcript_dir=transcript_dir,
                config=config,
                nj=nj,
                write_segments=(not bool(i)),
                use_segments=bool(i),
            )

    # Prepare manifests
    return prepare_manifests(
        transcript_dir=transcript_dir,
        dataset_parts=dataset_parts,
        manifest_dir=manifest_dir,
        num_jobs=nj,
    )
