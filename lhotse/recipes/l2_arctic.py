"""
The L2-ARCTIC corpus is a speech corpus of non-native English that is intended for research in voice conversion, accent conversion, and mispronunciation detection. In total, the corpus contains 26,867 utterances from 24 non-native speakers with a balanced gender and L1 distribution. Most speakers recorded the full [CMU ARCTIC set](http://festvox.org/cmu_arctic/). The total duration of the corpus is 27.1 hours, with an average of 67.7 minutes (std: 8.6 minutes) of speech per L2 speaker. On average, each utterance is 3.6 seconds in duration. The pause before and after each utterance is generally no longer than 100 ms. Using the forced alignment results, we estimate a speech to silence ratio of 7:1 across the whole dataset. The dataset contains over 238,702 word segments, giving an average of around nine (9) words per utterance, and over 851,830 phone segments (excluding silence).

Human annotators manually examined 3,599 utterances, annotating 14,098 phone substitutions, 3,420 phone deletions, and 1,092 phone additions.

Some speakers did not read all sentences, and a few sentences were removed for some speakers since those recordings did not have the required quality. We provide a list of those special cases in section **Notes**.

#### About the suitcase corpus (added on March 12, 2020)
This portion of the L2-ARCTIC corpus involves spontaneous speech. We include recordings and annotations from 22 of the 24 speakers who recorded the sentences. Speakers SKA and ASI did not participate in this task. Each speaker retold a story from a picture narrative used in applied linguistics research on comprehensibility, accentedness, and intelligibility. The pictures are generally known as the [suitcase story](https://www.iris-database.org/iris/app/home/detail?id=york:822279). Each retelling of the narrative was done after looking over the story and asking the researchers questions about what was happening. Few participants had questions. The annotations were carried out by two research assistants trained in phonetic transcription. Each did half of the transcriptions, then checked the other half done by the other research assistant. Finally, all transcriptions were checked by John Levis, a co-PI for the project. This project was funded by National Science Foundation award 1618953, titled “Developing Golden Speakers for Second-Language Pronunciation.”

The total duration of this subset is 26.1 minutes, with an average of 1.2 minutes (std: 41.5 seconds) per speaker. Using the manual annotation results, we estimate a speech to silence ratio of 2.3:1 across the whole dataset. The dataset contains around 3,083 word segments, giving an average of 140 words per recording, and around 9,458 phone segments (excluding silence). The manual annotations include 1,673 phone substitutions, 456 phone deletions, and 90 phone additions.

The corpus can be manually downloaded at https://psi.engr.tamu.edu/l2-arctic-corpus/

Note: Lhotse does not read the TextGrid files with word/phone alignment for now for this corpus.
"""

from os import makedirs
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike

SPEAKER_DESCRIPTION = """
|Speaker|Gender|Native Language|# Wav Files|# Annotations|
|---|---|---|---|---|
|ABA|M|Arabic|1129|150|
|SKA|F|Arabic|974|150|
|YBAA|M|Arabic|1130|149|
|ZHAA|F|Arabic|1132|150|
|BWC|M|Chinese|1130|150|
|LXC|F|Chinese|1131|150|
|NCC|F|Chinese|1131|150|
|TXHC|M|Chinese|1132|150|
|ASI|M|Hindi|1131|150|
|RRBI|M|Hindi|1130|150|
|SVBI|F|Hindi|1132|150|
|TNI|F|Hindi|1131|150|
|HJK|F|Korean|1131|150|
|HKK|M|Korean|1131|150|
|YDCK|F|Korean|1131|150|
|YKWK|M|Korean|1131|150|
|EBVS|M|Spanish|1007|150|
|ERMS|M|Spanish|1132|150|
|MBMPS|F|Spanish|1132|150|
|NJS|F|Spanish|1131|150|
|HQTV|M|Vietnamese|1132|150|
|PNV|F|Vietnamese|1132|150|
|THV|F|Vietnamese|1132|150|
|TLV|M|Vietnamese|1132|150|
|**Total**|||**26867**|**3599**|"""

# Un-used for now - might come useful later
PHONE_SET_DESCRIPTION = """
|Index|ARPAbet|Example|Annotation|Type|
|---|---|---|---|---|
|1|AA|odd|AA D|vowel|
|2|AE|at|AE T|vowel|
|3|AH|hut|HH AH T|vowel|
|4|AO|ought|AO T|vowel|
|5|AW|cow|K AW|vowel|
|6|AX|discus|D IH S K AX S|vowel|
|7|AY|hide|HH AY D|vowel|
|8|B|be|B IY|stop|
|9|CH|cheese|CH IY Z|affricate|
|10|D|dee|D IY|stop|
|11|DH|thee|DH IY|fricative|
|12|EH|Ed|EH D|vowel|
|13|ER|hurt|HH ER T|vowel|
|14|EY|ate|EY T|vowel|
|15|F|fee|F IY|fricative|
|16|G|green|G R IY N|stop|
|17|HH|he|HH IY|aspirate|
|18|IH|it|IH T|vowel|
|19|IY|eat|IY T|vowel|
|20|JH|gee|JH IY|affricate|
|21|K|key|K IY|stop|
|22|L|lee|L IY|liquid|
|23|M|me|M IY|nasal|
|24|N|knee|N IY|nasal|
|25|NG|ping|P IH NG|nasal|
|26|OW|oat|OW T|vowel|
|27|OY|toy|T OY|vowel|
|28|P|pee|P IY|stop|
|29|R|read|R IY D|liquid|
|30|S|sea|S IY|fricative|
|31|SH|she|SH IY|fricative|
|32|T|tea|T IY|stop|
|33|TH|theta|TH EY T AH|fricative|
|34|UH|hood|HH UH D|vowel|
|35|UW|two|T UW|vowel|
|36|V|vee|V IY|fricative|
|37|W|we|W IY|semivowel|
|38|Y|yield|Y IY L D|semivowel|
|39|Z|zee|Z IY|fricative|
|40|ZH|seizure|S IY ZH ER|fricative|"""


# TODO: Parse TextGrid files and create separate supervision sets
#       for word and phone alignments.


def prepare_l2_arctic(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepares and returns the L2 Arctic manifests which consist of Recordings and Supervisions.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a dict with keys "read" and "spontaneous".
        Each hold another dict of {'recordings': ..., 'supervisions': ...}
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    speaker_meta = _parse_speaker_description()

    recordings = RecordingSet.from_recordings(
        # Example ID: zhaa-arctic_b0126
        Recording.from_file(
            wav, recording_id=f"{wav.parent.parent.name.lower()}-{wav.stem}"
        )
        for wav in corpus_dir.rglob("*.wav")
    )
    supervisions = []
    for path in corpus_dir.rglob("*.txt"):
        # One utterance (line) per file
        text = path.read_text().strip()

        is_suitcase_corpus = "suitcase_corpus" in path.parts

        speaker = (
            path.parent.parent.name.lower()
        )  # <root>/ABA/transcript/arctic_a0051.txt -> aba
        if is_suitcase_corpus:
            speaker = path.stem  # <root>/suitcase_corpus/transcript/aba.txt -> aba

        seg_id = (
            f"suitcase_corpus-{speaker}"
            if is_suitcase_corpus
            else f"{speaker}-{path.stem}"
        )
        supervisions.append(
            SupervisionSegment(
                id=seg_id,
                recording_id=seg_id,
                start=0,
                duration=recordings[seg_id].duration,
                text=text,
                language="English",
                speaker=speaker,
                gender=speaker_meta[speaker]["gender"],
                custom={"accent": speaker_meta[speaker]["native_lang"]},
            )
        )
    supervisions = SupervisionSet.from_segments(supervisions)

    validate_recordings_and_supervisions(recordings, supervisions)

    splits = {
        "read": {
            "recordings": recordings.filter(lambda r: "suitcase_corpus" not in r.id),
            "supervisions": supervisions.filter(
                lambda s: "suitcase_corpus" not in s.recording_id
            ),
        },
        "suitcase": {
            "recordings": recordings.filter(lambda r: "suitcase_corpus" in r.id),
            "supervisions": supervisions.filter(
                lambda s: "suitcase_corpus" in s.recording_id
            ),
        },
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        makedirs(output_dir, exist_ok=True)
        for key, manifests in splits.items():
            manifests["recordings"].to_json(output_dir / f"recordings-{key}.json")
            manifests["supervisions"].to_json(output_dir / f"supervisions-{key}.json")

    return splits


def _parse_speaker_description():
    meta = {}
    for line in SPEAKER_DESCRIPTION.splitlines()[3:-1]:
        _, spk, gender, native_lang, *_ = line.split("|")
        meta[spk.lower()] = {"gender": gender, "native_lang": native_lang}
    return meta
