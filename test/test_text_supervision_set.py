from lhotse.text import TextSupervisionSet


def test_text_supervision_set():
    text_set = TextSupervisionSet()
    text_set.get_utterance('utt_id')
