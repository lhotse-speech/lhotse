from pytest import param, mark

from lhotse.audio import AudioSet


@mark.parametrize(
    'channel',
    [
        None,
        0,
        1,
        param(1000, marks=mark.xfail)
    ]
)
def test_audio_set_get_audio_multichannel(channel):
    audio_set = AudioSet()
    audio_set.get_audio('recording-id-1', channel=channel)


def test_audio_set_get_audio_from_multiple_files():
    audio_set = AudioSet()
    audio_set.get_audio('recording-id-1')


def test_audio_set_iteration():
    audio_set = AudioSet()
    assert ['recording-id-1'] == list(audio_set)


@mark.parametrize(
    ['begin_at', 'end_at'],
    [
        (None, None),
        (1.5, None),
        (None, 4.0),
        (1.5, 4.0),
        param(4.0, 1.5, marks=mark.xfail)
    ]
)
def test_audio_set_get_audio_chunks(begin_at, end_at):
    audio_set = AudioSet()
    audio_set.get_audio('recording-id-1', begin_at_seconds=begin_at, end_at_seconds=end_at)


def test_audio_set_get_metadata():
    audio_set = AudioSet()
    audio_set.get_number_of_channels('recording-id-1')
    audio_set.get_sampling_rate('recording-id-1')
    audio_set.get_number_of_samples('recording-id-1')
    audio_set.get_duration('recording-id-1')


def test_audio_set_serialization():
    audio_set = AudioSet()
    audio_set.to_yaml('.test.yaml')
    deserialized = AudioSet.from_yaml('.test.yaml')
    assert deserialized == audio_set
