import pytest

from lhotse.audio import AudioSource


def is_smart_open_installed():
    try:
        import smart_open
        return True
    except:
        return False


@pytest.mark.skipif(not is_smart_open_installed(), reason='URL downloading requires smart_open to be installed.')
def test_audio_url_downloading():
    audio_source = AudioSource(
        type='url',
        channels=[0],
        source='https://github.com/lhotse-speech/lhotse/blob/master/test/fixtures/mono_c0.wav?raw=true'
    )
    audio_source.load_audio()
