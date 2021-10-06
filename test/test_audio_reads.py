import pytest

from lhotse import Recording


@pytest.mark.parametrize('path', [
    'test/fixtures/mono_c0.wav',
    'test/fixtures/mono_c1.wav',
    'test/fixtures/stereo.wav',
    'test/fixtures/libri/libri-1088-134315-0000.wav',
    'test/fixtures/mono_c0.opus',
    'test/fixtures/stereo.opus',
    'test/fixtures/stereo.sph',
    'test/fixtures/stereo.mp3',
    'test/fixtures/common_voice_en_651325.mp3',
])
def test_info_and_read_audio_consistency(path):
    recording = Recording.from_file(path)
    audio = recording.load_audio()
    assert audio.shape[1] == recording.num_samples
