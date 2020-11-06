from contextlib import contextmanager
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
import soundfile

from lhotse import AudioSource, Cut, Fbank, LilcomFilesWriter, Recording


@contextmanager
def make_recording(sampling_rate: int, num_samples: int) -> Recording:
    # The idea is that we're going to write to a temporary file with a sine wave recording
    # of specified duration and sampling rate, and clean up only after the test is executed.
    with NamedTemporaryFile('wb', suffix='.wav') as f:
        duration = num_samples / sampling_rate
        samples: np.ndarray = np.sin(2 * np.pi * np.arange(0, num_samples) / sampling_rate)
        soundfile.write(f, samples, samplerate=sampling_rate)
        yield Recording(
            id=f'recording-{sampling_rate}-{duration}',
            sources=[
                AudioSource(
                    type='file',
                    channels=[0],
                    source=f.name
                )
            ],
            sampling_rate=sampling_rate,
            num_samples=num_samples,
            duration=duration
        )


@contextmanager
def make_cut(sampling_rate: int, num_samples: int) -> Cut:
    with make_recording(sampling_rate, num_samples) as recording:
        duration = num_samples / sampling_rate
        yield Cut(
            id=f'cut-{sampling_rate}-{duration}',
            start=0,
            duration=duration,
            channel=0,
            recording=recording
        )


@pytest.mark.parametrize(
    ['sampling_rate', 'num_samples'],
    [
        (16000, 15995),
        (16000, 15996),
        (16000, 15997),
        (16000, 15998),
        (16000, 15999),
        (16000, 16000),
        (16000, 16001),
        (16000, 16002),
        (16000, 16003),
        (16000, 16004),
        (16000, 16005),
    ]
)
def test_simple_cut_num_frames_and_samples_are_consistent(sampling_rate, num_samples):
    with make_cut(sampling_rate, num_samples) as cut, \
            TemporaryDirectory() as dir, \
            LilcomFilesWriter(dir) as storage:
        cut = cut.compute_and_store_features(
            extractor=Fbank(),
            storage=storage
        )
        feats = cut.load_features()
        samples = cut.load_audio()

        assert cut.has_features
        assert feats.shape[0] == cut.features.num_frames
        assert feats.shape[0] == cut.num_frames
        assert feats.shape[1] == cut.features.num_features
        assert feats.shape[1] == cut.num_features

        assert cut.has_recording
        assert samples.shape[0] == 1
        assert samples.shape[1] == cut.recording.num_samples
        assert samples.shape[1] == cut.num_samples


@pytest.mark.parametrize(
    ['sampling_rate', 'num_samples', 'padded_duration'],
    [
        (16000, 15995, 1.5),
        (16000, 15996, 1.5),
        (16000, 15997, 1.5),
        (16000, 15998, 1.5),
        (16000, 15999, 1.5),
        (16000, 16000, 1.5),
        (16000, 16001, 1.5),
        (16000, 16002, 1.5),
        (16000, 16003, 1.5),
        (16000, 16004, 1.5),
        (16000, 16005, 1.5),
    ]
)
def test_padded_cut_num_frames_and_samples_are_consistent(sampling_rate, num_samples, padded_duration):
    with make_cut(sampling_rate, num_samples) as cut, \
            TemporaryDirectory() as dir, \
            LilcomFilesWriter(dir) as storage:
        cut = cut.compute_and_store_features(
            extractor=Fbank(),
            storage=storage
        )
        cut = cut.pad(padded_duration)
        feats = cut.load_features()
        samples = cut.load_audio()

        assert cut.has_features
        assert feats.shape[0] == cut.num_frames
        assert feats.shape[1] == cut.num_features

        assert cut.has_recording
        assert samples.shape[0] == 1
        assert samples.shape[1] == cut.num_samples
