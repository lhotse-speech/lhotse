from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile

from lhotse import AudioSource, Cut, Recording


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
