import random
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import soundfile

from lhotse import AudioSource, Cut, CutSet, Fbank, LilcomFilesWriter, Recording, SupervisionSegment
from lhotse.utils import uuid4


def random_cut_set(n_cuts=100) -> CutSet:
    return CutSet.from_cuts(
        Cut(
            id=uuid4(),
            start=round(random.uniform(0, 5), ndigits=8),
            duration=round(random.uniform(3, 10), ndigits=8),
            channel=0
        ) for _ in range(n_cuts)
    )


class RandomCutTestCase:
    def setup_method(self, method):
        self.files = []
        self.dirs = []

    def teardown_method(self, method):
        self.cleanup()

    def cleanup(self):
        for f in self.files:
            f.close()
        self.files = []
        for d in self.dirs:
            d.cleanup()
        self.dirs = []

    def with_recording(self, sampling_rate: int, num_samples: int) -> Recording:
        f = NamedTemporaryFile('wb', suffix='.wav')
        self.files.append(f)
        duration = num_samples / sampling_rate
        samples = np.random.rand(num_samples)
        soundfile.write(f.name, samples, samplerate=sampling_rate)
        return Recording(
            id=str(uuid4()),
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

    def with_cut(
            self,
            sampling_rate: int,
            num_samples: int,
            features: bool = True,
            supervision: bool = False
    ) -> Cut:
        duration = num_samples / sampling_rate
        cut = Cut(
            id=str(uuid4()),
            start=0,
            duration=duration,
            channel=0,
            recording=self.with_recording(sampling_rate=sampling_rate, num_samples=num_samples)
        )
        if features:
            cut = self._with_features(cut)
        if supervision:
            cut.supervisions.append(SupervisionSegment(
                id=f'sup-{cut.id}',
                recording_id=cut.recording_id,
                start=0,
                duration=cut.duration,
                text='irrelevant'
            ))
        return cut

    def _with_features(self, cut: Cut) -> Cut:
        d = TemporaryDirectory()
        self.dirs.append(d)
        with LilcomFilesWriter(d.name) as storage:
            return cut.compute_and_store_features(Fbank(), storage=storage)