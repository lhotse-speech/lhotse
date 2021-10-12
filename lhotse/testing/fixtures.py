import random
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, List

import numpy as np

from lhotse import (
    AudioSource,
    CutSet,
    Fbank,
    FbankConfig,
    LilcomFilesWriter,
    MonoCut,
    Recording,
    SupervisionSegment,
)
from lhotse.supervision import AlignmentItem
from lhotse.utils import Seconds, uuid4


def random_cut_set(n_cuts=100) -> CutSet:
    return CutSet.from_cuts(
        MonoCut(
            id=uuid4(),
            start=round(random.uniform(0, 5), ndigits=8),
            duration=round(random.uniform(3, 10), ndigits=8),
            channel=0,
            recording=Recording(
                id=uuid4(),
                sources=[],
                sampling_rate=16000,
                num_samples=1600000,
                duration=100.0,
            ),
        )
        for _ in range(n_cuts)
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
        import soundfile

        f = NamedTemporaryFile("wb", suffix=".wav")
        self.files.append(f)
        duration = num_samples / sampling_rate
        samples = np.random.rand(num_samples)
        soundfile.write(f.name, samples, samplerate=sampling_rate)
        return Recording(
            id=str(uuid4()),
            sources=[AudioSource(type="file", channels=[0], source=f.name)],
            sampling_rate=sampling_rate,
            num_samples=num_samples,
            duration=duration,
        )

    def with_cut(
        self,
        sampling_rate: int,
        num_samples: int,
        features: bool = True,
        supervision: bool = False,
        alignment: bool = False,
        frame_shift: Seconds = 0.01,
    ) -> MonoCut:
        duration = num_samples / sampling_rate
        cut = MonoCut(
            id=str(uuid4()),
            start=0,
            duration=duration,
            channel=0,
            recording=self.with_recording(
                sampling_rate=sampling_rate, num_samples=num_samples
            ),
        )
        if features:
            cut = self._with_features(cut, frame_shift=frame_shift)
        if supervision:
            cut.supervisions.append(
                SupervisionSegment(
                    id=f"sup-{cut.id}",
                    recording_id=cut.recording_id,
                    start=0,
                    duration=cut.duration,
                    text="irrelevant",
                    alignment=self._with_alignment(cut, "irrelevant")
                    if alignment
                    else None,
                )
            )
        return cut

    def _with_features(self, cut: MonoCut, frame_shift: Seconds) -> MonoCut:
        d = TemporaryDirectory()
        self.dirs.append(d)
        extractor = Fbank(config=FbankConfig(frame_shift=frame_shift))
        with LilcomFilesWriter(d.name) as storage:
            return cut.compute_and_store_features(extractor, storage=storage)

    def _with_alignment(
        self, cut: MonoCut, text: str
    ) -> Dict[str, List[AlignmentItem]]:
        subwords = [
            text[i : i + 3] for i in range(0, len(text), 3)
        ]  # Create subwords of 3 chars
        dur = cut.duration / len(subwords)
        alignment = [
            AlignmentItem(symbol=sub, start=i * dur, duration=dur)
            for i, sub in enumerate(subwords)
        ]
        return {"subword": alignment}
