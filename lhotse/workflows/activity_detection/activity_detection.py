from typing import List

from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment


def detect_activity(recording: Recording, model) -> List[SupervisionSegment]:
    # TODO: work in parallel here
    return model(recording)
