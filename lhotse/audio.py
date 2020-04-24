from dataclasses import dataclass
from typing import List


class AudioSet:
    pass


# TODO: maybe change the API to that

@dataclass(order=True, frozen=True)
class Recording:
    id: str
    paths: List[str]
    sampling_rate: int
    num_channels: int
    num_samples: int
    duration_seconds: float
