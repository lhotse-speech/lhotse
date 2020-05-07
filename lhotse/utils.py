from pathlib import Path
from typing import Union

Pathlike = Union[Path, str]

Seconds = float
Milliseconds = float

INT16MAX = 32768


class SetContainingAnything:
    def __contains__(self, item):
        return True

    def intersection(self, iterable):
        return True
