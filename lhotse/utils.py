from pathlib import Path
from typing import Union

Pathlike = Union[Path, str]

INT16MAX = 32768


class DummySet:
    def __contains__(self, item):
        return True

    def intersection(self, iterable):
        return True
