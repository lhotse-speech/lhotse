from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

from lhotse.custom import CustomFieldMixin


@dataclass
class TextExample(CustomFieldMixin):
    """
    Represents a single text example. Useful e.g. for language modeling.
    """

    text: str
    tokens: Optional[np.ndarray] = None
    custom: Optional[Dict[str, Any]] = None

    @property
    def num_tokens(self) -> Optional[int]:
        if self.tokens is None:
            return None
        return len(self.tokens)


@dataclass
class TextPairExample(CustomFieldMixin):
    """
    Represents a pair of text examples. Useful e.g. for sequence-to-sequence tasks.
    """

    source: TextExample
    target: TextExample
    custom: Optional[Dict[str, Any]] = None

    @property
    def num_tokens(self) -> Optional[int]:
        return self.source.num_tokens
