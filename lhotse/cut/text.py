from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

from lhotse.custom import CustomFieldMixin


@dataclass
class TextExample(CustomFieldMixin):
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
    source_text: str
    target_text: str
    source_tokens: Optional[np.ndarray] = None
    target_tokens: Optional[np.ndarray] = None
    custom: Optional[Dict[str, Any]] = None

    @property
    def num_tokens(self) -> Optional[int]:
        if self.source_tokens is None:
            return None
        return len(self.source_tokens)

    @property
    def num_target_tokens(self) -> Optional[int]:
        if self.target_tokens is None:
            return None
        return len(self.target_tokens)
