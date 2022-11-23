"""
The following is the hierarchy of imports in this module (to avoid circular imports):

      ┌─────────────┐
      │ __init__.py │─────────────┬────────────────────────────────────────────┐
      └─────────────┘             │                                            │
             │                    │                                            │
             │                    ▼                                            │
             │           ┌────────────────┐                                    │
             ├──────────▶│  mono.MonoCut  │────────────────────┐               │
             │           └────────────────┘                    │               │
             │                                                 ▼               │
             │           ┌────────────────┐           ┌────────────────┐       │
             ├──────────▶│ multi.MultiCut │──────────▶│  data.DataCut  │───────┤
             │           └────────────────┘           └────────────────┘       │
             │                                                 ▲               ▼
             │           ┌────────────────────┐                │        ┌─────────────┐
             ├──────────▶│   mixed.MixedCut   │────────────────┴───────▶│  base.Cut   │
             │           └────────────────────┘                         └─────────────┘
             │                      │                                          ▲
             │                      │                                          │
             │                      │         ┌────────────────────┐           │
             ├──────────────────────┴────────▶│ padding.PaddingCut │───────────┤
             │                                └────────────────────┘           │
    ┌────────────────┐                                   ▲                     │
    │   set.CutSet   │───────────────────────────────────┴─────────────────────┘
    └────────────────┘

"""

from .base import Cut
from .mixed import MixedCut, MixTrack
from .mono import MonoCut
from .multi import MultiCut
from .padding import PaddingCut
from .set import (
    CutSet,
    append_cuts,
    compute_supervisions_frame_mask,
    create_cut_set_eager,
    create_cut_set_lazy,
    mix_cuts,
)

# The following functions are imported in other modules, so we need to import them here.
__all__ = [
    "Cut",
    "CutSet",
    "MixedCut",
    "MixTrack",
    "MonoCut",
    "MultiCut",
    "PaddingCut",
    "create_cut_set_eager",
    "create_cut_set_lazy",
    "compute_supervisions_frame_mask",
    "append_cuts",
    "mix_cuts",
]
