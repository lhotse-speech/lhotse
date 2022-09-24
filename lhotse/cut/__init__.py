from .base import Cut
from .mono import MonoCut
from .mixed import MixedCut, MixTrack
from .padding import PaddingCut
from .set import CutSet

"""
The following is the hierarchy of imports in this module (to avoid circular imports):

      ┌─────────────┐
      │ __init__.py │─────────────────────────────────┬────────────────────┐
      └─────────────┘                                 │                    │
             │                                        ▼                    │
             │                               ┌────────────────┐            │
             ├──────────────────────┬───────▶│  mono.MonoCut  │────────────┤
             │                      │        └────────────────┘            │
             │                      │                                      │
             │                      │                                      │
             │           ┌────────────────────┐                            ▼
             │           │   mixed.MixTrack   │                     ┌─────────────┐
             ├──────────▶│   mixed.MixedCut   │────────────────────▶│  base.Cut   │
             │           └────────────────────┘                     └─────────────┘
             │                      │                                      ▲
             │                      │                                      │
             │                      │                                      │
             │                      │        ┌────────────────────┐        │
             ├──────────────────────┴───────▶│ padding.PaddingCut │────────┤
             │                               └────────────────────┘        │
    ┌────────────────┐                                  ▲                  │
    │   set.CutSet   │──────────────────────────────────┴──────────────────┘
    └────────────────┘
"""

# The following functions are imported in other modules, so we need to import them here.
__all__ = [
    "create_cut_set_eager",
    "create_cut_set_lazy",
    "compute_supervisions_frame_mask",
    "append_cuts",
    "mix_cuts",
]

from .set import (
    create_cut_set_eager,
    create_cut_set_lazy,
    compute_supervisions_frame_mask,
    append_cuts,
    mix_cuts,
)
