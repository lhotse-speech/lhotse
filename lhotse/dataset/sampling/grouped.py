# Copyright (c) 2025, The Lhotse Project. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

from lhotse import CutSet
from lhotse.dataset.sampling.base import CutSampler
from lhotse.dataset.sampling.data_source import DataSource


class GroupedCutSampler(CutSampler):
    """
    Yields all consecutive cuts that share the same value of a custom attribute as a
    single mini-batch (a :class:`~lhotse.cut.CutSet`).

    This sampler is designed to be used after
    :meth:`~lhotse.cut.CutSet.cut_into_overlapping_windows`, which tags every sub-cut
    with ``custom["source_cut_id"]``.  All sub-cuts originating from the same parent
    cut will appear consecutively in the stream and will be emitted as one batch, so
    the downstream :class:`~torch.utils.data.Dataset` receives all chunks of a single
    audio window at once.

    Example usage::

        >>> cuts = CutSet.from_file("cuts.jsonl")
        >>> cuts = cuts.cut_into_windows(duration=3600, hop=3600)
        >>> cuts = cuts.cut_into_overlapping_windows(min_duration=30, max_duration=40, overlap=1)
        >>> sampler = GroupedCutSampler(cuts, group_by="source_cut_id")
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=None)

    .. note::
        The sampler relies on the assumption that all cuts belonging to the same group
        appear **consecutively** in the CutSet.  This is guaranteed when
        ``cut_into_overlapping_windows`` is applied lazily (the default), because every
        parent cut's sub-cuts are emitted together before moving to the next parent.

    :param cuts: the :class:`~lhotse.cut.CutSet` to iterate over.
    :param group_by: the key to look up inside ``cut.custom`` to determine group
        membership.  Defaults to ``"source_cut_id"``.
    :param world_size: total number of distributed nodes (auto-detected if not given).
    :param rank: index of the current distributed node (auto-detected if not given).
    :param seed: random seed (not used for shuffling here, but kept for API consistency).
    """

    def __init__(
        self,
        cuts: CutSet,
        group_by: str = "source_cut_id",
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__(
            drop_last=False,
            shuffle=False,
            world_size=world_size,
            rank=rank,
            seed=seed,
        )
        self.group_by = group_by
        self.data_source = DataSource(cuts)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["group_by"] = self.group_by
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.group_by = state_dict.pop("group_by", self.group_by)
        super().load_state_dict(state_dict)
        iter(self.data_source)

    def __iter__(self) -> "GroupedCutSampler":
        if self._just_restored_state:
            return self
        self.diagnostics.reset_current_epoch()
        iter(self.data_source)
        return self

    def _next_batch(self) -> CutSet:
        cuts = []
        current_group = None

        while True:
            try:
                cut = next(self.data_source)
            except StopIteration:
                if cuts:
                    return CutSet.from_cuts(cuts)
                raise StopIteration()

            if not self._filter_fn(cut):
                self.diagnostics.discard_single(cut)
                continue

            group_val = (cut.custom or {}).get(self.group_by)

            if current_group is None:
                current_group = group_val

            if group_val != current_group:
                # Encountered a new group — put this cut back and return the batch.
                self.data_source.take_back(cut)
                break

            cuts.append(cut)

        return CutSet.from_cuts(cuts)
