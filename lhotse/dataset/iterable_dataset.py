import warnings

import torch

from lhotse.dataset.sampling.base import CutSampler


class IterableDatasetWrapper(torch.utils.data.IterableDataset):
    """
    A wrapper that creates an iterable-style dataset out of a map-style dataset and a
    :class:`~lhotse.dataset.sampling.base.CutSampler`.

    It's intended use is for training with cuts (:class:`~lhotse.cut.Cut`) that contain
    binary data instead of paths/urls/keys pointing to an external storage.
    Currently, we support this kind of dataloading using WebDataset library
    (to export a :class:`~lhotse.CutSet` into WebDataset tarball format,
    see :func:`lhotse.dataset.webdataset.export_to_webdataset`).

    .. caution: With an iterable-style dataset, the sampler replicas exist in the
        dataloading worker subprocesses. That means unless you take extra steps to avoid
        data duplication, it may happen that each worker returns exactly the same data.
        This problem is avoided with WebDataset by using sharding -- we let WebDataset
        subset the shards visible in each subprocess (and each node in multi-GPU DDP training).

    Example usage::

        >>> from lhotse import CutSet
        >>> from lhotse.dataset import K2SpeechRecognitionDataset, DynamicCutSampler
        >>> # Preparing data -- WebDataset takes care of sharding and de-duplicating data
        >>> cuts = CutSet.from_webdataset(
        ...     "data/shard-{000000..000321}.tar",
        ...     shuffle_shards=True,
        ...     split_on_workers=True,
        ...     split_on_nodes=True,
        ... )
        >>> dataset = K2SpeechRecognitionDataset()
        >>> sampler = DynamicCutSampler(cuts, max_duration=200, shuffle=True)
        >>> # Creating terable dataset wrapper
        >>> iter_dset = IterableDatasetWrapper(dataset, sampler)
        >>> dloader = torch.utils.data.DataLoader(iter_dset, batch_size=None, num_workers=2)
        >>> # Training loop
        >>> for epoch in range(10):
        ...     dloader.dataset.set_epoch(epoch)
        ...     for batch in dloader:
        ...         pass  # training step
    """

    def __init__(self, dataset: torch.utils.data.Dataset, sampler: CutSampler) -> None:
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler

        rank = self.sampler.rank
        ws = self.sampler.world_size
        if rank != 0 or ws != 1:
            warnings.warn(
                f"We detected you're trying to use a CutSampler with rank {rank} and world_size {ws} "
                f"inside an IterableDatasetWrapper. Setting rank != 0 and world_size != 1 in Lhotse's "
                f"CutSampler is inteded for map-style datasets, when the sampler exists in the main "
                f"training loop. Make sure these settings are intentional or pass rank=0 and world_size=1 "
                f"to the sampler's constructor.\n"
            )

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

        # The code below is for WebDataset-powered CutSet. We have to set the epoch like
        # this so that the shards become shuffled. Maybe there is a cleaner way of doing this.
        if hasattr(self.sampler, "cuts") and isinstance(self.sampler.cuts, tuple):
            for cs in self.sampler.cuts:
                if hasattr(cs.data, "set_epoch"):
                    cs.data.set_epoch(epoch)

    def __iter__(self):
        self._sampler_iter = iter(self.sampler)
        return self

    def __next__(self) -> dict:
        return self.dataset[next(self._sampler_iter)]
