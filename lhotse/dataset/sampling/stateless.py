import logging
import os
import random
import secrets
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

from lhotse import CutSet, Seconds
from lhotse.cut.set import deserialize_cut
from lhotse.dataset import DynamicBucketingSampler, DynamicCutSampler
from lhotse.dataset.sampling.base import SamplingDiagnostics
from lhotse.lazy import Dillable
from lhotse.serialization import decode_json_line
from lhotse.utils import Pathlike

PathlikeAndScale = Tuple[Pathlike, float]


class StatelessSampler(torch.utils.data.Sampler, Dillable):
    """
    An infinite and stateless cut sampler that selects data at random from one or more cut manifests.
    The main idea is to make training resumption easy while guaranteeing the data seen each
    time by the model is shuffled differently.
    It discards the notion of an "epoch" and it never finishes iteration.
    It makes no strong guarantees about avoiding data duplication, but in practice you would
    rarely see duplicated data.

    The recommended way to use this sampler is by placing it into a dataloader worker
    subprocess with Lhotse's :class:``~lhotse.dataset.iterable_dataset.IterableDatasetWrapper``.
    This will guarantee each worker uses a different random seed
    (otherwise it will also be OK as long as your OS provides a TRNG)::

        >>> import torch
        >>> import lhotse
        >>> dloader = torch.utils.data.DataLoader(
        ...     lhotse.dataset.iterable_dataset.IterableDatasetWrapper(
        ...         dataset=lhotse.dataset.K2SpeechRecognitionDataset(...),
        ...         sampler=StatelessSampler(...),
        ...     ),
        ...     batch_size=None,
        ...     num_workers=4,
        ... )

    This sampler's design was originally proposed by Dan Povey. For details see:
    https://github.com/lhotse-speech/lhotse/issues/1096

    Example 1: Get a non-bucketing :class:``.StatelessSampler``::

        >>> sampler = StatelessSampler(
        ...     cuts_paths=["data/cuts_a.jsonl", "data/cuts_b.jsonl"],
        ...     index_path="data/files.idx",
        ...     max_duration=600.0,
        ... )

    Example 2: Get a bucketing :class:``.StatelessSampler``::

        >>> sampler = StatelessSampler(
        ...     cuts_paths=["data/cuts_a.jsonl", "data/cuts_b.jsonl"],
        ...     index_path="data/files.idx",
        ...     max_duration=600.0,
        ...     num_buckets=50,
        ...     quadratic_duration=30.0,
        ... )

    Example 3: Get a bucketing :class:``.StatelessSampler`` with scaled weights for each cutset::

        >>> sampler = StatelessSampler(
        ...     cuts_paths=[
        ...         ("data/cuts_a.jsonl", 2.0),
        ...         ("data/cuts_b.jsonl", 1.0),
        ...     ],
        ...     index_path="data/files.idx",
        ...     max_duration=600.0,
        ...     num_buckets=50,
        ...     quadratic_duration=30.0,
        ... )

    .. note:: This sampler works only with uncompressed jsonl manifests, as it creates extra index files
     with line byte offsets to quickly find and sample JSON lines.
     This means this sampler will not work with Webdataset and Lhotse Shar data format.

     .. warning:: If your OS doesn't provide TRNG, make sure to change the global Python seed when continuing
     the training. You should see a warning log being emitted in these cases.

    :param cuts_paths: Path, or list of paths, or list of tuples of (path, scale) to cutset files.
    :param index_path: Path to a file that contains the index of all cutsets and their line count
        (will be auto-created the first time this object is initialized).
    :param max_duration: Maximum total number of audio seconds in a mini-batch (dynamic batch size).
    :param max_cuts: Maximum number of examples in a mini-batch (static batch size).
    :param num_buckets: If set, enables bucketing (each mini-batch has examples of a similar duration).
    :param quadratic_duration: If set, adds a penalty term for longer duration cuts.
        Works well with models that have quadratic time complexity to keep GPU utilization similar
        when using bucketing. Suggested values are between 30 and 45.
    :param base_seed: If set, can be used to force this sampler to become deterministic
        (each node and worker are still going to produce different results).
        Primarily useful in debugging.
    """

    def __init__(
        self,
        cuts_paths: Union[Pathlike, Iterable[Pathlike], Iterable[PathlikeAndScale]],
        index_path: Pathlike,
        max_duration: Optional[Seconds] = None,
        max_cuts: Optional[int] = None,
        num_buckets: Optional[int] = None,
        quadratic_duration: Optional[Seconds] = None,
        base_seed: Optional[int] = None,
    ) -> None:
        super().__init__(data_source=None)

        # Parse file paths and scales
        self.paths = []
        self.scales = []
        if isinstance(cuts_paths, (Path, str)):
            # Single path input
            self.paths.append(Path(cuts_paths))
            self.scales.append(1.0)
        else:
            # Multiple path input
            cuts_paths = list(cuts_paths)
            self.paths = []
            if isinstance(cuts_paths[0], (Path, str)):
                # Without scales
                for p in cuts_paths:
                    assert isinstance(
                        p, (Path, str)
                    ), "Mixing paths with and without scales is not allowed."
                    self.paths.append(Path(p))
                    self.scales.append(1.0)
            else:
                for tpl in cuts_paths:
                    # With scales
                    assert len(tpl) == 2, (
                        f"Expected (path, scale) but got: {tpl} "
                        f"[note: mixing paths with and without scales is not allowed]"
                    )
                    p, scale = tpl
                    assert isinstance(
                        p, (Path, str)
                    ), f"Path must be a string or Path, got: {p}"
                    assert isinstance(
                        scale, (int, float)
                    ), f"Scale must be an int or float, got: {scale}"
                    self.paths.append(Path(p))
                    self.scales.append(scale)

        self.index_path = index_path
        self.max_duration = max_duration
        self.max_cuts = max_cuts
        self.num_buckets = num_buckets
        self.quadratic_duration = quadratic_duration
        self.base_seed = base_seed

        self.diagnostics = SamplingDiagnostics()
        self.index = ManifestIndex(self.paths, self.index_path)
        self.scaled_line_counts = [
            lc * scale
            for lc, scale in zip(self.index.line_counts.values(), self.scales)
        ]
        self.ddp_rank = get_rank()

    def state_dict(self) -> Dict:
        """Stub state_dict method that returns nothing - this sampler is stateless."""
        return {}

    def load_state_dict(self, state_dict: Dict) -> None:
        """Stub load_state_dict method that does nothing - this sampler is stateless."""
        return

    def __iter__(self) -> Generator[CutSet, None, None]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        my_id = worker_id + 1000 * self.ddp_rank
        # The seed depends on the global random state, DDP node ID, and dataloader worker ID.
        # It will be different each time the script is launched.
        # If the OS supports it, we'll try to get a true random number for the seed
        # so that we can guarantee this sampler is non-deterministic between runs.
        # Note: we could use TRNG for each sampling in the loop below, but with enough
        #       workers it might run into locking issues because of blocking syscalls.
        if self.base_seed is not None:
            # User requested deterministic results
            base_seed = self.base_seed
        else:
            try:
                # secrets uses TRNG provided by the OS; it doesn't depend on any seed
                base_seed = secrets.randbelow(2**20)
            except NotImplementedError:
                logging.warning(
                    "TRNG is not available on this system: make sure you changed Python's global random seed "
                    "if you are resuming model training to avoid iterating over identical data. "
                    "(hint: you may use lhotse.utils.fix_random_seed)"
                )
                base_seed = random.randint(0, 2**20)
        seed = base_seed + my_id
        rng = random.Random(seed)
        logging.info(
            f"[{type(self).__name__}] Initialized sampler RNG with seed {seed} (== base_seed={base_seed} + my_id={my_id}) "
            f"[ddp_rank={self.ddp_rank} worker_id={worker_id}]"
        )
        print(
            f"[{type(self).__name__}] Initialized sampler RNG with seed {seed} (== base_seed={base_seed} + my_id={my_id}) "
            f"[ddp_rank={self.ddp_rank} worker_id={worker_id}]"
        )

        def _inner():
            """
            Infinite generator of cuts.
            Each cut is samples in two steps:
            - first we select a cutset file, weighted by line count (num cuts)
            - then we randomly select a line from that file using uniform distribution
            """
            n = 0
            while True:
                # Choose a file
                path = rng.choices(self.paths, self.scaled_line_counts)[0]
                # Choose a line
                line_offsets = self.index.line_offsets[path]
                begin_idx = rng.randrange(len(line_offsets) - 1)
                begin, end = line_offsets[begin_idx], line_offsets[begin_idx + 1]
                # Read JSON line and initialize a Cut object
                with path.open() as f:
                    f.seek(begin)
                    line = f.read(end - begin)
                data = decode_json_line(line)
                cut = deserialize_cut(data)
                # Update cut ID since the same item may land in a single mini-batch
                # (note: CutSet prohibits two items sharing the same ID)
                cut.id = f"{cut.id}_it{n}"
                yield cut
                n += 1

        if self.num_buckets is not None and self.num_buckets > 1:
            inner_sampler = DynamicBucketingSampler(
                _inner(),
                max_duration=self.max_duration,
                max_cuts=self.max_cuts,
                num_buckets=self.num_buckets,
                shuffle=False,
                drop_last=False,
                quadratic_duration=self.quadratic_duration,
                world_size=1,
                rank=0,
            )
        else:
            inner_sampler = DynamicCutSampler(
                _inner(),
                max_duration=self.max_duration,
                max_cuts=self.max_cuts,
                shuffle=False,
                drop_last=False,
                world_size=1,
                rank=0,
            )
        self.diagnostics = inner_sampler.diagnostics
        yield from inner_sampler

    def get_report(self) -> str:
        """Returns a string describing the statistics of the sampling process so far."""
        return self.diagnostics.get_report()


class ManifestIndex:
    """
    An index of line count and line offset for each cutset manifest.
    When created for the first time, it writes a .jsonl.idx file for each .jsonl file that contains byte offsets for each line.
    It also writes a file at ``index_path`` that has the line count and path for each manifest.
    When this object is instantiated again (e.g. when resuming training), it will just load the contents of existing files from disk.

    Objects of this class expose two members: ``line_counts: Dict[Path, int]`` and ``line_offsets: Dict[Path, List[int]]`` to simplify working with manifests.

    :param manifest_paths: A list of paths to cut sets.
    :param index_path: A path where we should write the line count index (if it doesn't exist).
    :param force: When true, we'll ignore existing files and reindex the cutsets.
    """

    def __init__(
        self,
        manifest_paths: Sequence[Pathlike],
        index_path: Pathlike,
        force: bool = False,
    ) -> None:
        self.line_counts: Dict[Path, int] = {}
        self.line_offsets: Dict[Path, Tuple[int]] = {}
        for p in map(Path, manifest_paths):
            assert (
                p.suffix == ".jsonl"
            ), f"We only support uncompressed .jsonl files in this sampler, but received: {p}"

            offset_path = p.with_suffix(".jsonl.idx")
            if offset_path.is_file() and not force:
                offsets = self._load(offset_path)
            else:
                offsets = self._process(p, offset_path)
            self.line_counts[p] = len(offsets)
            self.line_offsets[p] = offsets

        # Write a single cutset index in format:
        # <number-of-lines> <cutset-path>
        # Example:
        # 10015 data/cuts-part-0001.jsonl
        # 376101 data/cuts-part-0002.jsonl
        # 572 data/cuts-part-0003.jsonl
        if not index_path.is_file() or force:
            with index_path.open("w") as index_f:
                for p, lc in self.line_counts.items():
                    print(f"{lc} {p}", file=index_f)

    def _load(self, file_index: Path) -> Tuple[int]:
        with file_index.open() as f:
            offsets = tuple(map(int, f))
        return offsets

    def _process(self, manifest: Path, file_index: Path) -> Tuple[int]:
        # Write line index for each cutset in format <begin-byte> per line, e.g.:
        # 0
        # 214
        # 357
        # ...
        offsets = [0]
        with manifest.open() as cuts_f, file_index.open("w") as index_f:
            print(0, file=index_f)
            line = cuts_f.readline()
            while line:
                offsets.append(cuts_f.tell())
                print(offsets[-1], file=index_f)
                line = cuts_f.readline()
        return tuple(offsets)


def get_world_size() -> int:
    """Source: https://github.com/danpovey/icefall/blob/74bf02bba6016c1eb37858a4e0e8a40f7d302bdb/icefall/dist.py#L56"""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_rank() -> int:
    """Source: https://github.com/danpovey/icefall/blob/74bf02bba6016c1eb37858a4e0e8a40f7d302bdb/icefall/dist.py#L56"""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0
