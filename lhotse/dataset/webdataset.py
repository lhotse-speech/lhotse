import pickle
from typing import Dict, Optional, Sequence, Union

from tqdm.auto import tqdm

from lhotse import CutSet
from lhotse.serialization import LazyIteratorChain
from lhotse.utils import Pathlike, is_module_available


def export_to_webdataset(
    cuts: CutSet,
    output_path: Pathlike,
    shard_size: Optional[int] = None,
    verbose: bool = True,
    audio_format: str = "flac",
    drop_audio: bool = False,
    drop_features: bool = False,
) -> None:
    """
    Saves the CutSet metadata along with audio/features data into a WebDataset archive.
    The audio and feature data is read, decoded, and encoded into ``audio_format`` for audio,
    lilcom for features and arrays with floating point type, and pickle for all other dtypes.
    The intended use of this function is to speed up the I/O in training data pipelines by
    converting random access reads to sequential access reads.

    Supported values for ``audio_format`` are the same as for the ``format`` argument in
    ``torchaudio.save`` function with ``sox_io`` backend.

    If ``shard_size`` is specified, we will leverage WebDataset's ``ShardWriter`` to
    create multiple tarballs with ``shard_size`` items per shard. In that mode, we expect
    that ``output_path`` contains a pattern like "/path/to/shard-%06d.tar", which will
    be internally expanded with the shard index.
    """
    if not is_module_available("webdataset"):
        raise ImportError("Please 'pip install webdataset' first.")
    import webdataset as wds

    if shard_size is not None:
        assert shard_size > 0
        sink = wds.ShardWriter(output_path, maxcount=shard_size)
    else:
        sink = wds.TarWriter(output_path)

    with sink:
        for idx, cut in tqdm(
            enumerate(cuts), desc="Creating WebDataset tarball(s)", disable=not verbose
        ):
            if drop_audio:
                cut = cut.drop_recording()
            if drop_features:
                cut = cut.drop_features()
            cut = cut.move_to_memory(audio_format=audio_format)
            data = pickle.dumps(cut.to_dict())
            sink.write({"__key__": cut.id, "data": data})


class LazyWebdatasetIterator:
    """
    LazyWebdatasetIterator provides the ability to read Lhotse objects from a
    WebDataset tarball on-the-fly, without reading its full contents into memory.

    This class is designed to be a partial "drop-in" replacement for ordinary dicts
    to support lazy loading of RecordingSet, SupervisionSet and CutSet.
    Since it does not support random access reads, some methods of these classes
    might not work properly.

    The behaviour of the underlying ``WebDataset`` instance can be customized by
    providing its kwargs directly to the constructor of this class.
    """

    def __init__(
        self, source: Union[Pathlike, Sequence[Pathlike]], **wds_kwargs
    ) -> None:
        if not is_module_available("webdataset"):
            raise ImportError("Please 'pip install webdataset' first.")

        self.source = source
        self.wds_kwargs = wds_kwargs

    def _reset(self) -> None:
        if not is_module_available("webdataset"):
            raise ImportError("Please 'pip install webdataset' first.")

        self._ds = mini_webdataset(self.source, **self.wds_kwargs)
        self._ds_iter = iter(self._ds)

    def __getstate__(self):
        """
        Store the state for pickling -- we'll only store the path + kwargs, and re-initialize
        this iterator when unpickled. This is necessary to transfer this object across processes
        for PyTorch's DataLoader workers.
        """
        state = {"source": self.source, "wds_kwargs": self.wds_kwargs}
        return state

    def __setstate__(self, state: Dict):
        """Restore the state when unpickled."""
        self.__dict__.update(state)

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        from lhotse.serialization import deserialize_item

        data_dict = next(self._ds_iter)
        data = pickle.loads(data_dict["data"])
        item = deserialize_item(data)
        return item

    def values(self):
        yield from self

    def keys(self):
        return (item.id for item in self)

    def items(self):
        return ((item.id, item) for item in self)

    def __add__(self, other) -> LazyIteratorChain:
        return LazyIteratorChain(self, other)


def mini_webdataset(
    urls,
    repeat: bool = False,
    shuffle_shards: bool = False,
    shuffle: bool = False,
    split_by_worker: bool = False,
    split_by_node: bool = False,
    shuffle_bufsize: int = 1000,
):
    """
    Return a pipeline for WebDataset-style data files.

    This is a convenience function for constructing a partial pipeline
    that reads from a set of sharded tar files, extracts the individual
    files, and groups them together into samples (dictionaries).

    You can use all the methods from `Composable` (`then`, `compose`) and
    from `Shorthands` (`batched`, `unbatched`, `decode`, `shuffle`, etc.)
    on the result.

    .. note: This is a reduced version of ``webdataset.WebDataset`` function,
        that only uses the functionalities relevant to Lhotse, and makes it
        possible to disable the node/worker splitting.

    :param urls: the source URLs: a string or a list.
    :param repeat: repeat infinitely if True.
    :param shuffle: shuffle the items if True (after shuffling the shards, if enabled).
    :param shuffle_shards: shuffle the shards if True.
        Only takes effect when ``urls`` is a list of shard paths/urls.
    :param split_by_worker: if True, shards are split per DataLoader worker subprocesses,
        otherwise each dataloader worker will yield the same data.
        Only takes effect when ``urls`` is a list of shard paths/urls.
    :param split_by_node: if True, shards are split per node in DDP training,
        otherwise on each node we'll yield the same data.
        Only takes effect when ``urls`` is a list of shard paths/urls.
    :param shuffle_bufsize: Buffer size for the ``shuffle`` argument.
        Larger bufsize means more memory usage but potentially improved randomness.
    """
    if not is_module_available("webdataset"):
        raise ImportError("Please 'pip install webdataset' first.")

    from webdataset import PytorchShardList, reraise_exception
    from webdataset import tariterators

    result = PytorchShardList(
        urls,
        shuffle=shuffle_shards,
        split_by_worker=split_by_worker,
        split_by_node=split_by_node,
    )
    result = result.then(tariterators.url_opener, handler=reraise_exception)
    result = result.then(tariterators.tar_file_expander, handler=reraise_exception)
    result = result.then(tariterators.group_by_keys)
    if repeat:
        result = result.repeat()
    if shuffle:
        result = result.shuffle(shuffle_bufsize)
    return result
