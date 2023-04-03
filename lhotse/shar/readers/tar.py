import tarfile
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

from lhotse import Features, Recording
from lhotse.array import Array, TemporalArray
from lhotse.serialization import decode_json_line, deserialize_item, open_best
from lhotse.shar.utils import fill_shar_placeholder
from lhotse.utils import Pathlike

Manifest = Union[Recording, Array, TemporalArray, Features]


class TarIterator:
    """
    TarIterator is a convenience class for reading arrays/audio stored in Lhotse Shar tar files.
    It is specific to Lhotse Shar format and expects the tar file to have the following structure:
    - each file is stored in a separate tar member
    - the file name is the key of the array
    - every array has two corresponding files:
        - the metadata: the file extension is ``.json`` and the file contains
          a Lhotse manifest (Recording, Array, TemporalArray, Features)
          for the data item.
        - the data: the file extension is the format of the array,
          and the file contents are the serialized array (possibly compressed).
        - the data file can be empty in case some cut did not contain that field.
          In that case, the data file has extension ``.nodata`` and the manifest file
          has extension ``.nometa``.
        - these files are saved one after another, the data is first, and the metadata follows.

    Iterating over TarReader yields tuples of ``(Optional[manifest], filename)`` where
    ``manifest`` is a Lhotse manifest with binary data attached to it, and ``filename``
    is the name of the data file inside tar archive.
    """

    def __init__(self, source: Pathlike) -> None:
        self.source = source

    def __iter__(
        self,
    ) -> Generator[Tuple[Optional[Manifest], Path], None, None]:
        with tarfile.open(fileobj=open_best(self.source, mode="rb"), mode="r|*") as tar:
            for ((data, data_path), (meta, meta_path)) in iterate_tarfile_pairwise(tar):
                if meta is not None:
                    meta = deserialize_item(decode_json_line(meta.decode("utf-8")))
                    fill_shar_placeholder(manifest=meta, data=data, tarpath=data_path)
                yield meta, data_path


def iterate_tarfile_pairwise(
    tar_file: tarfile.TarFile,
) -> Generator[Tuple[Optional[bytes], Optional[Manifest], Path, Path], None, None]:
    result = []
    for tarinfo in tar_file:
        if len(result) == 2:
            yield tuple(result)
            result = []
        result.append(parse_tarinfo(tarinfo, tar_file))

    if len(result) == 2:
        yield tuple(result)

    if len(result) == 1:
        raise RuntimeError(
            "Uneven number of files in the tarfile (expected to iterate pairs of binary data + JSON metadata."
        )


def parse_tarinfo(
    tarinfo: tarfile.TarInfo, tar_file: tarfile.TarFile
) -> Tuple[Optional[bytes], Path]:
    """
    Parse a tarinfo object and return the data it points to as well as the internal path.
    """
    path = Path(tarinfo.path)
    if path.suffix == ".nodata" or path.suffix == ".nometa":
        return None, path
    data = tar_file.extractfile(tarinfo).read()
    return data, path
