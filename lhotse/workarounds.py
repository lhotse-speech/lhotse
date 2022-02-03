import gzip
import io
import os


class Hdf5MemoryIssueFix:
    """
    Use this class to limit the growing memory use when reading from HDF5 files.

    It should be instantiated within the dataloading worker, i.e., the best place
    is likely inside the PyTorch Dataset class.

    Every time a new batch/example is returned, call ``.update()``.
    Once per ``reset_interval`` updates, this object will close all open HDF5 file
    handles, which seems to limit the memory use.
    """

    def __init__(self, reset_interval: int = 100) -> None:
        self.counter = 0
        self.reset_interval = reset_interval

    def update(self) -> None:
        from lhotse import close_cached_file_handles

        if self.counter > 0 and self.counter % self.reset_interval == 0:
            close_cached_file_handles()
            self.counter = 0
        self.counter += 1


class AltGzipFile(gzip.GzipFile):
    """
    This is a workaround for Python's stdlib gzip module
    not implementing gzip decompression correctly...
    Command-line gzip is able to discard "trailing garbage" in gzipped files,
    but Python's gzip is not.

    Original source: https://gist.github.com/nczeczulin/474ffbf6a0ab67276a62
    """

    def read(self, size=-1):
        chunks = []
        try:
            if size < 0:
                while True:
                    chunk = self.read1()
                    if not chunk:
                        break
                    chunks.append(chunk)
            else:
                while size > 0:
                    chunk = self.read1(size)
                    if not chunk:
                        break
                    size -= len(chunk)
                    chunks.append(chunk)
        except OSError as e:
            if not chunks or not str(e).startswith("Not a gzipped file"):
                raise
            # logging.warn('decompression OK, trailing garbage ignored')

        return b"".join(chunks)


def gzip_open_robust(
    filename,
    mode="rb",
    compresslevel=9,  # compat with Py 3.6
    encoding=None,
    errors=None,
    newline=None,
):
    """Open a gzip-compressed file in binary or text mode.

    The filename argument can be an actual filename (a str or bytes object), or
    an existing file object to read from or write to.

    The mode argument can be "r", "rb", "w", "wb", "x", "xb", "a" or "ab" for
    binary mode, or "rt", "wt", "xt" or "at" for text mode. The default mode is
    "rb", and the default compresslevel is 9.

    For binary mode, this function is equivalent to the GzipFile constructor:
    GzipFile(filename, mode, compresslevel). In this case, the encoding, errors
    and newline arguments must not be provided.

    For text mode, a GzipFile object is created, and wrapped in an
    io.TextIOWrapper instance with the specified encoding, error handling
    behavior, and line ending(s).

    Note: This method is copied from Python's 3.7 stdlib, and patched to handle
    "trailing garbage" in gzip files. We could monkey-patch the stdlib version,
    but we imagine that some users prefer third-party libraries like Lhotse
    not to do such things.
    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("Argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

    gz_mode = mode.replace("t", "")
    if isinstance(filename, (str, bytes, os.PathLike)):
        binary_file = AltGzipFile(filename, gz_mode, compresslevel)
    elif hasattr(filename, "read") or hasattr(filename, "write"):
        binary_file = AltGzipFile(None, gz_mode, compresslevel, filename)
    else:
        raise TypeError("filename must be a str or bytes object, or a file")

    if "t" in mode:
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file
