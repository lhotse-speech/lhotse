import errno
import gzip
import io
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from tarfile import TarFile
from typing import Iterator, List, Set, Union


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


def _resolve_path(path: Path) -> Path:
    """Resolve links, tolerating missing paths but no other filesystem errors."""
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        pass

    # Unlike resolve(strict=False) on older Python versions, this must not
    # suppress errors such as ELOOP, EACCES or ENAMETOOLONG.
    path = Path.cwd() / path
    if not path.is_absolute():
        raise ValueError(f"Cannot resolve archive path: {path}")
    resolved = Path(path.anchor)
    pending: List[Union[str, Path]] = list(reversed(path.parts[1:]))
    resolving: Set[Path] = set()
    while pending:
        part = pending.pop()
        if isinstance(part, Path):
            resolving.remove(part)
            continue
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        try:
            target = Path(os.readlink(candidate))
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.EINVAL):
                raise
            resolved = candidate
            continue
        if candidate in resolving:
            raise ValueError(f"Symlink loop in archive path: {candidate}")
        resolving.add(candidate)
        pending.append(candidate)
        if target.is_absolute():
            resolved = Path(target.anchor)
            pending.extend(reversed(target.parts[1:]))
        elif target.drive:
            raise ValueError(f"Cannot resolve archive link: {target}")
        else:
            pending.extend(reversed(target.parts))
    return resolved


@contextmanager
def guarded_tar_extraction(tar: TarFile, path: Path) -> Iterator[None]:
    """Guard actual writes without changing tarfile's filters or metadata policy."""
    root = _resolve_path(path)
    symlinks: Set[Path] = set()

    def check_path(target: Path, allow_root: bool = True) -> None:
        try:
            resolved = _resolve_path(target)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"Cannot safely resolve archive path: {target}") from error
        if root not in resolved.parents and not (allow_root and root == resolved):
            # ValueError is deliberately outside tarfile's suppressible errors:
            # containment failures must abort even when tar.errorlevel == 0.
            raise ValueError(f"Attempted Path Traversal in Tar File: {target}")

    # Retain the existing whole-archive path validation, including when only a
    # subset of members is selected. Later checks also see newly created links.
    for member in tar.getmembers():
        check_path(path / member.name, allow_root=member.isdir())

    def guard(method, extracting: bool = False):
        @wraps(method)
        def checked(member, targetpath, *args, **kwargs):
            target = Path(targetpath)
            check_path(target)
            if extracting:
                # makedirs() visits lexical parents, even when the final path
                # resolves inside root (e.g. ../outside/../destination/file).
                # Only the caller's destination itself may need outside parents.
                parent = target.parent
                while (
                    target not in (path, root)
                    and parent not in (path, root)
                    and parent != parent.parent
                ):
                    try:
                        parent.stat()
                    except FileNotFoundError:
                        check_path(parent)
                        parent = parent.parent
                    except OSError as error:
                        raise ValueError(
                            f"Cannot safely resolve archive path: {parent}"
                        ) from error
                    else:
                        break
                if member.issym():
                    check_path(target.parent / member.linkname)
                    symlinks.add(target)
                elif member.islnk():
                    check_path(
                        Path(getattr(member, "_link_target", path / member.linkname))
                    )
            return method(member, targetpath, *args, **kwargs)

        return checked

    methods = ("_extract_member", "chown", "chmod", "utime")
    overrides = {name: tar.__dict__[name] for name in methods if name in tar.__dict__}
    originals = {name: getattr(tar, name) for name in methods}
    try:
        # Guard the instance so recursive link fallback uses the same checks.
        # TarFile already requires exclusive access during extraction.
        for name, method in originals.items():
            setattr(tar, name, guard(method, extracting=name == "_extract_member"))
        yield
        # A later link can change the meaning of an earlier dangling target.
        for target in symlinks:
            check_path(target)
    finally:
        for name in methods:
            if name in overrides:
                setattr(tar, name, overrides[name])
            else:
                delattr(tar, name)
