import copy
import errno
import os
import stat
import tarfile
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

import pytest

from lhotse.utils import safe_extract


@pytest.fixture
def make_tar():
    with ExitStack() as stack:

        def create(entries):
            buffer = BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                for name, kind, value in entries:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.mode = 0o750 if member.isdir() else 0o444
                    member.mtime = 123456789
                    if member.isreg():
                        member.size = len(value)
                        archive.addfile(member, BytesIO(value))
                    else:
                        member.linkname = value
                        archive.addfile(member)
            buffer.seek(0)
            archive = stack.enter_context(tarfile.open(fileobj=buffer, mode="r:"))
            # Exercise Lhotse's checks independently of the Python version's
            # default filter. Older Python versions ignore this attribute.
            archive.extraction_filter = lambda member, path: member
            return archive

        yield create


@pytest.fixture
def symlinks_supported(tmp_path):
    target = tmp_path / "link-target"
    target.touch()
    try:
        (tmp_path / "symbolic-link").symlink_to(target)
        os.link(target, tmp_path / "hard-link")
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not support symbolic and hard links")


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("errorlevel", [0, 1])
def test_reject_external_links(make_tar, tmp_path, kind, absolute, errorlevel):
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "file"
    victim.write_bytes(b"original")
    if kind == tarfile.SYMTYPE:
        linkname = str(outside) if absolute else "../outside"
        payload_path = "link/file"
    else:
        linkname = str(victim) if absolute else "../outside/file"
        payload_path = "link"
    archive = make_tar(
        [("link", kind, linkname), (payload_path, tarfile.REGTYPE, b"replacement")]
    )
    archive.getmembers()[0].mode = 0o644
    archive.errorlevel = errorlevel

    with pytest.raises(ValueError, match="Path Traversal"):
        safe_extract(archive, path=tmp_path / "destination")

    assert victim.read_bytes() == b"original"
    assert not (tmp_path / "destination/link").is_symlink()


def test_reject_preexisting_external_symlink(make_tar, tmp_path, symlinks_supported):
    destination = tmp_path / "destination"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "link").symlink_to(outside, target_is_directory=True)
    archive = make_tar([("link/file", tarfile.REGTYPE, b"payload")])

    with pytest.raises(ValueError, match="Path Traversal"):
        safe_extract(archive, path=destination)

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("payload", [False, True])
def test_reject_changing_link_targets(
    make_tar, tmp_path, symlinks_supported, reverse, payload
):
    outside = tmp_path / "outside"
    outside.mkdir()
    entries = [("a", tarfile.SYMTYPE, "."), ("b", tarfile.SYMTYPE, "a/../outside")]
    if reverse:
        entries.reverse()
    if payload:
        entries.append(("b/file", tarfile.REGTYPE, b"payload"))
    archive = make_tar(entries)

    with pytest.raises(ValueError, match="Path Traversal"):
        safe_extract(archive, path=tmp_path / "destination")

    assert list(outside.iterdir()) == []


def test_guard_deferred_directory_metadata(make_tar, tmp_path, symlinks_supported):
    outside = tmp_path / "outside/dir"
    outside.mkdir(parents=True)
    os.utime(outside, (987654321, 987654321))
    original = outside.stat()
    archive = make_tar(
        [
            ("deep", tarfile.DIRTYPE, ""),
            ("a", tarfile.SYMTYPE, "deep"),
            ("a/dir", tarfile.DIRTYPE, ""),
            ("a", tarfile.SYMTYPE, "b/../outside"),
            ("b", tarfile.SYMTYPE, "."),
        ]
    )
    archive.errorlevel = 0

    with pytest.raises(ValueError, match="Path Traversal"):
        safe_extract(archive, path=tmp_path / "destination")

    assert outside.stat().st_mtime_ns == original.st_mtime_ns
    assert outside.stat().st_mode == original.st_mode


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_guard_relocated_link_fallback(make_tar, tmp_path, monkeypatch, kind):
    archive = make_tar(
        [("dir/link", tarfile.SYMTYPE, "../file"), ("alias", kind, "dir/link")]
    )
    archive.errorlevel = 0

    def unavailable(*args, **kwargs):
        raise NotImplementedError("Symbolic links unavailable")

    if kind == tarfile.SYMTYPE:
        monkeypatch.setattr(os, "symlink", unavailable)

    with pytest.raises(ValueError, match="Path Traversal"):
        safe_extract(
            archive,
            path=tmp_path / "destination",
            members=[archive.getmember("alias")],
        )

    assert not (tmp_path / "destination/alias").is_symlink()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_preserve_regular_file_link_fallback(make_tar, tmp_path, monkeypatch, kind):
    archive = make_tar(
        [("source", tarfile.REGTYPE, b"payload"), ("alias", kind, "source")]
    )
    original_members = [member.get_info() for member in archive.getmembers()]

    def unavailable(*args, **kwargs):
        raise NotImplementedError("Symbolic links unavailable")

    monkeypatch.setattr(os, "symlink", unavailable)
    safe_extract(archive, path=tmp_path, members=[archive.getmember("alias")])

    assert (tmp_path / "alias").read_bytes() == b"payload"
    assert not (tmp_path / "alias").is_symlink()
    assert not (tmp_path / "source").exists()
    assert [member.get_info() for member in archive.getmembers()] == original_members


def test_preserve_internal_links(make_tar, tmp_path, symlinks_supported):
    archive = make_tar(
        [
            ("folder/file", tarfile.REGTYPE, b"payload"),
            ("folder/symbolic", tarfile.SYMTYPE, "file"),
            ("folder/hard", tarfile.LNKTYPE, "folder/file"),
            ("alias", tarfile.SYMTYPE, "folder"),
            ("alias/new", tarfile.REGTYPE, b"new"),
            ("folder/sub/link", tarfile.SYMTYPE, "../file"),
            ("root", tarfile.SYMTYPE, "."),
        ]
    )

    safe_extract(archive, path=tmp_path)

    assert (tmp_path / "folder/symbolic").is_symlink()
    assert (tmp_path / "folder/symbolic").read_bytes() == b"payload"
    assert (tmp_path / "folder/hard").stat().st_ino == (
        tmp_path / "folder/file"
    ).stat().st_ino
    assert (tmp_path / "folder/new").read_bytes() == b"new"
    assert (tmp_path / "folder/sub/link").read_bytes() == b"payload"
    assert (tmp_path / "root").resolve() == tmp_path.resolve()


def test_preserve_confined_absolute_links(make_tar, tmp_path, symlinks_supported):
    archive = make_tar(
        [
            ("file", tarfile.REGTYPE, b"payload"),
            ("symbolic", tarfile.SYMTYPE, str(tmp_path / "file")),
            ("hard", tarfile.LNKTYPE, str(tmp_path / "file")),
        ]
    )

    safe_extract(archive, path=tmp_path)

    assert (tmp_path / "symbolic").is_symlink()
    assert (tmp_path / "symbolic").read_bytes() == b"payload"
    assert (tmp_path / "hard").stat().st_ino == (tmp_path / "file").stat().st_ino


def test_preserve_relative_destination_and_root_directory(
    make_tar, tmp_path, monkeypatch
):
    archive = make_tar(
        [(".", tarfile.DIRTYPE, ""), ("./file", tarfile.REGTYPE, b"payload")]
    )
    monkeypatch.chdir(tmp_path)

    safe_extract(archive, path="destination")

    assert (tmp_path / "destination/file").read_bytes() == b"payload"
    assert (tmp_path / "destination").stat().st_mtime == 123456789


def test_preserve_native_default(make_tar, tmp_path):
    archive = make_tar([("file", tarfile.REGTYPE, b"payload")])
    del archive.extraction_filter
    archive.extractall(tmp_path / "baseline")

    safe_extract(archive, path=tmp_path / "guarded")

    baseline = (tmp_path / "baseline/file").stat()
    guarded = (tmp_path / "guarded/file").stat()
    assert guarded.st_mode == baseline.st_mode
    assert guarded.st_mtime_ns == baseline.st_mtime_ns
    assert (guarded.st_uid, guarded.st_gid) == (baseline.st_uid, baseline.st_gid)
    assert (tmp_path / "guarded/file").read_bytes() == b"payload"


@pytest.mark.parametrize("numeric_owner", [False, True])
def test_preserve_metadata_and_instance_methods(make_tar, tmp_path, numeric_owner):
    archive = make_tar(
        [("folder", tarfile.DIRTYPE, ""), ("folder/file", tarfile.REGTYPE, b"payload")]
    )
    owners = []

    def chown(member, targetpath, numeric_owner):
        owners.append(numeric_owner)

    archive.chown = chown
    archive.getmembers()
    original_attributes = dict(archive.__dict__)

    safe_extract(archive, path=tmp_path, numeric_owner=numeric_owner)

    assert archive.__dict__ == original_attributes
    assert owners == [numeric_owner, numeric_owner]
    for name, mode in [("folder", 0o750), ("folder/file", 0o444)]:
        extracted = (tmp_path / name).stat()
        assert extracted.st_mtime == 123456789
        if os.name != "nt":
            assert stat.S_IMODE(extracted.st_mode) == mode


@pytest.mark.parametrize("reverse", [False, True])
def test_preserve_member_selection_and_order(make_tar, tmp_path, reverse):
    archive = make_tar(
        [
            ("file", tarfile.REGTYPE, b"first"),
            ("file", tarfile.REGTYPE, b"second"),
            ("ignored", tarfile.REGTYPE, b"ignored"),
        ]
    )
    members = archive.getmembers()[:2]
    for member in members:
        member.mode = 0o644
    if reverse:
        members.reverse()

    safe_extract(archive, path=tmp_path, members=iter(members))

    assert (tmp_path / "file").read_bytes() == (b"first" if reverse else b"second")
    assert not (tmp_path / "ignored").exists()


def test_restore_methods_after_failure(make_tar, tmp_path, monkeypatch):
    archive = make_tar([("file", tarfile.REGTYPE, b"payload")])
    archive.getmembers()

    def fail(*args, **kwargs):
        raise RuntimeError("Extraction failed")

    monkeypatch.setattr(archive, "makefile", fail)
    original_attributes = dict(archive.__dict__)

    with pytest.raises(RuntimeError, match="Extraction failed"):
        safe_extract(archive, path=tmp_path)

    assert archive.__dict__ == original_attributes


@pytest.mark.parametrize("error", [errno.EACCES, errno.ENAMETOOLONG, errno.ELOOP])
def test_fail_closed_on_resolution_errors(make_tar, tmp_path, monkeypatch, error):
    archive = make_tar([("missing/file", tarfile.REGTYPE, b"payload")])
    archive.errorlevel = 0
    readlink = os.readlink

    def fail(path, *args, **kwargs):
        if Path(path).name == "file":
            raise OSError(error, "Cannot resolve path")
        return readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", fail)

    with pytest.raises(ValueError, match="Cannot safely resolve"):
        safe_extract(archive, path=tmp_path)

    assert not (tmp_path / "missing").exists()


@pytest.mark.skipif(
    not hasattr(tarfile.TarFile, "extraction_filter"),
    reason="Native extraction filters unavailable",
)
@pytest.mark.parametrize("outside", [False, True])
def test_preserve_and_guard_native_filters(make_tar, tmp_path, outside):
    archive = make_tar([("original", tarfile.REGTYPE, b"payload")])
    original = archive.getmembers()[0].get_info()

    def rename(member, path):
        member = copy.copy(member)
        member.name = "../escaped" if outside else "renamed"
        return member

    archive.extraction_filter = rename
    if outside:
        with pytest.raises(ValueError, match="Path Traversal"):
            safe_extract(archive, path=tmp_path / "destination")
        assert not (tmp_path / "escaped").exists()
    else:
        safe_extract(archive, path=tmp_path)
        assert (tmp_path / "renamed").read_bytes() == b"payload"
        assert not (tmp_path / "original").exists()
    assert archive.getmembers()[0].get_info() == original


@pytest.mark.skipif(
    not hasattr(tarfile, "data_filter"), reason="Native extraction filters unavailable"
)
def test_preserve_data_filter_metadata(make_tar, tmp_path):
    archive = make_tar([("file", tarfile.REGTYPE, b"payload")])
    archive.extraction_filter = tarfile.data_filter

    safe_extract(archive, path=tmp_path)

    assert (tmp_path / "file").read_bytes() == b"payload"
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / "file").stat().st_mode) == 0o644
