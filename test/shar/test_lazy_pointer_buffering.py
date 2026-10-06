from io import BytesIO

from lhotse.shar import lazy_pointer


def test_open_seekable_bounds_local_header_read(tmp_path, monkeypatch):
    path = tmp_path / "archive.tar"
    payload = bytes(range(256)) * 8192
    path.write_bytes(payload)
    opened = []

    def open_with_large_buffer(path, mode):
        handle = open(path, mode, buffering=1024 * 1024)
        opened.append(handle.raw)
        return handle

    monkeypatch.setattr(lazy_pointer, "open_best", open_with_large_buffer)
    monkeypatch.setattr(
        lazy_pointer.os, "posix_fadvise", lambda *args: None, raising=False
    )
    monkeypatch.setattr(lazy_pointer.os, "POSIX_FADV_RANDOM", 1, raising=False)
    with lazy_pointer._open_seekable(str(path)) as handle:
        assert handle.read(512) == payload[:512]
        assert opened[0].tell() <= 8192
        handle.seek(123456)
        assert handle.read(65536) == payload[123456:188992]
    assert opened[0].closed


def test_open_seekable_preserves_non_file_backend(monkeypatch):
    handle = BytesIO(b"payload")
    monkeypatch.setattr(lazy_pointer, "open_best", lambda path, mode: handle)
    assert lazy_pointer._open_seekable("custom-backend") is handle


def test_open_seekable_hints_random_local_access(tmp_path, monkeypatch):
    path = tmp_path / "archive.tar"
    path.write_bytes(b"payload")
    calls = []
    monkeypatch.setattr(
        lazy_pointer.os,
        "posix_fadvise",
        lambda *args: calls.append(args),
        raising=False,
    )
    monkeypatch.setattr(lazy_pointer.os, "POSIX_FADV_RANDOM", 1, raising=False)
    with lazy_pointer._open_seekable(str(path)) as handle:
        assert calls == [(handle.fileno(), 0, 0, 1)]
        assert handle.read() == b"payload"


def test_open_seekable_tolerates_unsupported_access_hint(tmp_path, monkeypatch):
    path = tmp_path / "archive.tar"
    path.write_bytes(b"payload")

    def unsupported(*args):
        raise OSError("Access hints are unavailable")

    original = open(path, "rb", buffering=1024 * 1024)
    monkeypatch.setattr(lazy_pointer, "open_best", lambda path, mode: original)
    monkeypatch.setattr(lazy_pointer.os, "posix_fadvise", unsupported, raising=False)
    with lazy_pointer._open_seekable(str(path)) as handle:
        assert handle is original
        assert handle.read() == b"payload"


def test_open_seekable_without_posix_access_hints(tmp_path, monkeypatch):
    path = tmp_path / "archive.tar"
    path.write_bytes(b"payload")
    original = open(path, "rb", buffering=1024 * 1024)
    monkeypatch.setattr(lazy_pointer, "open_best", lambda path, mode: original)
    monkeypatch.delattr(lazy_pointer.os, "posix_fadvise", raising=False)
    with lazy_pointer._open_seekable(str(path)) as handle:
        assert handle is original
        assert handle.read() == b"payload"
