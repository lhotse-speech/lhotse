import io
import os
import tarfile

from lhotse import CutSet
from lhotse.ais.batch_loader import AISBatchLoader
from lhotse.audio.recording import Recording
from lhotse.audio.source import AudioSource, resolve_s3_to_local_mirror

_ENV_VAR = "LHOTSE_S3_LOCAL_MIRROR_ROOTS"


def _recording(source: str) -> Recording:
    return Recording(
        id="recording",
        sources=[
            AudioSource(
                type="url",
                channels=[0],
                source=source,
            )
        ],
        sampling_rate=16000,
        num_samples=16000,
        duration=1.0,
    )


def test_resolve_s3_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    source = "s3://bucket/key/audio.wav"

    assert resolve_s3_to_local_mirror(source) == source


def test_resolve_s3_file_to_first_available_mirror(tmp_path, monkeypatch):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    local = second_root / "bucket" / "key" / "audio.wav"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"audio")
    monkeypatch.setenv(_ENV_VAR, os.pathsep.join((str(first_root), str(second_root))))

    assert resolve_s3_to_local_mirror("s3://bucket/key/audio.wav") == str(local)


def test_resolve_s3_tar_member_to_local_mirror(tmp_path, monkeypatch):
    archive = tmp_path / "bucket" / "key" / "audio.tar"
    archive.parent.mkdir(parents=True)
    payload = b"audio"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("nested/member.wav")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    monkeypatch.setenv(_ENV_VAR, str(tmp_path))

    source_uri = "s3://bucket/key/audio.tar/nested/member.wav"
    assert resolve_s3_to_local_mirror(source_uri) == f"{archive}/nested/member.wav"

    source = AudioSource(type="url", channels=[0], source=source_uri)
    assert source._prepare_for_reading(0.0, None).read() == payload


def test_resolve_s3_keeps_uri_when_object_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, str(tmp_path))
    source = "s3://bucket/missing.tar/member.wav"

    assert resolve_s3_to_local_mirror(source) == source


def test_resolve_s3_rejects_parent_traversal(tmp_path, monkeypatch):
    mirror_root = tmp_path / "mirror"
    mirror_root.mkdir()
    (tmp_path / "outside.wav").write_bytes(b"audio")
    monkeypatch.setenv(_ENV_VAR, str(mirror_root))
    source = "s3://bucket/../../outside.wav"

    assert resolve_s3_to_local_mirror(source) == source


def test_ais_batch_loader_skips_mirrored_source_without_client(tmp_path, monkeypatch):
    local = tmp_path / "bucket" / "key" / "audio.wav"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"audio")
    monkeypatch.setenv(_ENV_VAR, str(tmp_path))
    cuts = CutSet.from_cuts([_recording("s3://bucket/key/audio.wav").to_cut()])
    loader = AISBatchLoader()

    assert not loader._cuts_have_ais_data(cuts)
    assert loader(cuts) is cuts
    assert loader._client is None


def test_ais_batch_loader_routes_mirrored_tar_member_locally(tmp_path, monkeypatch):
    archive = tmp_path / "bucket" / "key" / "audio.tar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"tar-placeholder")
    monkeypatch.setenv(_ENV_VAR, str(tmp_path))
    recording = _recording("s3://bucket/key/audio.tar/nested/member.wav")
    loader = AISBatchLoader.__new__(AISBatchLoader)

    has_ais_url = loader._collect_manifest_urls(
        recording,
        object(),
        shar_ptr_uses_batch=False,
        shar_ptr_fallback=[],
        manifest_idx=0,
    )

    assert not has_ais_url
    assert recording.sources[0].source == f"{archive}/nested/member.wav"
