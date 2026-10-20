import numpy as np
import pytest

from lhotse.indexing import IndexedJsonlReader


@pytest.mark.parametrize("offsets", [[0, 6, 3, 9], [0, 3, 3, 9], [0, 2**64 - 1, 9]])
def test_jsonl_reader_rejects_invalid_offsets_before_read(tmp_path, offsets):
    path = tmp_path / "data.jsonl"
    path.write_bytes(b"{}\n{}\n{}\n")
    np.array(offsets, dtype="<u8").tofile(str(path) + ".idx")
    with pytest.raises(ValueError, match="strictly increasing"):
        IndexedJsonlReader(path)


@pytest.mark.parametrize("offsets", [[0], [0, 3, 6, 9]])
def test_jsonl_reader_valid_offsets(tmp_path, offsets):
    path = tmp_path / "data.jsonl"
    path.write_bytes(b"" if len(offsets) == 1 else b"{}\n{}\n{}\n")
    np.array(offsets, dtype="<u8").tofile(str(path) + ".idx")
    reader = IndexedJsonlReader(path)
    try:
        assert list(reader) == ([] if len(offsets) == 1 else [{}, {}, {}])
    finally:
        reader.close()
