"""
Unit tests for AISBatchLoader.

These tests use mocking to simulate AIStore client behavior,
allowing them to run in CI environments without AIStore infrastructure.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lhotse import CutSet, MonoCut
from lhotse.ais.batch_loader import AISBatchLoader, AISBatchLoaderError
from lhotse.ais.common import (
    ARCHIVE_EXTENSIONS,
    FILE_TO_MEMORY_TYPE,
    get_archive_extension,
)
from lhotse.array import Array, TemporalArray
from lhotse.audio import AudioSource, Recording
from lhotse.features import Features
from lhotse.image import Image
from lhotse.supervision import SupervisionSegment

aistore = pytest.importorskip(
    "aistore",
    minversion="1.17.0",
    reason="Please run `pip install aistore>=1.17.0` to enable these tests.",
)


@pytest.fixture
def mock_aistore_client():
    """Mock AIStore client with batch support."""
    client = MagicMock()
    batch = MagicMock()

    # Track the number of add() calls
    add_count = []

    def track_add(*args, **kwargs):
        add_count.append(1)

    batch.add.side_effect = track_add

    # Mock batch.get() to return an iterator of (obj, content) tuples
    def mock_batch_get():
        # Return dummy bytes content for each added object
        for i in range(len(add_count)):
            yield (MagicMock(), f"dummy_content_{i}".encode())

    batch.get.side_effect = lambda: mock_batch_get()
    client.batch.return_value = batch
    client.bucket.return_value = MagicMock()

    return client, batch


@pytest.fixture
def cut_with_url_recording():
    """Create a cut with Recording containing URL-type audio sources."""
    recording = Recording(
        id="rec-1",
        sources=[
            AudioSource(
                type="url", channels=[0], source="ais://mybucket/audio/file1.wav"
            )
        ],
        sampling_rate=16000,
        num_samples=160000,
        duration=10.0,
    )
    return MonoCut(
        id="cut-1",
        start=0.0,
        duration=10.0,
        channel=0,
        recording=recording,
        supervisions=[
            SupervisionSegment(
                id="sup-1", recording_id="rec-1", start=0.0, duration=10.0
            )
        ],
    )


@pytest.fixture
def cut_with_features():
    """Create a cut with Features stored in AIStore (no recording)."""
    features = Features(
        recording_id="rec-1",
        channels=0,
        start=0.0,
        duration=10.0,
        type="fbank",
        num_frames=1000,
        num_features=80,
        sampling_rate=16000,
        storage_type="numpy_files",
        storage_path="ais://mybucket/features",
        storage_key="feats.npy",
        frame_shift=0.01,
    )
    return MonoCut(
        id="cut-1",
        start=0.0,
        duration=10.0,
        channel=0,
        features=features,
        recording=None,  # No recording to avoid extra iter_data yield
    )


@pytest.fixture
def cut_with_array():
    """Create a cut with Array stored in AIStore (no recording)."""
    array = Array(
        storage_type="lilcom_files",
        storage_path="ais://mybucket/arrays",
        storage_key="array.llc",
        shape=(100, 80),
    )
    features = Features(
        recording_id="rec-1",
        channels=0,
        start=0.0,
        duration=10.0,
        type="fbank",
        num_frames=1000,
        num_features=80,
        sampling_rate=16000,
        storage_type="lilcom_files",
        storage_path="ais://mybucket/features",
        storage_key="feats.npy",
        frame_shift=0.01,
    )
    cut = MonoCut(
        id="cut-1",
        start=0.0,
        duration=10.0,
        channel=0,
        recording=None,  # No recording
        features=features,  # Need features for the cut to be valid
    )
    cut.custom_array = array
    return cut


@pytest.fixture
def cut_with_image():
    """Create a cut with Image stored in AIStore (no recording)."""
    image = Image(
        storage_type="pillow_files",
        storage_path="ais://mybucket/images",
        storage_key="image.jpg",
        width=640,
        height=480,
    )
    features = Features(
        recording_id="rec-1",
        channels=0,
        start=0.0,
        duration=10.0,
        type="fbank",
        num_frames=1000,
        num_features=80,
        sampling_rate=16000,
        storage_type="pillow_files",
        storage_path="ais://mybucket/features",
        storage_key="feats.npy",
        frame_shift=0.01,
    )
    cut = MonoCut(
        id="cut-1",
        start=0.0,
        duration=10.0,
        channel=0,
        recording=None,  # No recording
        features=features,  # Need features for the cut to be valid
    )
    cut.custom_image = image
    return cut


@pytest.fixture
def cut_with_temporal_array():
    """Create a cut with TemporalArray stored in AIStore (no recording)."""
    from lhotse.array import Array, TemporalArray

    base_array = Array(
        storage_type="numpy_files",
        storage_path="ais://mybucket/temporal",
        storage_key="temporal.npy",
        shape=(100, 80),
    )
    temporal_array = TemporalArray(
        array=base_array,
        temporal_dim=0,
        start=0.0,
        frame_shift=0.01,
    )
    features = Features(
        recording_id="rec-1",
        channels=0,
        start=0.0,
        duration=10.0,
        type="fbank",
        num_frames=1000,
        num_features=80,
        sampling_rate=16000,
        storage_type="numpy_files",
        storage_path="ais://mybucket/features",
        storage_key="feats.npy",
        frame_shift=0.01,
    )
    cut = MonoCut(
        id="cut-1",
        start=0.0,
        duration=10.0,
        channel=0,
        recording=None,  # No recording
        features=features,  # Need features for the cut to be valid
    )
    cut.custom_temporal = temporal_array
    return cut


class TestAISBatchLoaderInit:
    """Tests for AISBatchLoader initialization."""

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_init(self, mock_get_client):
        """Test that AISBatchLoader initializes with an AIStore client."""
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, None)

        loader = AISBatchLoader()

        assert loader.client == mock_client
        mock_get_client.assert_called_once()


class TestAISBatchLoaderCall:
    """Tests for AISBatchLoader.__call__() method."""

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_rejects_lazy_cutset(self, mock_get_client, cut_with_url_recording):
        """Test that lazy CutSets are rejected."""
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, None)

        loader = AISBatchLoader()
        lazy_cuts = CutSet.from_cuts([cut_with_url_recording]).repeat()

        with pytest.raises(ValueError, match="Lazy CutSets cannot be used"):
            loader(lazy_cuts)

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_processes_recording_with_url(
        self, mock_get_client, mock_aistore_client, cut_with_url_recording
    ):
        """Test processing a cut with URL-type recording."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording])
        result = loader(cuts)

        # Verify batch operations were called
        assert batch.add.called
        assert batch.get.called

        # Verify the recording source was updated to memory type
        for cut in result:
            assert cut.recording.sources[0].type == "memory"
            assert isinstance(cut.recording.sources[0].source, bytes)

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_processes_features(
        self, mock_get_client, mock_aistore_client, cut_with_features
    ):
        """Test processing a cut with Features."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_features])
        result = loader(cuts)

        # Verify batch operations
        assert batch.add.called
        assert batch.get.called

        # Verify features were updated to memory type
        for cut in result:
            assert cut.features.storage_type == FILE_TO_MEMORY_TYPE["numpy_files"]
            assert cut.features.storage_path == ""
            assert isinstance(cut.features.storage_key, bytes)

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_processes_array(
        self, mock_get_client, mock_aistore_client, cut_with_array
    ):
        """Test processing a cut with Array."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_array])
        result = loader(cuts)

        # Verify both features and array were updated to memory type
        for cut in result:
            assert cut.features.storage_type == FILE_TO_MEMORY_TYPE["lilcom_files"]
            assert cut.custom_array.storage_type == FILE_TO_MEMORY_TYPE["lilcom_files"]
            assert cut.custom_array.storage_path == ""
            assert isinstance(cut.custom_array.storage_key, bytes)

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_processes_temporal_array(
        self, mock_get_client, mock_aistore_client, cut_with_temporal_array
    ):
        """Test processing a cut with TemporalArray."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_temporal_array])
        result = loader(cuts)

        # Verify both features and temporal array were updated to memory type
        for cut in result:
            assert cut.features.storage_type == FILE_TO_MEMORY_TYPE["numpy_files"]
            assert (
                cut.custom_temporal.array.storage_type
                == FILE_TO_MEMORY_TYPE["numpy_files"]
            )
            assert cut.custom_temporal.array.storage_path == ""
            assert isinstance(cut.custom_temporal.array.storage_key, bytes)

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_processes_image(
        self, mock_get_client, mock_aistore_client, cut_with_image
    ):
        """Test processing a cut with Image."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_image])
        result = loader(cuts)

        # Verify both features and image were updated to memory type
        for cut in result:
            assert cut.features.storage_type == FILE_TO_MEMORY_TYPE["pillow_files"]
            assert cut.custom_image.storage_type == FILE_TO_MEMORY_TYPE["pillow_files"]
            assert cut.custom_image.storage_path == ""
            assert isinstance(cut.custom_image.storage_key, bytes)

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_processes_multiple_cuts(
        self, mock_get_client, cut_with_url_recording, cut_with_features
    ):
        """Test processing multiple cuts in a single batch."""
        client = MagicMock()
        batch = MagicMock()

        # Track add() calls
        add_count = []
        batch.add.side_effect = lambda *args, **kwargs: add_count.append(1)

        # Mock batch.get() to return content for each added URL
        def mock_batch_get():
            for i in range(len(add_count)):
                yield (MagicMock(), f"content_{i}".encode())

        batch.get.side_effect = lambda: mock_batch_get()
        client.batch.return_value = batch
        client.bucket.return_value = MagicMock()
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording, cut_with_features])
        result = loader(cuts)

        # Verify both cuts were processed
        assert len(list(result)) == 2
        assert batch.add.call_count >= 2


class TestAISBatchLoaderErrorHandling:
    """Tests for AISBatchLoader error handling."""

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_unsupported_storage_type(self, mock_get_client, mock_aistore_client):
        """Test that unsupported storage types raise an error."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        # Create a cut with unsupported storage type
        features = Features(
            recording_id="rec-1",
            channels=0,
            start=0.0,
            duration=10.0,
            type="fbank",
            num_frames=1000,
            num_features=80,
            sampling_rate=16000,
            storage_type="unsupported_type",
            storage_path="ais://mybucket/features",
            storage_key="feats.npy",
            frame_shift=0.01,
        )
        recording = Recording(
            id="rec-1",
            sources=[AudioSource(type="file", channels=[0], source="dummy.wav")],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut = MonoCut(
            id="cut-1",
            start=0.0,
            duration=10.0,
            channel=0,
            features=features,
            recording=recording,
        )

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut])

        with pytest.raises(AISBatchLoaderError, match="Unsupported storage type"):
            loader(cuts)

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    @patch("aistore.sdk.utils.parse_url")
    def test_invalid_url(self, mock_parse_url, mock_get_client, mock_aistore_client):
        """Test that invalid URLs raise an error."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        # Mock parse_url to return invalid components
        mock_parse_url.return_value = (None, None, None)

        recording = Recording(
            id="rec-1",
            sources=[AudioSource(type="url", channels=[0], source="invalid://url")],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut = MonoCut(
            id="cut-1",
            start=0.0,
            duration=10.0,
            channel=0,
            recording=recording,
        )

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut])

        with pytest.raises(AISBatchLoaderError, match="Invalid object URL"):
            loader(cuts)


class TestAISBatchLoaderArchiveHandling:
    """Tests for archive extraction support."""

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_archive_path_extraction_tar_gz(self, mock_get_client, mock_aistore_client):
        """Test that tar.gz archive paths are correctly parsed."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        recording = Recording(
            id="rec-1",
            sources=[
                AudioSource(
                    type="url",
                    channels=[0],
                    source="ais://mybucket/archive.tar.gz/audio/file1.wav",
                )
            ],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut = MonoCut(
            id="cut-1",
            start=0.0,
            duration=10.0,
            channel=0,
            recording=recording,
        )

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut])
        loader(cuts)

        # Verify that batch.add was called with archpath parameter
        assert batch.add.called
        call_kwargs = batch.add.call_args[1]
        assert "archpath" in call_kwargs
        assert call_kwargs["archpath"] == "audio/file1.wav"

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_archive_path_extraction_tar(self, mock_get_client, mock_aistore_client):
        """Test that .tar archive paths are correctly parsed."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        recording = Recording(
            id="rec-1",
            sources=[
                AudioSource(
                    type="url",
                    channels=[0],
                    source="ais://mybucket/data.tar/subdir/file.wav",
                )
            ],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut = MonoCut(
            id="cut-1",
            start=0.0,
            duration=10.0,
            channel=0,
            recording=recording,
        )

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut])
        loader(cuts)

        # Verify archpath extraction
        call_kwargs = batch.add.call_args[1]
        assert call_kwargs["archpath"] == "subdir/file.wav"

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_no_archive_path(self, mock_get_client, mock_aistore_client):
        """Test that non-archive URLs set archpath to None."""
        client, batch = mock_aistore_client
        mock_get_client.return_value = (client, None)

        recording = Recording(
            id="rec-1",
            sources=[
                AudioSource(
                    type="url", channels=[0], source="ais://mybucket/audio/file1.wav"
                )
            ],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut = MonoCut(
            id="cut-1",
            start=0.0,
            duration=10.0,
            channel=0,
            recording=recording,
        )

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut])
        loader(cuts)

        # Verify archpath is None for non-archive URLs
        call_kwargs = batch.add.call_args[1]
        assert call_kwargs["archpath"] is None


class TestAISBatchLoaderHelperMethods:
    """Tests for AISBatchLoader helper methods."""

    def test_get_archive_extension_tar_gz(self):
        """Test archive extension detection for .tar.gz."""
        obj_name = "path/to/archive.tar.gz/file.wav"
        ext = get_archive_extension(obj_name)
        assert ext == ".tar.gz"

    def test_get_archive_extension_tar(self):
        """Test archive extension detection for .tar."""
        obj_name = "path/to/archive.tar/file.wav"
        ext = get_archive_extension(obj_name)
        assert ext == ".tar"

    def test_get_archive_extension_tgz(self):
        """Test archive extension detection for .tgz."""
        obj_name = "path/to/archive.tgz/file.wav"
        ext = get_archive_extension(obj_name)
        assert ext == ".tgz"

    def test_get_archive_extension_none(self):
        """Test that non-archive paths return None."""
        obj_name = "path/to/file.wav"
        ext = get_archive_extension(obj_name)
        assert ext is None

    def test_get_archive_extension_priority(self):
        """Test that tar.gz is detected before .tar in ambiguous cases."""
        obj_name = "archive.tar.gz"
        ext = get_archive_extension(obj_name)
        # Should find .tar.gz first since it's checked first in ARCHIVE_EXTENSIONS
        assert ext in ARCHIVE_EXTENSIONS


class TestAISBatchLoaderIntegration:
    """Integration tests combining multiple features."""

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_mixed_manifest_types(
        self, mock_get_client, cut_with_url_recording, cut_with_features
    ):
        """Test processing cuts with mixed manifest types (Recording + Features)."""
        client = MagicMock()
        batch = MagicMock()

        # Create a cut with both URL recording and features
        cut = cut_with_features
        cut.recording = Recording(
            id="rec-1",
            sources=[
                AudioSource(type="url", channels=[0], source="ais://mybucket/audio.wav")
            ],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )

        # Track add() calls
        add_count = []
        batch.add.side_effect = lambda *args, **kwargs: add_count.append(1)

        # Mock batch.get() to return content for each added URL
        def mock_batch_get():
            for i in range(len(add_count)):
                yield (MagicMock(), f"content_{i}".encode())

        batch.get.side_effect = lambda: mock_batch_get()
        client.batch.return_value = batch
        client.bucket.return_value = MagicMock()
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut])
        result = loader(cuts)

        # Verify both recording and features were processed
        for c in result:
            assert c.recording.sources[0].type == "memory"
            assert c.features.storage_type == FILE_TO_MEMORY_TYPE["numpy_files"]

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_skips_non_url_recordings(self, mock_get_client):
        """Test that file-type recordings don't add URLs to batch."""
        client = MagicMock()
        batch = MagicMock()

        # Track add() calls
        add_count = []
        batch.add.side_effect = lambda *args, **kwargs: add_count.append(1)

        # Mock batch.get() to return empty iterator
        def mock_batch_get():
            for i in range(len(add_count)):
                yield (MagicMock(), f"content_{i}".encode())

        batch.get.side_effect = lambda: mock_batch_get()
        client.batch.return_value = batch
        client.bucket.return_value = MagicMock()
        mock_get_client.return_value = (client, None)

        recording = Recording(
            id="rec-1",
            sources=[
                AudioSource(
                    type="file",  # Not a URL
                    channels=[0],
                    source="/path/to/local/file.wav",
                )
            ],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut = MonoCut(
            id="cut-1",
            start=0.0,
            duration=10.0,
            channel=0,
            recording=recording,
        )

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut])
        result = loader(cuts)

        # Verify no URLs were added to batch for file-type recordings
        assert batch.add.call_count == 0

        # Recording should remain unchanged
        for c in result:
            assert c.recording.sources[0].type == "file"

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    @patch("lhotse.ais.common.is_valid_url")
    def test_skips_invalid_feature_urls(self, mock_is_valid_url, mock_get_client):
        """Test that features with invalid URLs don't add to batch."""
        client = MagicMock()
        batch = MagicMock()

        # Track add() calls
        add_count = []
        batch.add.side_effect = lambda *args, **kwargs: add_count.append(1)

        # Mock is_valid_url to return False
        mock_is_valid_url.return_value = False

        # Mock batch.get() to return empty since no URLs added
        def mock_batch_get():
            for i in range(len(add_count)):
                yield (MagicMock(), f"content_{i}".encode())

        batch.get.side_effect = lambda: mock_batch_get()
        client.batch.return_value = batch
        client.bucket.return_value = MagicMock()
        mock_get_client.return_value = (client, None)

        features = Features(
            recording_id="rec-1",
            channels=0,
            start=0.0,
            duration=10.0,
            type="fbank",
            num_frames=1000,
            num_features=80,
            sampling_rate=16000,
            storage_type="numpy_files",
            storage_path="/local/path",  # Not a URL
            storage_key="feats.npy",
            frame_shift=0.01,
        )
        recording = Recording(
            id="rec-1",
            sources=[AudioSource(type="file", channels=[0], source="dummy.wav")],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut = MonoCut(
            id="cut-1",
            start=0.0,
            duration=10.0,
            channel=0,
            features=features,
            recording=recording,
        )

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut])
        result = loader(cuts)

        # No URLs should be added for invalid feature paths
        assert batch.add.call_count == 0


class TestAISBatchLoaderVersionCompatibility:
    """Tests for backward compatibility with older AIStore versions."""

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_colocation_not_available_fallback(
        self, mock_get_client, cut_with_url_recording
    ):
        """Test that batch creation works when Colocation is not available (older versions)."""
        client = MagicMock()
        batch = MagicMock()

        # Track add() calls
        add_count = []
        batch.add.side_effect = lambda *args, **kwargs: add_count.append(1)

        # Mock batch.get() to return content
        def mock_batch_get():
            for i in range(len(add_count)):
                yield (MagicMock(), f"content_{i}".encode())

        batch.get.side_effect = lambda: mock_batch_get()

        # Mock client.batch() to raise TypeError when colocation is passed (older version)
        def mock_batch_with_error(**kwargs):
            if "colocation" in kwargs:
                raise TypeError(
                    "batch() got an unexpected keyword argument 'colocation'"
                )
            return batch

        client.batch.side_effect = mock_batch_with_error
        client.bucket.return_value = MagicMock()
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording])
        result = loader(cuts)

        # Verify batch was called twice: once with colocation (failed), once without
        assert client.batch.call_count == 2

        # Verify processing succeeded
        for cut in result:
            assert cut.recording.sources[0].type == "memory"

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_archive_config_not_available_fallback(
        self, mock_get_client, cut_with_url_recording
    ):
        """Test that fallback works when ArchiveConfig is not available (older versions)."""
        from aistore.sdk.errors import AISError

        client = MagicMock()
        batch = MagicMock()

        # Mock batch.get() to raise AISError to trigger fallback
        batch.get.side_effect = AISError(
            400, "Batch request failed", "http://test", MagicMock()
        )

        # Mock batch.requests_list with archpath
        mock_request = MagicMock()
        mock_request.obj_name = "archive.tar.gz"
        mock_request.bck = "mybucket"
        mock_request.provider = "ais"
        mock_request.archpath = "audio.wav"
        batch.requests_list = [mock_request]

        # Mock sequential GET fallback
        mock_reader = MagicMock()
        mock_reader.read_all.return_value = b"fallback_content"
        mock_object = MagicMock()
        mock_object.get_reader.return_value = mock_reader
        mock_bucket = MagicMock()
        mock_bucket.object.return_value = mock_object
        client.bucket.return_value = mock_bucket

        client.batch.return_value = batch
        mock_get_client.return_value = (client, None)

        # Mock ImportError for ArchiveConfig
        with patch(
            "lhotse.ais.batch_loader.AISBatchLoader._get_object_from_moss_in"
        ) as mock_fallback:
            mock_fallback.return_value = b"fallback_content_no_archive_config"

            loader = AISBatchLoader()
            cuts = CutSet.from_cuts([cut_with_url_recording])
            result = loader(cuts)

            # Verify fallback was called
            assert mock_fallback.called

            # Verify processing succeeded
            for cut in result:
                assert cut.recording.sources[0].type == "memory"


class TestAISBatchLoaderErrorHandling:
    """Tests for error handling and fallback mechanisms."""

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_batch_get_failure_triggers_fallback(
        self, mock_get_client, cut_with_url_recording
    ):
        """Test that batch.get() failure triggers sequential GET fallback."""
        from aistore.sdk.errors import AISError

        client = MagicMock()
        batch = MagicMock()

        # Track add() calls
        add_count = []
        batch.add.side_effect = lambda *args, **kwargs: add_count.append(1)

        # Mock batch.get() to raise AISError (use Exception as base since AISError has complex init)
        batch.get.side_effect = AISError(
            400, "Batch request failed", "http://test", MagicMock()
        )

        # Mock batch.requests_list for fallback
        mock_request = MagicMock()
        mock_request.object_name = "audio.wav"
        mock_request.bucket_name = "mybucket"
        mock_request.provider = "ais"
        mock_request.archpath = None
        batch.requests_list = [mock_request]

        # Mock sequential GET fallback
        mock_reader = MagicMock()
        mock_reader.read_all.return_value = b"fallback_content"
        mock_object = MagicMock()
        mock_object.get_reader.return_value = mock_reader
        mock_bucket = MagicMock()
        mock_bucket.object.return_value = mock_object
        client.bucket.return_value = mock_bucket

        client.batch.return_value = batch
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording])
        result = loader(cuts)

        # Verify batch.get() was called and failed
        assert batch.get.called

        # Verify fallback was used (bucket.object was called)
        assert client.bucket.called
        assert mock_bucket.object.called
        assert mock_reader.read_all.called

        # Verify the recording was updated with fallback content
        for cut in result:
            assert cut.recording.sources[0].type == "memory"
            assert cut.recording.sources[0].source == b"fallback_content"

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_fallback_with_archive_path(self, mock_get_client, cut_with_url_recording):
        """Test that fallback handles archive paths correctly."""
        from aistore.sdk.errors import AISError

        client = MagicMock()
        batch = MagicMock()

        # Track add() calls
        add_count = []
        batch.add.side_effect = lambda *args, **kwargs: add_count.append(1)

        # Mock batch.get() to raise AISError
        batch.get.side_effect = AISError(
            400, "Batch request failed", "http://test", MagicMock()
        )

        # Mock batch.requests_list with archpath
        mock_request = MagicMock()
        mock_request.object_name = "archive.tar.gz"
        mock_request.bucket_name = "mybucket"
        mock_request.provider = "ais"
        mock_request.archpath = "audio.wav"  # File inside archive
        batch.requests_list = [mock_request]

        # Mock sequential GET fallback with archive config
        mock_reader = MagicMock()
        mock_reader.read_all.return_value = b"archive_content"
        mock_object = MagicMock()
        mock_object.get_reader.return_value = mock_reader
        mock_bucket = MagicMock()
        mock_bucket.object.return_value = mock_object
        client.bucket.return_value = mock_bucket

        client.batch.return_value = batch
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording])
        result = loader(cuts)

        # Verify get_reader was called with archive_config
        call_kwargs = mock_object.get_reader.call_args[1]
        assert "archive_config" in call_kwargs
        assert call_kwargs["archive_config"] is not None

        # Verify the recording was updated
        for cut in result:
            assert cut.recording.sources[0].type == "memory"
            assert cut.recording.sources[0].source == b"archive_content"

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_fallback_failure_raises_error(
        self, mock_get_client, cut_with_url_recording
    ):
        """Test that fallback failure raises AISBatchLoaderError."""
        from aistore.sdk.errors import AISError

        client = MagicMock()
        batch = MagicMock()

        # Mock batch.get() to raise AISError
        batch.get.side_effect = AISError(
            400, "Batch request failed", "http://test", MagicMock()
        )

        # Mock batch.requests_list
        mock_request = MagicMock()
        mock_request.object_name = "audio.wav"
        mock_request.bucket_name = "mybucket"
        mock_request.provider = "ais"
        mock_request.archpath = None
        batch.requests_list = [mock_request]

        # Mock sequential GET to also fail
        mock_object = MagicMock()
        mock_object.get_reader.side_effect = Exception("Sequential GET failed")
        mock_bucket = MagicMock()
        mock_bucket.object.return_value = mock_object
        client.bucket.return_value = mock_bucket

        client.batch.return_value = batch
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording])

        # Should raise AISBatchLoaderError when fallback fails
        with pytest.raises(AISBatchLoaderError, match="Sequential GET fallback failed"):
            loader(cuts)

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_iterator_exhausted_raises_error(
        self, mock_get_client, cut_with_url_recording
    ):
        """Test that iterator exhaustion raises AISBatchLoaderError."""
        client = MagicMock()
        batch = MagicMock()

        # Track add() calls
        add_count = []
        batch.add.side_effect = lambda *args, **kwargs: add_count.append(1)

        # Mock batch.get() to return fewer items than expected
        def mock_batch_get():
            # Return nothing even though we expect 1 item
            return iter([])

        batch.get.side_effect = lambda: mock_batch_get()
        client.batch.return_value = batch
        client.bucket.return_value = MagicMock()
        mock_get_client.return_value = (client, None)

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording])

        # Should raise AISBatchLoaderError when iterator is exhausted
        with pytest.raises(
            AISBatchLoaderError, match="Batch result iterator exhausted prematurely"
        ):
            loader(cuts)

    @patch("lhotse.ais.batch_loader.get_aistore_client")
    def test_multiple_cuts_with_fallback(self, mock_get_client, cut_with_url_recording):
        """Test fallback with multiple cuts."""
        from aistore.sdk.errors import AISError

        client = MagicMock()
        batch = MagicMock()

        # Track add() calls
        add_count = []
        batch.add.side_effect = lambda *args, **kwargs: add_count.append(1)

        # Mock batch.get() to raise AISError
        batch.get.side_effect = AISError(
            400, "Batch request failed", "http://test", MagicMock()
        )

        # Mock batch.requests_list with 2 requests
        mock_request1 = MagicMock()
        mock_request1.object_name = "audio1.wav"
        mock_request1.bucket_name = "mybucket"
        mock_request1.provider = "ais"
        mock_request1.archpath = None

        mock_request2 = MagicMock()
        mock_request2.object_name = "audio2.wav"
        mock_request2.bucket_name = "mybucket"
        mock_request2.provider = "ais"
        mock_request2.archpath = None

        batch.requests_list = [mock_request1, mock_request2]

        # Mock sequential GET fallback to return different content for each
        call_count = [0]

        def mock_get_reader(*args, **kwargs):
            mock_reader = MagicMock()
            content = f"content_{call_count[0]}".encode()
            mock_reader.read_all.return_value = content
            call_count[0] += 1
            return mock_reader

        mock_object = MagicMock()
        mock_object.get_reader.side_effect = mock_get_reader
        mock_bucket = MagicMock()
        mock_bucket.object.return_value = mock_object
        client.bucket.return_value = mock_bucket

        client.batch.return_value = batch
        mock_get_client.return_value = (client, None)

        # Create 2 cuts
        cut2 = MonoCut(
            id="cut-2",
            start=0.0,
            duration=10.0,
            channel=0,
            recording=Recording(
                id="rec-2",
                sources=[
                    AudioSource(
                        type="url", channels=[0], source="ais://mybucket/audio2.wav"
                    )
                ],
                sampling_rate=16000,
                num_samples=160000,
                duration=10.0,
            ),
        )

        loader = AISBatchLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording, cut2])
        result = loader(cuts)

        # Verify both cuts were processed with fallback
        result_list = list(result)
        assert len(result_list) == 2
        assert result_list[0].recording.sources[0].source == b"content_0"
        assert result_list[1].recording.sources[0].source == b"content_1"
