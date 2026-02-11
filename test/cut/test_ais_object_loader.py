"""
Unit tests for AISObjectLoader.

These tests use mocking to simulate AIStore client behavior,
allowing them to run in CI environments without AIStore infrastructure.
"""
from unittest.mock import MagicMock, patch

import pytest

from lhotse import CutSet, MonoCut
from lhotse.ais.common import ARCHIVE_EXTENSIONS, FILE_TO_MEMORY_TYPE, AISLoaderError
from lhotse.ais.object_loader import AISObjectLoader, AISObjectLoaderError
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


# ----------------------------- Fixtures -----------------------------


@pytest.fixture
def mock_aistore_client():
    """Mock AIStore client with object-level GET support."""
    client = MagicMock()

    # Default: return dummy bytes for any object read
    mock_reader = MagicMock()
    mock_reader.read_all.return_value = b"dummy_content"
    mock_object = MagicMock()
    mock_object.get_reader.return_value = mock_reader
    mock_bucket = MagicMock()
    mock_bucket.object.return_value = mock_object
    client.bucket.return_value = mock_bucket

    return client


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
        recording=None,
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
        recording=None,
        features=features,
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
        recording=None,
        features=features,
    )
    cut.custom_image = image
    return cut


@pytest.fixture
def cut_with_temporal_array():
    """Create a cut with TemporalArray stored in AIStore (no recording)."""
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
        recording=None,
        features=features,
    )
    cut.custom_temporal = temporal_array
    return cut


# ----------------------------- Init Tests -----------------------------


class TestAISObjectLoaderInit:
    """Tests for AISObjectLoader initialization."""

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_init_default_workers(self, mock_get_client):
        """Test that AISObjectLoader initializes with default max_workers=1."""
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, None)

        loader = AISObjectLoader()

        assert loader.client == mock_client
        assert loader.max_workers == 1
        mock_get_client.assert_called_once()

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_init_custom_workers(self, mock_get_client):
        """Test initialization with custom max_workers."""
        mock_get_client.return_value = (MagicMock(), None)

        loader = AISObjectLoader(max_workers=8)

        assert loader.max_workers == 8

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_init_invalid_workers(self, mock_get_client):
        """Test that invalid max_workers raises ValueError."""
        mock_get_client.return_value = (MagicMock(), None)

        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            AISObjectLoader(max_workers=0)

        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            AISObjectLoader(max_workers=-1)

    def test_init_missing_aistore(self):
        """Test that missing aistore raises ImportError."""
        with patch("lhotse.ais.object_loader.is_module_available", return_value=False):
            with pytest.raises(ImportError, match="pip install aistore"):
                AISObjectLoader()

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_exception_hierarchy(self, mock_get_client):
        """Test that AISObjectLoaderError is a subclass of AISLoaderError."""
        assert issubclass(AISObjectLoaderError, AISLoaderError)


# ----------------------------- Call Flow Tests -----------------------------


class TestAISObjectLoaderCall:
    """Tests for AISObjectLoader.__call__() method."""

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_rejects_lazy_cutset(self, mock_get_client, cut_with_url_recording):
        """Test that lazy CutSets are rejected."""
        mock_get_client.return_value = (MagicMock(), None)

        loader = AISObjectLoader()
        lazy_cuts = CutSet.from_cuts([cut_with_url_recording]).repeat()

        with pytest.raises(ValueError, match="Lazy CutSets cannot be used"):
            loader(lazy_cuts)

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_processes_recording_with_url(
        self, mock_get_client, mock_aistore_client, cut_with_url_recording
    ):
        """Test processing a cut with URL-type recording."""
        mock_get_client.return_value = (mock_aistore_client, None)

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording])
        result = loader(cuts)

        # Verify the recording source was updated to memory type
        for cut in result:
            assert cut.recording.sources[0].type == "memory"
            assert isinstance(cut.recording.sources[0].source, bytes)

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_processes_features(
        self, mock_get_client, mock_aistore_client, cut_with_features
    ):
        """Test processing a cut with Features."""
        mock_get_client.return_value = (mock_aistore_client, None)

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut_with_features])
        result = loader(cuts)

        for cut in result:
            assert cut.features.storage_type == FILE_TO_MEMORY_TYPE["numpy_files"]
            assert cut.features.storage_path == ""
            assert isinstance(cut.features.storage_key, bytes)

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_processes_array(
        self, mock_get_client, mock_aistore_client, cut_with_array
    ):
        """Test processing a cut with Array."""
        mock_get_client.return_value = (mock_aistore_client, None)

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut_with_array])
        result = loader(cuts)

        for cut in result:
            assert cut.features.storage_type == FILE_TO_MEMORY_TYPE["lilcom_files"]
            assert cut.custom_array.storage_type == FILE_TO_MEMORY_TYPE["lilcom_files"]
            assert cut.custom_array.storage_path == ""
            assert isinstance(cut.custom_array.storage_key, bytes)

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_processes_image(
        self, mock_get_client, mock_aistore_client, cut_with_image
    ):
        """Test processing a cut with Image."""
        mock_get_client.return_value = (mock_aistore_client, None)

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut_with_image])
        result = loader(cuts)

        for cut in result:
            assert cut.features.storage_type == FILE_TO_MEMORY_TYPE["pillow_files"]
            assert cut.custom_image.storage_type == FILE_TO_MEMORY_TYPE["pillow_files"]
            assert cut.custom_image.storage_path == ""
            assert isinstance(cut.custom_image.storage_key, bytes)

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_processes_temporal_array(
        self, mock_get_client, mock_aistore_client, cut_with_temporal_array
    ):
        """Test processing a cut with TemporalArray."""
        mock_get_client.return_value = (mock_aistore_client, None)

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut_with_temporal_array])
        result = loader(cuts)

        for cut in result:
            assert cut.features.storage_type == FILE_TO_MEMORY_TYPE["numpy_files"]
            assert (
                cut.custom_temporal.array.storage_type
                == FILE_TO_MEMORY_TYPE["numpy_files"]
            )
            assert cut.custom_temporal.array.storage_path == ""
            assert isinstance(cut.custom_temporal.array.storage_key, bytes)

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_processes_multiple_cuts(
        self, mock_get_client, cut_with_url_recording, cut_with_features
    ):
        """Test processing multiple cuts."""
        client = MagicMock()

        # Track calls and return different content each time
        call_count = [0]

        def mock_read_all():
            content = f"content_{call_count[0]}".encode()
            call_count[0] += 1
            return content

        mock_reader = MagicMock()
        mock_reader.read_all.side_effect = mock_read_all
        mock_object = MagicMock()
        mock_object.get_reader.return_value = mock_reader
        mock_bucket = MagicMock()
        mock_bucket.object.return_value = mock_object
        client.bucket.return_value = mock_bucket
        mock_get_client.return_value = (client, None)

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut_with_url_recording, cut_with_features])
        result = loader(cuts)

        assert len(list(result)) == 2

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_no_url_passthrough(self, mock_get_client):
        """Test that cuts without AIS URLs pass through unmodified."""
        mock_get_client.return_value = (MagicMock(), None)

        recording = Recording(
            id="rec-1",
            sources=[AudioSource(type="file", channels=[0], source="/local/file.wav")],
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

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut])
        result = loader(cuts)

        for c in result:
            assert c.recording.sources[0].type == "file"
            assert c.recording.sources[0].source == "/local/file.wav"


# ----------------------------- Concurrency Tests -----------------------------


class TestAISObjectLoaderConcurrency:
    """Tests for sequential and concurrent fetch paths."""

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_sequential_path(
        self, mock_get_client, mock_aistore_client, cut_with_url_recording
    ):
        """Test that max_workers=1 uses sequential fetching (no thread pool)."""
        mock_get_client.return_value = (mock_aistore_client, None)

        loader = AISObjectLoader(max_workers=1)
        cuts = CutSet.from_cuts([cut_with_url_recording])

        with patch.object(loader, "_fetch_sequential") as mock_seq, patch.object(
            loader, "_fetch_concurrent"
        ) as mock_conc:
            mock_seq.return_value = None
            loader(cuts)
            assert mock_seq.called
            assert not mock_conc.called

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_concurrent_path(
        self, mock_get_client, mock_aistore_client, cut_with_url_recording
    ):
        """Test that max_workers>1 uses concurrent fetching."""
        mock_get_client.return_value = (mock_aistore_client, None)

        loader = AISObjectLoader(max_workers=4)
        cuts = CutSet.from_cuts([cut_with_url_recording])

        with patch.object(loader, "_fetch_sequential") as mock_seq, patch.object(
            loader, "_fetch_concurrent"
        ) as mock_conc:
            mock_conc.return_value = None
            loader(cuts)
            assert not mock_seq.called
            assert mock_conc.called

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_concurrent_partial_failure(self, mock_get_client):
        """Test that concurrent fetch reports all errors."""
        client = MagicMock()
        call_count = [0]

        def mock_read_all():
            idx = call_count[0]
            call_count[0] += 1
            if idx == 1:
                raise Exception("Fetch failed for object 2")
            return f"content_{idx}".encode()

        mock_reader = MagicMock()
        mock_reader.read_all.side_effect = mock_read_all
        mock_object = MagicMock()
        mock_object.get_reader.return_value = mock_reader
        mock_bucket = MagicMock()
        mock_bucket.object.return_value = mock_object
        client.bucket.return_value = mock_bucket
        mock_get_client.return_value = (client, None)

        # Create two cuts with URLs
        rec1 = Recording(
            id="rec-1",
            sources=[
                AudioSource(
                    type="url", channels=[0], source="ais://mybucket/audio1.wav"
                )
            ],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        rec2 = Recording(
            id="rec-2",
            sources=[
                AudioSource(
                    type="url", channels=[0], source="ais://mybucket/audio2.wav"
                )
            ],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut1 = MonoCut(id="cut-1", start=0.0, duration=10.0, channel=0, recording=rec1)
        cut2 = MonoCut(id="cut-2", start=0.0, duration=10.0, channel=0, recording=rec2)

        loader = AISObjectLoader(max_workers=2)
        cuts = CutSet.from_cuts([cut1, cut2])

        with pytest.raises(AISObjectLoaderError, match="failed to fetch"):
            loader(cuts)


# ----------------------------- Error Handling Tests -----------------------------


class TestAISObjectLoaderErrorHandling:
    """Tests for error handling."""

    @patch("lhotse.ais.object_loader.get_aistore_client")
    @patch("aistore.sdk.utils.parse_url")
    def test_invalid_url(self, mock_parse_url, mock_get_client):
        """Test that invalid URLs raise an error."""
        mock_get_client.return_value = (MagicMock(), None)
        mock_parse_url.return_value = (None, None, None)

        recording = Recording(
            id="rec-1",
            sources=[AudioSource(type="url", channels=[0], source="invalid://url")],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut = MonoCut(
            id="cut-1", start=0.0, duration=10.0, channel=0, recording=recording
        )

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut])

        with pytest.raises(AISObjectLoaderError, match="Invalid object URL"):
            loader(cuts)

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_unsupported_storage_type(self, mock_get_client):
        """Test that unsupported storage types raise an error."""
        mock_get_client.return_value = (MagicMock(), None)

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

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut])

        with pytest.raises(AISLoaderError, match="Unsupported storage type"):
            loader(cuts)

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_fetch_failure(self, mock_get_client):
        """Test that fetch failures raise AISObjectLoaderError."""
        client = MagicMock()
        mock_object = MagicMock()
        mock_object.get_reader.side_effect = Exception("Connection refused")
        mock_bucket = MagicMock()
        mock_bucket.object.return_value = mock_object
        client.bucket.return_value = mock_bucket
        mock_get_client.return_value = (client, None)

        recording = Recording(
            id="rec-1",
            sources=[
                AudioSource(type="url", channels=[0], source="ais://mybucket/audio.wav")
            ],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0,
        )
        cut = MonoCut(
            id="cut-1", start=0.0, duration=10.0, channel=0, recording=recording
        )

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut])

        with pytest.raises(AISObjectLoaderError, match="Failed to fetch object"):
            loader(cuts)


# ----------------------------- Archive Handling Tests -----------------------------


class TestAISObjectLoaderArchiveHandling:
    """Tests for archive extraction support."""

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_archive_path_extraction_tar_gz(self, mock_get_client, mock_aistore_client):
        """Test that tar.gz archive paths are correctly parsed and ArchiveConfig used."""
        mock_get_client.return_value = (mock_aistore_client, None)

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
            id="cut-1", start=0.0, duration=10.0, channel=0, recording=recording
        )

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut])
        loader(cuts)

        # Verify that get_reader was called with archive_config
        mock_bucket = mock_aistore_client.bucket.return_value
        mock_obj = mock_bucket.object.return_value
        call_kwargs = mock_obj.get_reader.call_args[1]
        assert "archive_config" in call_kwargs
        assert call_kwargs["archive_config"] is not None

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_archive_path_extraction_tar(self, mock_get_client, mock_aistore_client):
        """Test that .tar archive paths are correctly parsed."""
        mock_get_client.return_value = (mock_aistore_client, None)

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
            id="cut-1", start=0.0, duration=10.0, channel=0, recording=recording
        )

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut])
        loader(cuts)

        # Verify bucket/object were called with correct names
        mock_aistore_client.bucket.assert_called()
        mock_bucket = mock_aistore_client.bucket.return_value
        mock_bucket.object.assert_called_with("data.tar")

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_no_archive_path(self, mock_get_client, mock_aistore_client):
        """Test that non-archive URLs don't use ArchiveConfig."""
        mock_get_client.return_value = (mock_aistore_client, None)

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
            id="cut-1", start=0.0, duration=10.0, channel=0, recording=recording
        )

        loader = AISObjectLoader()
        cuts = CutSet.from_cuts([cut])
        loader(cuts)

        # Verify get_reader was called with archive_config=None
        mock_bucket = mock_aistore_client.bucket.return_value
        mock_obj = mock_bucket.object.return_value
        call_kwargs = mock_obj.get_reader.call_args[1]
        assert call_kwargs["archive_config"] is None


# ----------------------------- AudioSamples Integration Tests -----------------------------


class TestAudioSamplesObjectLoaderIntegration:
    """Tests for AudioSamples integration with AISObjectLoader."""

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_object_loader_delegation(
        self, mock_get_client, mock_aistore_client, cut_with_url_recording
    ):
        """Test that AudioSamples delegates to AISObjectLoader when enabled."""
        mock_get_client.return_value = (mock_aistore_client, None)

        from lhotse.dataset.input_strategies import AudioSamples

        strategy = AudioSamples(use_object_loader=True, object_loader_max_workers=2)

        assert strategy.ais_object_loader is not None
        assert strategy.ais_batch_loader is None
        assert strategy.use_object_loader is True

    def test_mutual_exclusion(self):
        """Test that enabling both loaders raises ValueError."""
        from lhotse.dataset.input_strategies import AudioSamples

        with pytest.raises(ValueError, match="Cannot enable both"):
            AudioSamples(use_batch_loader=True, use_object_loader=True)

    @patch("lhotse.ais.object_loader.get_aistore_client")
    def test_object_loader_custom_workers(self, mock_get_client):
        """Test that custom max_workers is passed through."""
        mock_get_client.return_value = (MagicMock(), None)

        from lhotse.dataset.input_strategies import AudioSamples

        strategy = AudioSamples(use_object_loader=True, object_loader_max_workers=8)

        assert strategy.ais_object_loader.max_workers == 8
