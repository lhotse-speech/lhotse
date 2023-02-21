import pytest
import torch
from torch.utils.data import DataLoader

from lhotse import RecordingSet, Seconds, compute_num_samples
from lhotse.audio import torchaudio_supports_ffmpeg
from lhotse.dataset.unsupervised import (
    RecordingChunkIterableDataset,
    audio_chunk_collate,
    audio_chunk_worker_init_fn,
)


@pytest.mark.skipif(
    not torchaudio_supports_ffmpeg(), reason="Requires torchaudio 0.12.0+ to run."
)
@pytest.mark.parametrize("CHUNK_SHIFT", [10.0, 8.0])
def test_audio_chunk_dataset_usage(CHUNK_SHIFT: Seconds):
    CHUNK_SIZE: Seconds = 10.0
    SAMPLING_RATE = 16000
    EXPECTED_CHUNK_NUM_SAMPLES = compute_num_samples(CHUNK_SIZE, SAMPLING_RATE)

    # RecordingSet of 4 audio files
    recordings = RecordingSet.from_file("test/fixtures/libri/audio.json").repeat(4)

    dataset = RecordingChunkIterableDataset(
        recordings, chunk_size=CHUNK_SIZE, chunk_shift=CHUNK_SHIFT
    )

    dloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=audio_chunk_collate,
        num_workers=0,
        worker_init_fn=audio_chunk_worker_init_fn,
    )

    tot_items = 0
    for batch in dloader:
        assert isinstance(batch, dict)

        assert set(batch.keys()) == {"recording_id", "begin_time", "end_time", "audio"}

        assert isinstance(batch["recording_id"], list)
        assert len(batch["recording_id"]) == 2
        assert isinstance(batch["recording_id"][0], str)
        assert isinstance(batch["recording_id"][1], str)

        assert torch.is_tensor(batch["begin_time"])
        assert batch["begin_time"].shape == (2,)
        assert batch["begin_time"].dtype == torch.float

        assert torch.is_tensor(batch["end_time"])
        assert batch["end_time"].shape == (2,)
        assert batch["end_time"].dtype == torch.float

        assert torch.is_tensor(batch["audio"])
        assert batch["audio"].shape == (2, EXPECTED_CHUNK_NUM_SAMPLES)
        assert batch["audio"].dtype == torch.float

        tot_items += batch["audio"].shape[0]

    # Check that no items were duplicated in the dataloader.
    EXPECTED_TOT_ITEMS = (
        len(recordings) * 2
    )  # == 8 (because there are 2 chunks per recording)
    assert tot_items == EXPECTED_TOT_ITEMS
