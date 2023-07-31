import pytest
import torch
from torch import tensor
from torch.utils.data import DataLoader

from lhotse import Fbank, FbankConfig
from lhotse.cut import CutSet
from lhotse.dataset import RandomizedSmoothing
from lhotse.dataset.cut_transforms import CutConcatenate, CutMix
from lhotse.dataset.cut_transforms.extra_padding import ExtraPadding
from lhotse.dataset.input_strategies import AudioSamples, OnTheFlyFeatures
from lhotse.dataset.sampling import SimpleCutSampler
from lhotse.dataset.speech_recognition import K2SpeechRecognitionDataset
from lhotse.testing.dummies import DummyManifest


@pytest.fixture
def libri_cut_set():
    return CutSet.from_json("test/fixtures/libri/cuts.json")


@pytest.fixture
def k2_cut_set(libri_cut_set):
    # Create a cut set with 4 cuts, one of them having two supervisions
    return CutSet.from_cuts(
        [
            libri_cut_set[0],
            libri_cut_set[0].with_id("copy-1"),
            libri_cut_set[0].with_id("copy-2"),
            libri_cut_set[0].append(libri_cut_set[0]),
        ]
    )


@pytest.mark.parametrize("num_workers", [0, 1])
def test_k2_speech_recognition_iterable_dataset(k2_cut_set, num_workers):
    dataset = K2SpeechRecognitionDataset(cut_transforms=[CutConcatenate()])
    sampler = SimpleCutSampler(k2_cut_set, shuffle=False, max_cuts=1000)
    # Note: "batch_size=None" disables the automatic batching mechanism,
    #       which is required when Dataset takes care of the collation itself.
    dloader = DataLoader(
        dataset, batch_size=None, sampler=sampler, num_workers=num_workers
    )
    batch = next(iter(dloader))
    assert batch["inputs"].shape == (4, 2000, 40)
    # Each list has 5 items, to account for:
    # one cut with two supervisions + 3 three cuts with one supervision
    assert (batch["supervisions"]["sequence_idx"] == tensor([0, 0, 1, 2, 3])).all()
    assert (
        batch["supervisions"]["text"] == ["EXAMPLE OF TEXT"] * 5
    )  # a list, not tensor
    assert (batch["supervisions"]["start_frame"] == tensor([0, 1000, 0, 0, 0])).all()
    assert (batch["supervisions"]["num_frames"] == tensor([1000] * 5)).all()


@pytest.mark.parametrize("num_workers", [2, 3, 4])
def test_k2_speech_recognition_iterable_dataset_multiple_workers(
    k2_cut_set, num_workers
):
    k2_cut_set = k2_cut_set.pad()
    dataset = K2SpeechRecognitionDataset(cut_transforms=[CutConcatenate()])
    sampler = SimpleCutSampler(k2_cut_set, shuffle=False, max_cuts=1000)
    dloader = DataLoader(
        dataset, batch_size=None, sampler=sampler, num_workers=num_workers
    )

    # We expect a variable number of batches for each parametrized num_workers value,
    # because the dataset is small with 4 cuts that are partitioned across the workers.
    batches = [item for item in dloader]

    features = torch.cat([b["inputs"] for b in batches])
    assert features.shape == (4, 2000, 40)
    text = [t for b in batches for t in b["supervisions"]["text"]]
    assert text == ["EXAMPLE OF TEXT"] * 5  # a list, not tensor
    start_frame = torch.cat(
        [b["supervisions"]["start_frame"] for b in batches]
    ).tolist()
    # The multi-worker dataloader might not preserve order, because the workers
    # might finish processing in different order. To compare ground truth
    # start times with actual start times, we need to sort.
    start_frame = sorted(start_frame)
    assert start_frame == [0] * 4 + [1000]
    num_frames = torch.cat([b["supervisions"]["num_frames"] for b in batches]).tolist()
    assert num_frames == [1000] * 5


def test_k2_speech_recognition_iterable_dataset_shuffling():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    dataset = K2SpeechRecognitionDataset(
        return_cuts=True,
        cut_transforms=[
            CutConcatenate(),
        ],
    )
    sampler = SimpleCutSampler(
        cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_duration=10.0,
    )
    dloader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=2)
    dloader_cut_ids = []
    batches = []
    for batch in dloader:
        batches.append(batch)
        dloader_cut_ids.extend(c.id for c in batch["supervisions"]["cut"])

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(dloader_cut_ids) == len(cut_set)
    # Invariant 2: the items are not duplicated
    assert len(set(dloader_cut_ids)) == len(dloader_cut_ids)
    # Invariant 3: the items are shuffled, i.e. the order is different than that in the CutSet
    assert dloader_cut_ids != [c.id for c in cut_set]


def test_k2_speech_recognition_iterable_dataset_low_max_duration(k2_cut_set):
    dataset = K2SpeechRecognitionDataset()
    sampler = SimpleCutSampler(k2_cut_set, shuffle=False, max_duration=0.02)
    dloader = DataLoader(dataset, sampler=sampler, batch_size=None)
    # Check that it does not crash
    for batch in dloader:
        # There will be only a single item in each batch as we're exceeding the limit each time.
        assert batch["inputs"].shape[0] == 1


@pytest.fixture
def k2_noise_cut_set(libri_cut_set):
    # Create a cut set with 4 cuts, one of them having two supervisions
    return CutSet.from_cuts(
        [
            libri_cut_set[0].with_id("noise-1").truncate(duration=3.5),
            libri_cut_set[0].with_id("noise-2").truncate(duration=7.3),
        ]
    )


def test_k2_speech_recognition_augmentation(k2_cut_set, k2_noise_cut_set):
    dataset = K2SpeechRecognitionDataset(
        cut_transforms=[CutConcatenate(), CutMix(k2_noise_cut_set)]
    )
    sampler = SimpleCutSampler(k2_cut_set, shuffle=False, max_cuts=1000)
    dloader = DataLoader(dataset, sampler=sampler, batch_size=None)
    # Check that it does not crash by just running all dataloader iterations
    batches = [item for item in dloader]
    assert len(batches) > 0


def test_extra_padding_transform(k2_cut_set):
    transform = ExtraPadding(extra_frames=20)
    padded_cuts = transform(k2_cut_set)
    for cut, padded in zip(k2_cut_set, padded_cuts):
        # first track is for padding
        assert padded.tracks[0].cut.num_frames == 10
        # second track is for padding
        assert padded.tracks[-1].cut.num_frames == 10
        # total num frames is OK
        assert padded.num_frames == cut.num_frames + 20


@pytest.mark.parametrize("use_batch_extract", [True, False])
@pytest.mark.parametrize("fault_tolerant", [True, False])
def test_k2_speech_recognition_on_the_fly_feature_extraction(
    k2_cut_set, use_batch_extract, fault_tolerant
):
    precomputed_dataset = K2SpeechRecognitionDataset()
    on_the_fly_dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(
            Fbank(FbankConfig(num_mel_bins=40)),
            use_batch_extract=use_batch_extract,
            fault_tolerant=fault_tolerant,
        )
    )
    sampler = SimpleCutSampler(k2_cut_set, shuffle=False, max_cuts=1)
    for cut_ids in sampler:
        batch_pc = precomputed_dataset[cut_ids]
        batch_otf = on_the_fly_dataset[cut_ids]

        # Check that the features do not differ too much.
        norm_pc = torch.linalg.norm(batch_pc["inputs"])
        norm_diff = torch.linalg.norm(batch_pc["inputs"] - batch_otf["inputs"])
        # The precomputed and on-the-fly features are different due to mixing in time/fbank domains
        # and lilcom compression.
        assert norm_diff < 0.01 * norm_pc

        # Check that the supervision boundaries are the same.
        assert (
            batch_pc["supervisions"]["start_frame"]
            == batch_otf["supervisions"]["start_frame"]
        ).all()
        assert (
            batch_pc["supervisions"]["num_frames"]
            == batch_otf["supervisions"]["num_frames"]
        ).all()


def test_k2_speech_recognition_on_the_fly_feature_extraction_with_randomized_smoothing(
    k2_cut_set,
):
    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(
            extractor=Fbank(),
        )
    )
    rs_dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(
            extractor=Fbank(),
            # Use p=1.0 to ensure that smoothing is applied in this test.
            wave_transforms=[RandomizedSmoothing(sigma=0.5, p=1.0)],
        )
    )
    sampler = SimpleCutSampler(k2_cut_set, shuffle=False, max_cuts=1)
    for cut_ids in sampler:
        batch = dataset[cut_ids]
        rs_batch = rs_dataset[cut_ids]
        # Additive noise should cause the energies to go up
        assert (rs_batch["inputs"] - batch["inputs"]).sum() > 0


def test_k2_speech_recognition_audio_inputs(k2_cut_set):
    on_the_fly_dataset = K2SpeechRecognitionDataset(
        input_strategy=AudioSamples(),
    )
    # all cuts in one batch
    sampler = SimpleCutSampler(k2_cut_set, shuffle=False, max_cuts=1000)
    cut_ids = next(iter(sampler))
    batch = on_the_fly_dataset[cut_ids]
    assert batch["inputs"].shape == (4, 320000)
    # Each list has 5 items, to account for:
    # one cut with two supervisions + 3 three cuts with one supervision
    assert (batch["supervisions"]["sequence_idx"] == tensor([0, 0, 1, 2, 3])).all()
    assert (
        batch["supervisions"]["text"] == ["EXAMPLE OF TEXT"] * 5
    )  # a list, not tensor
    assert (batch["supervisions"]["start_sample"] == tensor([0, 160000, 0, 0, 0])).all()
    assert (batch["supervisions"]["num_samples"] == tensor([160000] * 5)).all()


def test_k2_speech_recognition_audio_inputs_with_workers_in_input_strategy(k2_cut_set):
    on_the_fly_dataset = K2SpeechRecognitionDataset(
        input_strategy=AudioSamples(num_workers=2),
    )
    # all cuts in one batch
    sampler = SimpleCutSampler(k2_cut_set, shuffle=False, max_duration=100000.0)
    dloader = DataLoader(
        on_the_fly_dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=0,  # has to be 0 because DataLoader workers can't spawn subprocesses
    )
    batch = next(iter(dloader))
    assert batch["inputs"].shape == (4, 320000)
    # Each list has 5 items, to account for:
    # one cut with two supervisions + 3 three cuts with one supervision
    assert (batch["supervisions"]["sequence_idx"] == tensor([0, 0, 1, 2, 3])).all()
    assert (
        batch["supervisions"]["text"] == ["EXAMPLE OF TEXT"] * 5
    )  # a list, not tensor
    assert (batch["supervisions"]["start_sample"] == tensor([0, 160000, 0, 0, 0])).all()
    assert (batch["supervisions"]["num_samples"] == tensor([160000] * 5)).all()
