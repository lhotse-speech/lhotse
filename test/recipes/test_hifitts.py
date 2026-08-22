import lhotse.recipes.hifitts as hifitts


class FakeManifest:
    def __init__(self, partition_id):
        self.partition_id = partition_id
        self.output_path = None

    def to_file(self, output_path):
        self.output_path = output_path


class FakeFuture:
    def __init__(self, partition_id):
        self.partition_id = partition_id

    def result(self):
        return FakeManifest(self.partition_id), FakeManifest(self.partition_id)


class FakeExecutor:
    def __init__(self, num_jobs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def submit(
        self,
        prepare_single_partition,
        raw_manifest_path,
        corpus_dir,
        speaker_id,
        clean_or_other,
    ):
        return FakeFuture(hifitts.to_partition_id(raw_manifest_path))


def test_prepare_hifitts_preserves_partition_identity_when_completion_reversed(
    tmp_path, monkeypatch
):
    partition_ids = ("92_clean_train", "6097_other_dev")
    for partition_id in partition_ids:
        speaker_id, clean_or_other, part = partition_id.split("_")
        (tmp_path / f"{speaker_id}_manifest_{clean_or_other}_{part}.json").touch()

    monkeypatch.setattr(hifitts, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        hifitts, "as_completed", lambda futures: reversed(list(futures))
    )

    output_dir = tmp_path / "manifests"
    manifests = hifitts.prepare_hifitts(tmp_path, output_dir=output_dir, num_jobs=2)

    for partition_id in partition_ids:
        recordings = manifests[partition_id]["recordings"]
        supervisions = manifests[partition_id]["supervisions"]

        assert recordings.partition_id == partition_id
        assert recordings.output_path == (
            output_dir / f"hifitts_recordings_{partition_id}.jsonl.gz"
        )
        assert supervisions.partition_id == partition_id
        assert supervisions.output_path == (
            output_dir / f"hifitts_supervisions_{partition_id}.jsonl.gz"
        )
