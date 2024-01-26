# Activity Detection

The Activity Detection module provides tools for detecting activity in audio recordings. It includes the following components:

- `ActivityDetector` base class, designed to facilitate the integration of new activity detection methods in the future.
- A runner class named `ActivityDetectionProcessor` for parallel execution of activity detection on the `RecordingSet``.
- Two classes, `SileroVAD8k` and `SileroVAD16k`, which integrate the [Silero VAD](https://github.com/snakers4/silero-vad) model for activity detection.


## Example usage in command line interface (CLI)

1. Read the Help

    ```bash
    lhotse workflows activity-detection --help
    ```

2. Run activity detection using the Silero VAD model:

    ```bash
    lhotse workflows activity-detection \
    --model-name silero_vad_16k \
    --recordings-manifest data/librispeech_recordings_train-clean-5.jsonl.gz \
    --output-supervisions-manifest librispeech_recordings_train-clean-5.jsonl.gz \
    --jobs 2 \
    --device cpu
    ```

    or on a GPU (cuda):

    ```bash
    lhotse workflows activity-detection \
    --model-name silero_vad_16k \
    --recordings-manifest data/librispeech_recordings_train-clean-5.jsonl.gz \
    --output-supervisions-manifest librispeech_recordings_train-clean-5.jsonl.gz \
    --jobs 2 \
    --device cuda
    ```

    Output:

    ```bash
    Loading recordings from data/librispeech_recordings_train-clean-5.jsonl.gz...
    Making activity detection processor for 'silero_vad_16k'...
    Running activity detection using 'silero_vad_16k'...
    Using cache found in ~/.cache/torch/hub/snakers4_silero-vad_master
    ...
    Detecting activities: 100%|████████████████| 1519/1519 [04:50<00:00,  5.22rec/s]
    Saving 'silero_vad_16k' results ...
    Results saved to:
    .../librispeech_recordings_train-clean-5.jsonl.gz
    ```

## Trubleshooting

If you encounter the following errors while running the activity detection.

- **FileNotFoundError**: No such file or directory: /.../.cache/...
- **ValueError**: The provided filename /.../.cache/... does not exist.

Try to run the activity detection workflow with the `--force_download` flag. It will clear the cache and download the model again.


## Example usage in python code

### Silero VAD for single Recording

```python
from lhotse.audio import RecordingSet
from lhotse.workflows.activity_detection import SileroVAD16k

vad = SileroVAD16k(device="cpu") # or device="cuda"

recordings = RecordingSet.from_file("data/librispeech_recordings_train-clean-5.jsonl.gz")
record = recordings[25]

vad(record)
```

Output (a list of SupervisionSegment objects indicating detected activities in the recording):

```bash
[SupervisionSegment(id='6272-70171-0025-silero_vad_16k-0-00000', recording_id='6272-70171-0025', start=0.194, duration=2.396, channel=0, text=None, language=None, speaker=None, gender=None, custom=None, alignment=None),
 SupervisionSegment(id='6272-70171-0025-silero_vad_16k-0-00001', recording_id='6272-70171-0025', start=3.682, duration=1.02, channel=0, text=None, language=None, speaker=None, gender=None, custom=None, alignment=None),
 SupervisionSegment(id='6272-70171-0025-silero_vad_16k-0-00002', recording_id='6272-70171-0025', start=4.994, duration=0.956, channel=0, text=None, language=None, speaker=None, gender=None, custom=None, alignment=None),
 SupervisionSegment(id='6272-70171-0025-silero_vad_16k-0-00003', recording_id='6272-70171-0025', start=6.146, duration=2.652, channel=0, text=None, language=None, speaker=None, gender=None, custom=None, alignment=None),
 SupervisionSegment(id='6272-70171-0025-silero_vad_16k-0-00004', recording_id='6272-70171-0025', start=9.122, duration=4.316, channel=0, text=None, language=None, speaker=None, gender=None, custom=None, alignment=None),
 SupervisionSegment(id='6272-70171-0025-silero_vad_16k-0-00005', recording_id='6272-70171-0025', start=13.634, duration=3.006, channel=0, text=None, language=None, speaker=None, gender=None, custom=None, alignment=None)]
 ```

### Silero VAD for a RecordingSet

```python
from lhotse.audio import RecordingSet
from lhotse.cut import CutSet
from lhotse.workflows.activity_detection import ActivityDetectionProcessor, SileroVAD16k


recordings = RecordingSet.from_file("data/librispeech_recordings_train-clean-5.jsonl.gz")

processor = ActivityDetectionProcessor(SileroVAD16k, num_jobs=2, device="cuda", verbose=True)
supervisions = processor(recordings)

cutset = CutSet.from_manifests(recordings=recordings, supervisions = supervisions)
cutset[25]
```

Output (a Cut object with detected activities):

```bash
MonoCut(
    id='6272-70171-0025-25',
    start=0,
    duration=16.64,
    channel=0,
    supervisions=[
        SupervisionSegment(
            id='6272-70171-0025-silero_vad_16k-0-00000',
            recording_id='6272-70171-0025',
            start=0.194,
            duration=2.396,
            channel=0,
            text=None,
            language=None,
            speaker=None,
            gender=None,
            custom=None,
            alignment=None
        ),
        ...,
        SupervisionSegment(
            id='6272-70171-0025-silero_vad_16k-0-00005',
            recording_id='6272-70171-0025',
            start=13.634,
            duration=3.006,
            channel=0,
            text=None,
            language=None,
            speaker=None,
            gender=None,
            custom=None,
            alignment=None
        )
    ],
    features=None,
    recording=Recording(
        id='6272-70171-0025',
        sources=[
            AudioSource(
                type='file',
                channels=[0],
                source='data/LibriSpeech/train-clean-5/6272/70171/6272-70171-0025.flac'
            )
        ],
        sampling_rate=16000,
        num_samples=266240,
        duration=16.64,
        channel_ids=[0],
        transforms=None
    ),
    custom=None
)
```
