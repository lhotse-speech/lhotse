# Activity Detection

The Activity Detection module provides tools for detecting activity in audio recordings. It includes the following components:

- `ActivityDetector` base class, designed to facilitate the integration of new activity detection methods in the future.
- Two classes, `SileroVAD8k` and `SileroVAD16k`, which integrate the [Silero VAD](https://github.com/snakers4/silero-vad) model for activity detection.



## Example usage in command line interface (CLI)

1. Read the Help

    ```bash
    lhotse workflows detect-activity --help
    ```

2. Run activity detection using the Silero VAD model:

    ```bash
    lhotse workflows detect-activity \
    --model-name silero_vad_16k \
    --recordings-manifest data/librispeech_recordings_train-clean-5.jsonl.gz \
    --output-supervisions-manifest librispeech_recordings_train-clean-5.jsonl.gz \
    --jobs 2 \
    --device cpu
    ```

    or on a GPU (cuda):

    ```bash
    lhotse workflows detect-activity \
    --model-name silero_vad_16k \
    --recordings-manifest data/librispeech_recordings_train-clean-5.jsonl.gz \
    --output-supervisions-manifest librispeech_recordings_train-clean-5.jsonl.gz \
    --jobs 2 \
    --device cuda
    ```

    Output:

    ```bash
    Checking model state in cache...
        Using cache found in /.../.cache/torch/hub/snakers4_silero-vad_master
    Making activity detection processor for 'silero_vad_16k'...
    Loading recordings from /.../data/librispeech_recordings_train-clean-5.jsonl.gz...
    Multiprocessing:   0%|                               | 0/1519 [00:00<?, ?task/s]
        Using cache found in /.../.cache/torch/hub/snakers4_silero-vad_master
        Using cache found in /.../.cache/torch/hub/snakers4_silero-vad_master
    Multiprocessing: 100%|████████████████████| 1519/1519 [07:13<00:00,  3.50task/s]
    Saving 'silero_vad_16k' results ...
    Results saved to:
    /.../librispeech_recordings_train-clean-5.jsonl.gz
    ```

## Trubleshooting

If you encounter the following errors while running the activity detection.

- **FileNotFoundError**: No such file or directory: /.../.cache/...
- **ValueError**: The provided filename /.../.cache/... does not exist.

Try to run the activity detection workflow with the `--force_download` flag. It will clear the cache and download the model again.


## Example usage in python code

### Silero VAD for single Recording

```python
from pathlib import Path

import lhotse
from lhotse.audio import RecordingSet
from lhotse.workflows.activity_detection import SileroVAD16k

vad = SileroVAD16k(device="cpu")  # or device="cuda"

# load recordings
lhotse_root = Path(lhotse.__file__).parent.parent
recordings_path = lhotse_root / "test" / "fixtures" / "libri" / "audio.json"
recordings = RecordingSet.from_file(recordings_path).with_path_prefix(lhotse_root)
recordings = recordings.resample(vad.sampling_rate)

# run voice activity detection
vad(recordings[0].load_audio())

```

Detection result for the first recording:

```
[Activity(start=0.194, duration=2.204),
 Activity(start=2.754, duration=5.98),
 Activity(start=9.602, duration=6.438)]
```

### Silero VAD for a RecordingSet

```python
from pathlib import Path

import lhotse
from lhotse.audio import RecordingSet
from lhotse.workflows.activity_detection import detect_activity

# load recordings
lhotse_root = Path(lhotse.__file__).parent.parent
recordings_path = lhotse_root / "test" / "fixtures" / "libri" / "audio.json"
recordings = RecordingSet.from_file(recordings_path).with_path_prefix(lhotse_root)
recordings = recordings.resample(vad.sampling_rate)

supervisions = detect_activity(recordings, "silero_vad_16k", device="cpu", verbose=True)
supervisions.segments
```

Detection result for all recordings:

```
Multiprocessing: 100%|██████████████████████████| 1/1 [00:04<00:00,  4.33s/task]
{'recording-1-silero_vad_16k-0-00000': SupervisionSegment(id='recording-1-silero_vad_16k-0-00000', recording_id='recording-1', start=0.194, duration=2.204, channel=0, text=None, language=None, speaker=None, gender=None, custom=None, alignment=None),
 'recording-1-silero_vad_16k-0-00001': SupervisionSegment(id='recording-1-silero_vad_16k-0-00001', recording_id='recording-1', start=2.754, duration=5.98, channel=0, text=None, language=None, speaker=None, gender=None, custom=None, alignment=None),
 'recording-1-silero_vad_16k-0-00002': SupervisionSegment(id='recording-1-silero_vad_16k-0-00002', recording_id='recording-1', start=9.602, duration=6.438, channel=0, text=None, language=None, speaker=None, gender=None, custom=None, alignment=None)}
```
