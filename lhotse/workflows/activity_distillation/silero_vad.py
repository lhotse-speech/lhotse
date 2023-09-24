"""
Silero VAD integration module

This module integrates Silero VAD (Voice Activity Detector),
a high-precision pre-trained model for voice activity detection in audio streams.

GitHub Repository: https://github.com/snakers4/silero-vad

Citations:

@misc{Silero VAD,
  author = {Silero Team},
  title = {
      Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD),
      Number Detector and Language Classifier
  },
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}

Please cite Silero VAD when used in your projects. Details can be found in the repository.
"""

from typing import Dict, List

import torch

from lhotse.audio.recording import Recording
from lhotse.supervision import SupervisionSegment

from .base import ActivityDetector


def _dicts_to_supervision(
    murkup: List[Dict[str, int]],
    recording_id: str,
    channel: int,
    sampling_rate: int,
) -> List[SupervisionSegment]:
    return [
        SupervisionSegment(
            id=f"{recording_id}-ch{channel}-silero_vad-{i:05}",
            recording_id=recording_id,
            start=state["start"] / sampling_rate,
            duration=(state["end"] - state["start"]) / sampling_rate,
            channel=channel,
        )
        for i, state in enumerate(murkup)
    ]


class SileroVAD(ActivityDetector):
    """Silero Voice Activity Detector model wrapper"""

    def __init__(self, device: str):
        super().__init__(device=device)
        self._model, utils = torch.hub.load(  # type: ignore
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        # get_speech_timestamps - function that returns speech timestamps
        self._predict = utils[0]
        self._model.to(self.device)

    def __call__(self, recording: Recording) -> List[SupervisionSegment]:
        """Predict voice activity for audio"""

        rate = recording.sampling_rate
        # TODO: convert to 16000 Hz or 8000 Hz

        audio_cpu = recording.load_audio()  # type: ignore
        audio = torch.Tensor(audio_cpu).to(self.device)

        result: List[SupervisionSegment] = []

        with torch.no_grad():
            for channel, track in enumerate(audio):
                murkup = self._predict(
                    audio=track,
                    model=self._model,
                    sampling_rate=rate,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=float("inf"),
                    min_silence_duration_ms=100,
                    window_size_samples=512,
                    speech_pad_ms=30,
                    return_seconds=False,
                    visualize_probs=False,
                    progress_tracking_callback=None,
                )
                supervisions = _dicts_to_supervision(
                    murkup=murkup,
                    recording_id=recording.id,
                    channel=channel,
                    sampling_rate=rate,
                )
                result.extend(supervisions)

        return result
