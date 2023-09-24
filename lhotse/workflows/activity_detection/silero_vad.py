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

import numpy as np
import torch

from .base import Activity, ActivityDetector


def _to_activity_maker(sampling_rate: int):
    def make_activity(state: Dict[str, float]) -> Activity:
        start, end = state["start"], state["end"]
        return Activity(
            start=start / sampling_rate,
            duration=(end - start) / sampling_rate,
        )

    return make_activity


class SileroVAD(ActivityDetector):
    """Silero Voice Activity Detector model wrapper"""

    def __init__(self, device: str):
        sampling_rate = 16000
        super().__init__(
            detector_name="silero_vad",
            device=device,
            sampling_rate=sampling_rate,
        )
        self._model, utils = torch.hub.load(  # type: ignore
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        # utils[0] := get_speech_timestamps - function that returns speech timestamps
        self._predict = utils[0]
        self._model.to(self.device)
        self._to_activity = _to_activity_maker(sampling_rate)

    def forward(self, track: np.ndarray) -> List[Activity]:
        """Predict voice activity for audio"""
        audio = torch.Tensor(track).to(self.device)

        with torch.no_grad():
            murkup = self._predict(
                audio=audio,
                model=self._model,
                sampling_rate=self._sampling_rate,
                min_speech_duration_ms=250,
                max_speech_duration_s=float("inf"),
                min_silence_duration_ms=100,
                window_size_samples=512,
                speech_pad_ms=30,
                return_seconds=False,
                visualize_probs=False,
                progress_tracking_callback=None,
            )

        return list(map(self._to_activity, murkup))
