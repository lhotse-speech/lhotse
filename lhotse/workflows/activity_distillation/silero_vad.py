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

from functools import partial
from typing import List

import torch

from lhotse.audio.recording import Recording
from lhotse.supervision import AlignmentItem

from .base import ActivityDetector


def _dict_to_aligment(state: dict[str, int], sampling_rate: int) -> AlignmentItem:
    return AlignmentItem(
        symbol="",
        start=state["start"] / sampling_rate,
        duration=(state["end"] - state["start"]) / sampling_rate,
    )


class SileroVAD(ActivityDetector):
    """Silero Voice Activity Detector model wrapper"""

    def __init__(self, device: str, onnx: bool = False, force_reload: bool = False):
        super().__init__(device=device)
        self._model, (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=force_reload,
            onnx=onnx,
        )

        self._predict = get_speech_timestamps

    def __call__(self, recording: Recording) -> List[AlignmentItem]:
        """Predict voice activity for audio"""

        # TODO: convert to mono?
        audio = recording.load_audio()
        rate = recording.sampling_rate

        murkup = self._predict(audio, self._model, sampling_rate=rate)
        to_aligment = partial(_dict_to_aligment, sampling_rate=rate)
        return list(map(to_aligment, murkup))
