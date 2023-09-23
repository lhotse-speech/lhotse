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
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from lhotse.supervision import AlignmentItem


def _dict_to_aligment(state: dict[str, int], sampling_rate: int) -> AlignmentItem:
    return AlignmentItem(
        symbol="voice",
        start=state["start"] / sampling_rate,
        duration=(state["end"] - state["start"]) / sampling_rate,
    )


class SileroVad:
    """Silero VAD model wrapper"""

    def __init__(
        self,
        sampling_rate: int = 16000,
        onnx: bool = False,
        force_reload: bool = False,
    ):
        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=force_reload,
            onnx=onnx,
        )
        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = utils
        self._rate = sampling_rate
        self._predict = get_speech_timestamps
        self._read_audio = read_audio

    def read_audio(
        self,
        path: Union[str, Path],
        sampling_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Read audio from file"""
        rate = sampling_rate or self._rate
        path = Path(path).expanduser().absolute()
        return self._read_audio(str(path), sampling_rate=rate)

    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor, str, Path],
        sampling_rate: Optional[int] = None,
    ) -> List[AlignmentItem]:
        """Predict voice activity for audio"""

        rate = sampling_rate or self._rate
        if isinstance(audio, (str, Path)):
            audio = self.read_audio(audio, sampling_rate=rate)
        murkup = self._predict(audio, self._model, sampling_rate=rate)

        to_aligment = partial(_dict_to_aligment, sampling_rate=rate)
        return list(map(to_aligment, murkup))
