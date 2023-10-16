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

import shutil
from contextlib import suppress
from pathlib import Path
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

    def __init__(
        self,
        *,
        device: str = "cpu",
        sampling_rate: int = 16_000,
        force_download: bool = False,
    ):
        if sampling_rate not in [8_000, 16_000]:  # pragma: no cover
            msg = (
                "Sampling rate must be either 8000 or 16000, ",
                f"but got {sampling_rate}",
            )
            raise ValueError(msg)
        super().__init__(
            detector_name=f"silero_vad_{sampling_rate//1000}k",
            device=device,
            sampling_rate=sampling_rate,
        )

        self._model, utils = self._get_model(force_download=force_download)

        # utils[0] := get_speech_timestamps - function that returns speech timestamps
        self._predict = utils[0]
        self._model.to(self.device)
        self._to_activity = _to_activity_maker(sampling_rate)

    @classmethod
    def _cache_dirs(cls) -> List[Path]:
        cache_dir = torch.hub.get_dir()  # type: ignore
        if not isinstance(cache_dir, str):  # pragma: no cover
            raise TypeError(f"Bad cache directory path. Got {cache_dir}")
        return list(Path(cache_dir).glob("snakers4_silero-vad_*"))

    @classmethod
    def _clear_cache(cls):  # pragma: no cover
        """Remove Silero VAD models from cache"""
        for directory in cls._cache_dirs():
            if directory.is_dir():
                shutil.rmtree(directory)

    @classmethod
    def _get_model(cls, *, force_download: bool = False):
        if force_download:  # pragma: no cover
            cls._clear_cache()
        if not cls._cache_dirs():  # pragma: no cover
            force_download = True

        config = {
            "repo_or_dir": "snakers4/silero-vad",
            "model": "silero_vad",
            "onnx": False,
            "force_reload": force_download,
        }
        with suppress(Exception):
            from pkg_resources import parse_version  # pylint: disable=C0415

            # trust_repo is a new parameter in torch.hub.load and requires torch >= 2.0
            if parse_version(torch.__version__) >= parse_version("1.12"):
                config["trust_repo"] = True
        return torch.hub.load(**config)  # type: ignore

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

    @classmethod
    def force_download(cls):  # pragma: no cover
        cls._get_model(force_download=True)


class SileroVAD8k(SileroVAD):
    def __init__(self, device: str = "cpu", *, force_download: bool = False):
        super().__init__(
            sampling_rate=8_000,
            device=device,
            force_download=force_download,
        )


class SileroVAD16k(SileroVAD):
    def __init__(self, device: str = "cpu", *, force_download: bool = False):
        super().__init__(
            sampling_rate=16_000,
            device=device,
            force_download=force_download,
        )
