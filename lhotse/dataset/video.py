from typing import Any, Dict

import torch

from lhotse import CutSet
from lhotse.dataset.collation import collate_video


class UnsupervisedAudioVideoDataset(torch.utils.data.Dataset):
    """
    A basic dataset that loads, pads, collates, and returns video and audio tensors.

    Returns:

    .. code-block::

        {
            'video': (B x NumFrames x Color x Height x Width) uint8 tensor
            'video_lens': (B, ) int32 tensor
            'audio': (B x NumChannels x NumSamples) float32 tensor
            'audio_lens': (B, ) int32 tensor
            'cuts': CutSet of length B
        }
    """

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        video, video_lens, audio, audio_lens, cuts = collate_video(
            cuts, fault_tolerant=True
        )
        return {
            "cuts": cuts,
            "video": video,
            "video_lens": video_lens,
            "audio": audio,
            "audio_lens": audio_lens,
        }
