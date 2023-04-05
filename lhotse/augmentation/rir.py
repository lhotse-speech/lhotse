from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from lhotse.augmentation.transform import AudioTransform
from lhotse.augmentation.utils import FastRandomRIRGenerator, convolve1d
from lhotse.utils import Seconds


@dataclass
class ReverbWithImpulseResponse(AudioTransform):
    """
    Reverberation effect by convolving with a room impulse response.
    This code is based on Kaldi's wav-reverberate utility:
    https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/wav-reverberate.cc
    If no ``rir_recording`` is provided, we will generate an impulse response using a fast random
    generator (https://arxiv.org/abs/2208.04101).
    The impulse response can possibly be multi-channel, in which case multi-channel reverberated
    audio can be obtained by appropriately setting `rir_channels`. For example, `rir_channels=[0,1]`
    will convolve using the first two channels of the impulse response, generating a stereo
    reverberated audio.
    Note that we enforce the --shift-output option in Kaldi's wav-reverberate utility,
    which means that the output length will be equal to the input length.
    """

    rir: Optional[dict] = None
    normalize_output: bool = True
    early_only: bool = False
    rir_channels: List[int] = field(default_factory=lambda: [0])
    rir_generator: Optional[Union[dict, Callable]] = None

    RIR_SCALING_FACTOR: float = 0.5**15

    def __post_init__(self):
        if isinstance(self.rir, dict):
            from lhotse import Recording

            # Pass a shallow copy of the RIR dict since `from_dict()` pops the `sources` key.
            self.rir = Recording.from_dict(self.rir.copy())

        assert (
            self.rir is not None or self.rir_generator is not None
        ), "Either `rir` or `rir_generator` must be provided."

        if self.rir is not None:
            assert all(
                c < self.rir.num_channels for c in self.rir_channels
            ), "Invalid channel index in `rir_channels`"

        if self.rir_generator is not None and isinstance(self.rir_generator, dict):
            self.rir_generator = FastRandomRIRGenerator(**self.rir_generator)

    def __call__(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ) -> np.ndarray:
        """
        :param samples: The audio samples to reverberate.
        :param sampling_rate: The sampling rate of the audio samples.
        """
        sampling_rate = int(sampling_rate)  # paranoia mode

        D_in, N_in = samples.shape
        input_is_mono = D_in == 1

        # The following cases are possible:
        # Case 1: input is mono, rir is mono -> mono output
        #   We will generate a random mono rir if not provided explicitly.
        # Case 2: input is mono, rir is multi-channel -> multi-channel output
        #   This requires a user-provided rir, since we cannot simulate array microphone.
        # Case 3: input is multi-channel, rir is mono -> multi-channel output
        #   This does not make much sense, but we will apply the same rir to all channels.
        # 4. input is multi-channel, rir is multi-channel -> multi-channel output
        #   This also requires a user-provided rir. Also, the number of channels in the rir
        #   must match the number of channels in the input.

        # Let us make some assertions based on the above.
        if input_is_mono:
            assert (
                self.rir is not None or len(self.rir_channels) == 1
            ), "For mono input, either provide an RIR explicitly or set rir_channels to [0]."
        else:
            assert len(self.rir_channels) == 1 or len(self.rir_channels) == D_in, (
                "For multi-channel input, we only support mono RIR or RIR with the same number "
                "of channels as the input."
            )

        # Generate a random RIR if not provided.
        if self.rir is None:
            rir_ = self.rir_generator(nsource=1)
        else:
            rir_ = (
                self.rir.load_audio(channels=self.rir_channels)
                if not self.early_only
                else self.rir.load_audio(channels=self.rir_channels, duration=0.05)
            )

        D_rir, N_rir = rir_.shape
        N_out = N_in  # Enforce shift output
        # output is multi-channel if either input or rir is multi-channel
        D_out = D_rir if input_is_mono else D_in

        # if RIR is mono, repeat it to match the number of channels in the input
        rir_ = rir_.repeat(D_out, axis=0) if D_rir == 1 else rir_

        # Initialize output matrix with the specified input channel.
        augmented = np.zeros((D_out, N_out), dtype=samples.dtype)

        for d in range(D_out):
            d_in = 0 if input_is_mono else d
            augmented[d, :N_in] = samples[d_in]
            power_before_reverb = np.sum(np.abs(samples[d_in]) ** 2) / N_in
            rir_d = rir_[d, :] * self.RIR_SCALING_FACTOR

            # Convolve the signal with impulse response.
            aug_d = convolve1d(
                torch.from_numpy(samples[d_in]), torch.from_numpy(rir_d)
            ).numpy()
            shift_index = np.argmax(rir_d)
            augmented[d, :] = aug_d[shift_index : shift_index + N_out]

            if self.normalize_output:
                power_after_reverb = np.sum(np.abs(augmented[d, :]) ** 2) / N_out
                if power_after_reverb > 0:
                    augmented[d, :] *= np.sqrt(power_before_reverb / power_after_reverb)

        return augmented

    def reverse_timestamps(
        self,
        offset: Seconds,
        duration: Optional[Seconds],
        sampling_rate: Optional[int],  # Not used, made for compatibility purposes
    ) -> Tuple[Seconds, Optional[Seconds]]:
        """
        This method just returns the original offset and duration since we have
        implemented output shifting which preserves these properties.
        """

        return offset, duration
