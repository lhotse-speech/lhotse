from typing import Optional

import numpy as np

from lhotse.features.base import FeatureExtractor
from lhotse.utils import Decibels, NonPositiveEnergyError, Seconds, compute_num_frames


class FeatureMixer:
    """
    Utility class to mix multiple feature matrices into a single one.
    It should be instantiated separately for each mixing session (i.e. each ``MixedCut``
    will create a separate ``FeatureMixer`` to mix its tracks).
    It is initialized with a numpy array of features (typically float32)
    that represents the "reference" signal for the mix.
    Other signals can be mixed to it with different time offsets and SNRs using the
    ``add_to_mix`` method.
    The time offset is relative to the start of the reference signal
    (only positive values are supported).
    The SNR is relative to the energy of the signal used to initialize the ``FeatureMixer``.

    It relies on the ``FeatureExtractor`` to have defined ``mix`` and ``compute_energy`` methods,
    so that the ``FeatureMixer`` knows how to scale and add two feature matrices together.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        base_feats: np.ndarray,
        frame_shift: Seconds,
        padding_value: float = -1000.0,
        reference_energy: Optional[float] = None,
    ):
        """
        FeatureMixer's constructor.

        :param feature_extractor: The ``FeatureExtractor`` instance that specifies how to mix the features.
        :param base_feats: The features used to initialize the ``FeatureMixer`` are a point of reference
            in terms of energy and offset for all features mixed into them.
        :param frame_shift: Required to correctly compute offset and padding during the mix.
        :param padding_value: The value used to pad the shorter features during the mix.
            This value is adequate only for log space features. For non-log space features,
            e.g. energies, use either 0 or a small positive value like 1e-5.
        :param reference_energy: Optionally pass a reference energy value to compute SNRs against.
            This might be required when ``base_feats`` correspond to padding energies.
        """
        self.feature_extractor = feature_extractor
        self.tracks = [base_feats]
        self.gains = []
        self.frame_shift = frame_shift
        self.padding_value = padding_value
        self.dtype = self.tracks[0].dtype

        # Keep a pre-computed energy value of the features that we initialize the Mixer with;
        # it is required to compute gain ratios that satisfy SNR during the mix.
        if reference_energy is None:
            self.reference_energy = feature_extractor.compute_energy(base_feats)
        else:
            self.reference_energy = reference_energy
        assert (
            self.reference_energy > 0.0
        ), f"To perform mix, energy must be non-zero and non-negative (got {self.reference_energy})"

    @property
    def num_features(self):
        return self.tracks[0].shape[1]

    @property
    def unmixed_feats(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (num_tracks, num_frames, num_features), where each track's
        feature matrix is padded and scaled adequately to the offsets and SNR used in ``add_to_mix`` call.
        """
        return np.stack(self.tracks)

    @property
    def mixed_feats(self) -> np.ndarray:
        """
        Return a numpy ndarray with the shape (num_frames, num_features) - a mono mixed feature matrix
        of the tracks supplied with ``add_to_mix`` calls.
        """
        result = self.tracks[0]
        for feats_to_add, gain in zip(self.tracks[1:], self.gains):
            result = self.feature_extractor.mix(
                features_a=result, features_b=feats_to_add, energy_scaling_factor_b=gain
            )
        return result

    def add_to_mix(
        self,
        feats: np.ndarray,
        sampling_rate: int,
        snr: Optional[Decibels] = None,
        offset: Seconds = 0.0,
    ):
        """
        Add feature matrix of a new track into the mix.
        :param feats: A 2D feature matrix to be mixed in.
        :param sampling_rate: The sampling rate of ``feats``
        :param snr: Signal-to-noise ratio, assuming ``feats`` represents noise (positive SNR - lower ``feats`` energy,
        negative SNR - higher ``feats`` energy)
        :param offset: How many seconds to shift ``feats`` in time. For mixing, the signal will be padded before
        the start with low energy values.
        """
        if len(feats) == 0:
            return  # do nothing for empty arrays

        assert offset >= 0.0, "Negative offset in mixing is not supported."

        reference_feats = self.tracks[0]
        num_frames_offset = compute_num_frames(
            duration=offset, frame_shift=self.frame_shift, sampling_rate=sampling_rate
        )
        current_num_frames = reference_feats.shape[0]
        incoming_num_frames = feats.shape[0] + num_frames_offset
        mix_num_frames = max(current_num_frames, incoming_num_frames)

        feats_to_add = feats

        # When the existing frames are less than what we anticipate after the mix,
        # we need to pad after the end of the existing features mixed so far.
        if current_num_frames < mix_num_frames:
            for idx in range(len(self.tracks)):
                padded_track = np.vstack(
                    [
                        self.tracks[idx],
                        self.padding_value
                        * np.ones(
                            (mix_num_frames - current_num_frames, self.num_features),
                            dtype=self.dtype,
                        ),
                    ]
                )
                self.tracks[idx] = padded_track

        # When there is an offset, we need to pad before the start of the features we're adding.
        if offset > 0:
            feats_to_add = np.vstack(
                [
                    self.padding_value
                    * np.ones((num_frames_offset, self.num_features), dtype=self.dtype),
                    feats_to_add,
                ]
            )

        # When the features we're mixing in are shorter that the anticipated mix length,
        # we need to pad after their end.
        # Note: we're doing that inefficiently, as we potentially re-allocate numpy arrays twice,
        # during this padding and the offset padding before. If that's a bottleneck, we'll optimize.
        if incoming_num_frames < mix_num_frames:
            feats_to_add = np.vstack(
                [
                    feats_to_add,
                    self.padding_value
                    * np.ones(
                        (mix_num_frames - incoming_num_frames, self.num_features),
                        dtype=self.dtype,
                    ),
                ]
            )

        # When SNR is requested, find what gain is needed to satisfy the SNR
        gain = 1.0
        if snr is not None:
            # Compute the added signal energy before it was padded
            added_feats_energy = self.feature_extractor.compute_energy(feats)
            if added_feats_energy <= 0.0:
                raise NonPositiveEnergyError(
                    f"To perform mix, energy must be non-zero and non-negative (got {added_feats_energy}). "
                )
            target_energy = self.reference_energy * (10.0 ** (-snr / 10))
            gain = target_energy / added_feats_energy

        self.tracks.append(feats_to_add)
        self.gains.append(gain)
