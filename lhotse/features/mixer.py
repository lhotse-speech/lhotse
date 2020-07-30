from typing import Optional

import numpy as np

from lhotse.features.base import FeatureExtractor
from lhotse.utils import Seconds, Decibels


class FeatureMixer:
    """
    Utility class to mix multiple log-mel energy feature matrices into a single one.
    It pads the signals with low energy values to account for differing lengths and offsets.
    """

    def __init__(
            self,
            feature_extractor: FeatureExtractor,
            base_feats: np.ndarray,
            frame_shift: Seconds,
            log_energy_floor: float = -1000.0
    ):
        """
        :param base_feats: The features used to initialize the FbankMixer are a point of reference
            in terms of energy and offset for all features mixed into them.
        :param frame_shift: Required to correctly compute offset and padding during the mix.
        :param frame_length: Required to correctly compute offset and padding during the mix.
        :param log_energy_floor: The value used to pad the shorter features during the mix.
        """
        self.feature_extractor = feature_extractor
        # The mixing output will be available in self.mixed_feats
        self.mixed_feats = base_feats
        # Keep a pre-computed energy value of the features that we initialize the Mixer with;
        # it is required to compute gain ratios that satisfy SNR during the mix.
        self.frame_shift = frame_shift
        self.reference_energy = feature_extractor.compute_energy(base_feats)
        self.log_energy_floor = log_energy_floor

    @property
    def num_features(self):
        return self.mixed_feats.shape[1]

    def add_to_mix(
            self,
            feats: np.ndarray,
            snr: Optional[Decibels] = None,
            offset: Seconds = 0.0
    ):
        """
        Add feature matrix of a new track into the mix.
        :param feats: A 2-d feature matrix to be mixed in.
        :param snr: Signal-to-noise ratio, assuming `feats` represents noise (positive SNR - lower `feats` energy,
        negative SNR - higher `feats` energy)
        :param offset: How many seconds to shift `feats` in time. For mixing, the signal will be padded before
        the start with low energy values.
        :return:
        """
        assert offset >= 0.0, "Negative offset in mixing is not supported."

        num_frames_offset = round(offset / self.frame_shift)
        current_num_frames = self.mixed_feats.shape[0]
        incoming_num_frames = feats.shape[0] + num_frames_offset
        mix_num_frames = max(current_num_frames, incoming_num_frames)

        existing_feats = self.mixed_feats
        feats_to_add = feats

        # When the existing frames are less than what we anticipate after the mix,
        # we need to pad after the end of the existing features mixed so far.
        if current_num_frames < mix_num_frames:
            existing_feats = np.vstack([
                self.mixed_feats,
                self.log_energy_floor * np.ones((mix_num_frames - current_num_frames, self.num_features))
            ])

        # When there is an offset, we need to pad before the start of the features we're adding.
        if offset > 0:
            feats_to_add = np.vstack([
                self.log_energy_floor * np.ones((num_frames_offset, self.num_features)),
                feats_to_add
            ])

        # When the features we're mixing in are shorter that the anticipated mix length,
        # we need to pad after their end.
        # Note: we're doing that non-efficiently, as it we potentially re-allocate numpy arrays twice,
        # during this padding and the  offset padding before. If that's a bottleneck, we'll optimize.
        if incoming_num_frames < mix_num_frames:
            feats_to_add = np.vstack([
                feats_to_add,
                self.log_energy_floor * np.ones((mix_num_frames - incoming_num_frames, self.num_features))
            ])

        # When SNR is requested, find what gain is needed to satisfy the SNR
        gain = 1.0
        if snr is not None:
            # Compute the added signal energy before it was padded
            added_feats_energy = self.feature_extractor.compute_energy(feats)
            target_energy = self.reference_energy * (10.0 ** (-snr / 10))
            gain = target_energy / added_feats_energy

        self.mixed_feats = self.feature_extractor.mix(
            features_a=existing_feats,
            features_b=feats_to_add,
            gain_b=gain
        )
