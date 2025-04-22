import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import lhotse.augmentation.lowpass
from lhotse import CutSet


@dataclass
class Lowpass:
    """
    Applies a low-pass filter to each Cut in a CutSet.

    The filter is applied with a probability of ``p``. When applied, the filter
    randomly selects a cutoff frequency from the list of provided frequencies,
    with optional weights controlling the selection.

    The filter type is randomly selected from the list of provided filter types,
    with optional weights controlling the selection.

    Filter order, stopband attenuation, and passband ripple can be either constant, or sampled uniformly from provided intervals.

    :param p: The probability of applying the low-pass filter (default: 0.5).
    :param frequencies: A list of cutoff frequencies.
    :param frequency_weights: Optional weights for each frequency (default: equal weights).
    :param filter_types: List of filter types to use. One or more of: "butter", "cheby1", "cheby2", "ellip", "bessel" (default: ["butter"]).
    :param filter_type_weights: Optional weights for each filter type (default: equal weights).
    :param order: Filter order. Can be a single value or an interval. If an interval is provided, the value is sampled uniformly.
    :param stopband_attenuation_db: Stopband attenuation in dB. Used for Chebyshev II and Elliptical filters. Can be a single value or an interval. If an interval is provided, the value is sampled uniformly.
    :param ripple_db: Passband ripple in dB. Used for Chebyshev I and Elliptical filters. Can be a single value or an interval. If an interval is provided, the value is sampled uniformly.
    :param randgen: An optional random number generator (default: a new instance).
    """

    p: float = 0.5
    frequencies: List[float] = (3500,)
    frequency_weights: Optional[List[float]] = None
    filter_types: List[
        lhotse.augmentation.lowpass.Filter
    ] = lhotse.augmentation.lowpass.Lowpass.supported_filters
    filter_type_weights: Optional[List[float]] = None
    order: Union[int, Tuple[int, int]] = 4
    stopband_attenuation_db: Union[float, Tuple[float, float]] = 40.0
    ripple_db: Union[float, Tuple[float, float]] = 0.1
    randgen: random.Random = None

    def __post_init__(self) -> None:
        assert self.frequencies, "Cutoff frequencies must be provided, at least one"

        if self.frequency_weights:
            assert len(self.frequency_weights) == len(self.frequencies)
        else:
            # all frequencies have equal weights by default
            self.frequency_weights = [1.0 for _ in self.frequencies]

        assert self.filter_types, "Filter types must be provided, at least one"

        if self.filter_type_weights:
            assert len(self.filter_type_weights) == len(self.filter_types)
            assert all(
                w >= 0 for w in self.filter_type_weights
            ), "Filter type weights must be non-negative"
        else:
            # all filter types have equal weights by default
            self.filter_type_weights = [1.0 for _ in self.filter_types]

        if isinstance(self.stopband_attenuation_db, tuple):
            min_db, max_db = self.stopband_attenuation_db
            assert (
                min_db <= max_db
            ), f"Minimum stopband attenuation ({min_db}dB) must be less than or equal to maximum ({max_db}dB)"
            assert (
                min_db > 0
            ), f"Minimum stopband attenuation must be positive, got {min_db}dB"

        if not isinstance(self.stopband_attenuation_db, (int, float, tuple, list)):
            raise TypeError(
                f"stopband_attenuation_db must be a number or a two-value tuple/list, got {type(self.stopband_attenuation_db)}"
            )
        if isinstance(self.stopband_attenuation_db, (tuple, list)):
            if len(self.stopband_attenuation_db) != 2:
                raise ValueError(
                    f"stopband_attenuation_db tuple/list must have exactly 2 values, got {len(self.stopband_attenuation_db)}"
                )

        if not isinstance(self.ripple_db, (int, float, tuple, list)):
            raise TypeError(
                f"ripple_db must be a number or a two-value tuple/list, got {type(self.ripple_db)}"
            )
        if isinstance(self.ripple_db, (tuple, list)):
            if len(self.ripple_db) != 2:
                raise ValueError(
                    f"ripple_db tuple/list must have exactly 2 values, got {len(self.ripple_db)}"
                )
        if isinstance(self.ripple_db, tuple):
            min_db, max_db = self.ripple_db
            assert (
                min_db <= max_db
            ), f"Minimum ripple ({min_db}dB) must be less than or equal to maximum ({max_db}dB)"
            assert min_db >= 0, f"Minimum ripple must be non-negative, got {min_db}dB"
            assert max_db >= 0, f"Maximum ripple must be non-negative, got {max_db}dB"
        else:
            assert (
                self.ripple_db > 0
            ), f"Ripple must be positive, got {self.ripple_db}dB"

        if not isinstance(self.order, (int, tuple, list)):
            raise TypeError(
                f"Order must be an integer or a two-value tuple/list, got {type(self.order)}"
            )
        if isinstance(self.order, (tuple, list)):
            if len(self.order) != 2:
                raise ValueError(
                    f"Order tuple/list must have exactly 2 values, got {len(self.order)}"
                )

        if isinstance(self.order, (tuple, list)):
            min_order, max_order = self.order
            assert isinstance(min_order, int) and isinstance(
                max_order, int
            ), "Order values must be integers"
            assert (
                min_order <= max_order
            ), f"Minimum order ({min_order}) must be less than or equal to maximum ({max_order})"
            assert min_order > 0, f"Minimum order must be positive, got {min_order}"
        else:
            assert self.order > 0, f"Order must be positive, got {self.order}"
            assert isinstance(self.order, int), "Order must be an integer"

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.randgen is None:
            self.randgen = random.Random()

        lowpassed_cuts = []
        for cut in cuts:
            if self.randgen.random() <= self.p:
                frequency, *_ = self.randgen.choices(
                    self.frequencies, weights=self.frequency_weights
                )

                filter_type, *_ = self.randgen.choices(
                    self.filter_types, weights=self.filter_type_weights
                )

                if isinstance(self.stopband_attenuation_db, tuple):
                    min_db, max_db = self.stopband_attenuation_db
                    stopband_attenuation = self.randgen.uniform(min_db, max_db)
                else:
                    stopband_attenuation = self.stopband_attenuation_db

                if isinstance(self.ripple_db, tuple):
                    min_db, max_db = self.ripple_db
                    ripple = self.randgen.uniform(min_db, max_db)
                else:
                    ripple = self.ripple_db

                if isinstance(self.order, tuple):
                    min_order, max_order = self.order
                    order = self.randgen.randint(min_order, max_order)
                else:
                    order = self.order

                new_cut = cut.lowpass(
                    frequency=frequency,
                    stopband_attenuation_db=stopband_attenuation,
                    ripple_db=ripple,
                    order=order,
                    filter_type=filter_type,
                )
                new_cut.id = f"{cut.id}_lowpassed{frequency:.0f}_{filter_type}_{order}"
                lowpassed_cuts.append(new_cut)
            else:
                lowpassed_cuts.append(cut)

        return CutSet(lowpassed_cuts)
