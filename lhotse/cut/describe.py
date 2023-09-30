from collections import Counter, defaultdict
from copy import deepcopy
from math import ceil
from typing import List, Optional, Tuple

import numpy as np

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.utils import Seconds, TimeSpan, ifnone, is_module_available


class CutSetStatistics:
    """
    Low-level utility for creating an overview human-readable description for a CutSet.
    It powers the :meth:`~lhotse.cut.set.CutSet.describe` utility. Unlike `describe`,
    it allows to gather stats for multiple cut sets in parallel and combine and display them later.

    Example workflow (typically, the loop in the second line would be parallelized)::

        >>> cut_sets = [CutSet(...), CutSet(...)]
        >>> stats = [CutSetStatistics().accumulate(cuts) for cuts in cut_sets]
        >>> total_stats = stats[0].combine(*stats[1:])
        >>> total_stats.describe()
    """

    def __init__(self, full: bool = False):
        if not is_module_available("tabulate"):
            raise ValueError(
                "Since Lhotse v1.11, this function requires the `tabulate` package to be "
                "installed. Please run 'pip install tabulate' to continue."
            )

        self.full = full
        self.counters = defaultdict(int)
        self.cut_custom, self.sup_custom = Counter(), Counter()
        self.cut_durations = []
        self.speaking_time_durations, self.speech_durations = [], []

        if self.full:
            self.durations_by_num_speakers = defaultdict(list)
            self.single_durations, self.overlapped_durations = [], []

    def combine(self, *other: "CutSetStatistics") -> "CutSetStatistics":
        """Combinemultiple statistics into a new statistics object (does not modify self)."""
        lhs = deepcopy(self)
        for rhs in other:

            assert (
                lhs.full == rhs.full
            ), "Cannot combine statistics gathered with full=True and full=False."

            # Update Dict[str, Number] types
            for attr in ("counters", "cut_custom", "sup_custom"):
                for k in getattr(lhs, attr):
                    getattr(lhs, attr)[k] += getattr(rhs, attr)[k]

            # Update List[Any] types
            for attr in (
                "cut_durations",
                "speaking_time_durations",
                "speech_durations",
            ) + (("single_durations", "overlapped_durations") if lhs.full else ()):
                getattr(lhs, attr).extend(getattr(rhs, attr))

            if lhs.full:
                for k in lhs.durations_by_num_speakers:
                    lhs.durations_by_num_speakers[k].extend(
                        rhs.durations_by_num_speakers[k]
                    )

        return lhs

    def accumulate(self, cuts: CutSet) -> "CutSetStatistics":
        """Gather statistics to display later for a cut set."""

        def total_duration_(segments: List[TimeSpan]) -> float:
            return sum(segment.duration for segment in segments)

        for c in cuts:
            self.cut_durations.append(c.duration)
            if hasattr(c, "custom"):
                for key in ifnone(c.custom, ()):
                    self.cut_custom[key] += 1
            self.counters["recordings"] += int(c.has_recording)
            self.counters["features"] += int(c.has_features)

            # Total speaking time duration is computed by summing the duration of all
            # supervisions in the cut.
            for s in c.trimmed_supervisions:
                self.speaking_time_durations.append(s.duration)
                self.counters["supervisions"] += 1
                for key in ifnone(s.custom, ()):
                    self.sup_custom[key] += 1

            # Total speech duration is the sum of intervals where 1 or more speakers are
            # active.
            self.speech_durations.append(
                total_duration_(find_segments_with_speaker_count(c, min_speakers=1))
            )

            if self.full:
                # Duration of single-speaker segments
                self.single_durations.append(
                    total_duration_(
                        find_segments_with_speaker_count(
                            c, min_speakers=1, max_speakers=1
                        )
                    )
                )
                # Duration of overlapped segments
                self.overlapped_durations.append(
                    total_duration_(
                        find_segments_with_speaker_count(
                            c, min_speakers=2, max_speakers=None
                        )
                    )
                )
                # Durations by number of speakers (we assume that overlaps can happen between
                # at most 4 speakers. This is a reasonable assumption for most datasets.)
                self.durations_by_num_speakers[1].append(self.single_durations[-1])
                for num_spk in range(2, 5):
                    self.durations_by_num_speakers[num_spk].append(
                        total_duration_(
                            find_segments_with_speaker_count(
                                c, min_speakers=num_spk, max_speakers=num_spk
                            )
                        )
                    )

        return self

    def describe(self) -> None:
        """Display accumulated statistics in the CLI."""
        from tabulate import tabulate

        def convert_(seconds: Seconds) -> Tuple[int, int, int]:
            hours, seconds = divmod(seconds, 3600)
            minutes, seconds = divmod(seconds, 60)
            return int(hours), int(minutes), ceil(seconds)

        def time_as_str_(seconds: Seconds) -> str:
            h, m, s = convert_(seconds)
            return f"{h:02d}:{m:02d}:{s:02d}"

        cut_durations = self.cut_durations

        total_sum = np.array(cut_durations).sum()

        cut_stats = []
        cut_stats.append(["Cuts count:", len(cut_durations)])
        cut_stats.append(["Total duration (hh:mm:ss)", time_as_str_(total_sum)])
        cut_stats.append(["mean", f"{np.mean(cut_durations):.1f}"])
        cut_stats.append(["std", f"{np.std(cut_durations):.1f}"])
        cut_stats.append(["min", f"{np.min(cut_durations):.1f}"])
        cut_stats.append(["25%", f"{np.percentile(cut_durations, 25):.1f}"])
        cut_stats.append(["50%", f"{np.median(cut_durations):.1f}"])
        cut_stats.append(["75%", f"{np.percentile(cut_durations, 75):.1f}"])
        cut_stats.append(["99%", f"{np.percentile(cut_durations, 99):.1f}"])
        cut_stats.append(["99.5%", f"{np.percentile(cut_durations, 99.5):.1f}"])
        cut_stats.append(["99.9%", f"{np.percentile(cut_durations, 99.9):.1f}"])
        cut_stats.append(["max", f"{np.max(cut_durations):.1f}"])

        for key, val in self.counters.items():
            cut_stats.append([f"{key.title()} available:", val])

        print("Cut statistics:")
        print(tabulate(cut_stats, tablefmt="fancy_grid"))

        if self.cut_custom:
            print("CUT custom fields:")
            for key, val in self.cut_custom.most_common():
                print(f"- {key} (in {val} cuts)")

        if self.sup_custom:
            print("SUPERVISION custom fields:")
            for key, val in self.sup_custom.most_common():
                cut_stats.append(f"- {key} (in {val} cuts)")

        total_speech = np.array(self.speech_durations).sum()
        total_speaking_time = np.array(self.speaking_time_durations).sum()
        total_silence = total_sum - total_speech
        speech_stats = []
        speech_stats.append(
            [
                "Total speech duration",
                time_as_str_(total_speech),
                f"{total_speech / total_sum:.2%} of recording",
            ]
        )
        speech_stats.append(
            [
                "Total speaking time duration",
                time_as_str_(total_speaking_time),
                f"{total_speaking_time / total_sum:.2%} of recording",
            ]
        )
        speech_stats.append(
            [
                "Total silence duration",
                time_as_str_(total_silence),
                f"{total_silence / total_sum:.2%} of recording",
            ]
        )
        if self.full:
            total_single = np.array(self.single_durations).sum()
            total_overlap = np.array(self.overlapped_durations).sum()
            speech_stats.append(
                [
                    "Single-speaker duration",
                    time_as_str_(total_single),
                    f"{total_single / total_sum:.2%} ({total_single / total_speech:.2%} of speech)",
                ]
            )
            speech_stats.append(
                [
                    "Overlapped speech duration",
                    time_as_str_(total_overlap),
                    f"{total_overlap / total_sum:.2%} ({total_overlap / total_speech:.2%} of speech)",
                ]
            )
        print("Speech duration statistics:")
        print(tabulate(speech_stats, tablefmt="fancy_grid"))

        if not self.full:
            return

        # Additional statistics for full report
        speaker_stats = [
            [
                "Number of speakers",
                "Duration (hh:mm:ss)",
                "Speaking time (hh:mm:ss)",
                "% of speech",
                "% of speaking time",
            ]
        ]
        for num_spk, durations in self.durations_by_num_speakers.items():
            speaker_sum = np.array(durations).sum()
            speaking_time = num_spk * speaker_sum
            speaker_stats.append(
                [
                    num_spk,
                    time_as_str_(speaker_sum),
                    time_as_str_(speaking_time),
                    f"{speaker_sum / total_speech:.2%}",
                    f"{speaking_time / total_speaking_time:.2%}",
                ]
            )

        speaker_stats.append(
            [
                "Total",
                time_as_str_(total_speech),
                time_as_str_(total_speaking_time),
                "100.00%",
                "100.00%",
            ]
        )

        print("Speech duration statistics by number of speakers:")
        print(tabulate(speaker_stats, headers="firstrow", tablefmt="fancy_grid"))


def find_segments_with_speaker_count(
    cut: Cut, min_speakers: int = 0, max_speakers: Optional[int] = None
) -> List[TimeSpan]:
    """
    Given a Cut, find a list of intervals that contain the specified number of speakers.

    :param cuts: the Cut to search.
    :param min_speakers: the minimum number of speakers.
    :param max_speakers: the maximum number of speakers.
    :return: a list of TimeSpans.
    """
    if max_speakers is None:
        max_speakers = float("inf")

    assert (
        min_speakers >= 0 and min_speakers <= max_speakers
    ), f"min_speakers={min_speakers} and max_speakers={max_speakers} are not valid."

    # First take care of trivial cases.
    if min_speakers == 0 and max_speakers == float("inf"):
        return [TimeSpan(0, cut.duration)]
    if len(cut.supervisions) == 0:
        return [] if min_speakers > 0 else [TimeSpan(0, cut.duration)]

    # We collect all the timestamps of the supervisions in the cut. Each timestamp is
    # a tuple of (time, is_speaker_start).
    timestamps = []
    # Add timestamp for cut start
    timestamps.append((0.0, None))
    for segment in cut.supervisions:
        timestamps.append((segment.start, True))
        timestamps.append((segment.end, False))
    # Add timestamp for cut end
    timestamps.append((cut.duration, None))

    # Sort the timestamps. We need the following priority order:
    # 1. Time mark of the timestamp: lower time mark comes first.
    # 2. For timestamps with the same time mark, None < False < True.
    timestamps.sort(key=lambda x: (x[0], x[1] is not None, x[1] is True))

    # We remove the timestamps that are not relevant for the search. The desired range
    # is given by the range of the cut start and end timestamps.
    cut_start_idx, cut_end_idx = [i for i, t in enumerate(timestamps) if t[1] is None]
    timestamps = timestamps[cut_start_idx : cut_end_idx + 1]

    # Now we iterate over the timestamps and count the number of speakers in any
    # given time interval. If the number of speakers is in the desired range,
    # we keep the interval.
    num_speakers = 0
    seg_start = 0.0
    intervals = []
    for timestamp, is_start in timestamps[1:]:
        if num_speakers >= min_speakers and num_speakers <= max_speakers:
            intervals.append((seg_start, timestamp))
        if is_start is not None:
            num_speakers += 1 if is_start else -1
        seg_start = timestamp

    # Merge consecutive intervals and remove empty intervals.
    merged_intervals = []
    for start, end in intervals:
        if start == end:
            continue
        if merged_intervals and merged_intervals[-1][1] == start:
            merged_intervals[-1] = (merged_intervals[-1][0], end)
        else:
            merged_intervals.append((start, end))

    return [TimeSpan(start, end) for start, end in merged_intervals]
