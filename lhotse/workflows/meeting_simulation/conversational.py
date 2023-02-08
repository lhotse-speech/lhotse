import logging
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, List, Optional, Union

import numpy as np
from tqdm import tqdm

from lhotse import RecordingSet, SupervisionSet
from lhotse.cut import CutSet, MixedCut, MixTrack
from lhotse.cut.set import mix
from lhotse.parallel import parallel_map
from lhotse.utils import uuid4
from lhotse.workflows.meeting_simulation.base import (
    MAX_TASKS_WAITING,
    BaseMeetingSimulator,
    MeetingSampler,
    reverberate_cuts,
)


class ConversationalMeetingSimulator(BaseMeetingSimulator):
    """
    This simulator uses the method described in https://arxiv.org/abs/2204.00890 and
    implemented in https://github.com/BUTSpeechFIT/EEND_dataprep. Note that a similar
    method of meeting simulation is also implemented in
    https://github.com/jsalt2020-asrdiar/jsalt2020_simulate.

    The basic idea is to sample the silence and overlap durations collectively for all
    speakers so that we get similar speech/silence/overlap characteristics as the
    original distribution. In general, this method produces more realistic overlap
    durations than the `SpeakerIndependentMeetingSimulator`. This is done by learning
    the histogram of 3 distributions: same speaker pause, different speaker pause, and
    different speaker overlap. In this implementation, we learn the histograms from
    provided data, otherwise we use the initialization values as shift of a Gamma
    distribution with scale 1.0.
    """

    def __init__(
        self,
        same_spk_pause: float = 1.0,
        diff_spk_pause: float = 1.0,
        diff_spk_overlap: float = 2.0,
        prob_diff_spk_overlap: float = 0.5,
    ):
        """
        :param same_spk_pause: the mean pause duration between utterances of the same
            speaker. [Default: 2.0]
        :param diff_spk_pause: the mean pause duration between utterances of different
            speakers. [Default: 3.0]
        :param diff_spk_overlap: the mean overlap duration between utterances of
            different speakers. [Default: 2.0]
        :param prob_diff_spk_overlap: the probability of overlap between utterances of
            different speakers. [Default: 0.5]
        """
        super().__init__()
        for duration in [same_spk_pause, diff_spk_pause, diff_spk_overlap]:
            assert duration is None or duration > 0, "Durations must be > 0."

        self.same_spk_pause = same_spk_pause
        self.diff_spk_pause = diff_spk_pause
        self.diff_spk_overlap = diff_spk_overlap
        self.prob_diff_spk_overlap = prob_diff_spk_overlap

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            f"(same_spk_pause={self.same_spk_pause:.2f}, "
            f"diff_spk_pause={self.diff_spk_pause:.2f}, "
            f"diff_spk_overlap={self.diff_spk_overlap:.2f}), "
            f"prob_diff_spk_overlap={self.prob_diff_spk_overlap:.2f})"
        )

    def _init_defaults(self):
        from scipy.stats import bernoulli, gamma

        self.same_spk_pause_dist = gamma(a=1.0, scale=1.0, loc=self.same_spk_pause)
        self.diff_spk_pause_dist = gamma(a=1.0, scale=1.0, loc=self.diff_spk_pause)
        self.diff_spk_overlap_dist = gamma(a=1.0, scale=1.0, loc=self.diff_spk_overlap)

    def _compute_histogram_dist(self, values: np.ndarray) -> Any:
        """
        Compute the histogram of the given values and return the bin edges and the
        corresponding probabilities.
        """
        from scipy.stats import rv_histogram

        hist, bin_edges = np.histogram(values, bins=100, density=True)
        return rv_histogram((hist, bin_edges))

    def fit(self, meetings: Optional[SupervisionSet] = None) -> None:
        """
        Learn the distribution of the meeting parameters from a given dataset.
        :param meetings: a SupervisionSet containing the meetings to be used for
        """
        if meetings is None:
            logging.info(f"No meetings provided, using default parameters.")
            self._init_defaults()
            return

        assert isinstance(
            meetings, SupervisionSet
        ), "The meetings must be provided as a SupervisionSet."

        from cytoolz.itertoolz import groupby

        # Generate same speaker pause distribution.
        same_spk_pause_values = []

        speaker_segments = groupby(
            lambda s: (s.recording_id, s.speaker),
            sorted(meetings, key=lambda s: (s.recording_id, s.speaker)),
        )

        for segments in speaker_segments.values():
            segments = sorted(segments, key=lambda s: s.start)
            for i in range(1, len(segments)):
                same_spk_pause_values.append(
                    max(0, segments[i].start - segments[i - 1].end)
                )

        self.same_spk_pause_dist = self._compute_histogram_dist(
            np.array(same_spk_pause_values)
        )

        # Generate different speaker pause and overlap distributions.
        diff_spk_pause_values = []
        diff_spk_overlap_values = []

        recording_segments = groupby(
            lambda s: s.recording_id,
            sorted(meetings, key=lambda s: s.recording_id),
        )
        for segments in recording_segments.values():
            segments = sorted(segments, key=lambda s: s.start)
            for i in range(1, len(segments)):
                if segments[i].speaker == segments[i - 1].speaker:
                    continue
                if segments[i].start > segments[i - 1].end:
                    diff_spk_pause_values.append(
                        segments[i].start - segments[i - 1].end
                    )
                else:
                    diff_spk_overlap_values.append(
                        segments[i - 1].end - segments[i].start
                    )

        self.diff_spk_pause_dist = self._compute_histogram_dist(
            np.array(diff_spk_pause_values)
        )
        self.diff_spk_overlap_dist = self._compute_histogram_dist(
            np.array(diff_spk_overlap_values)
        )
        self.prob_diff_spk_overlap = (
            len(diff_spk_overlap_values)
            / (len(diff_spk_pause_values) + len(diff_spk_overlap_values))
            if (len(diff_spk_pause_values) + len(diff_spk_overlap_values)) > 0
            else 0.5
        )

        # Update the parameters.
        self.same_spk_pause = self.same_spk_pause_dist.mean()
        self.diff_spk_pause = self.diff_spk_pause_dist.mean()
        self.diff_spk_overlap = self.diff_spk_overlap_dist.mean()

        print(f"Learned parameters: {self}")

    def _create_mixture(
        self, utterances: CutSet, allow_3fold_overlap: bool = False
    ) -> MixedCut:
        """
        Create a MixedCut object from a list of MonoCuts (utterances).
        We sample pauses and/or overlaps from the initilized or learned distributions.
        Then, we create a MixedCut where each track represents a different speaker.

        :param utterances: a CutSet containing the utterances to be mixed.
        :param allow_3fold_overlap: if True, allow 3-fold overlaps between speakers.
            [Default: False]
        :return: a MixedCut object.
        """
        # We keep track of the end time of the last utterance for each speaker.
        speakers = utterances.speakers
        last_utt_end = {spkr: 0.0 for spkr in speakers}
        last_utt_end_times = sorted(list(last_utt_end.values()), reverse=True)

        # Generate pause/overlap timings for all utterances.
        N = len(utterances)
        same_spk_pauses = self.same_spk_pause_dist.rvs(size=N)
        diff_spk_pauses = self.diff_spk_pause_dist.rvs(size=N)
        diff_spk_overlaps = self.diff_spk_overlap_dist.rvs(size=N)
        diff_spk_bernoulli = self.bernoulli.rvs(p=self.prob_diff_spk_overlap, size=N)

        utterances = list(utterances.data.values())
        # First sample offsets for each utterance. These are w.r.t. start of the meeting.
        offsets = [0.0]
        cur_offset = utterances[0].duration

        for i in range(1, len(utterances)):
            cur_spk = utterances[i].supervisions[0].speaker
            prev_spk = utterances[i - 1].supervisions[0].speaker
            if cur_spk == prev_spk:
                ot = same_spk_pauses[i]
            else:
                if diff_spk_bernoulli[i] == 0:
                    # No overlap between speakers.
                    ot = diff_spk_pauses[i]
                else:
                    # Overlap between speakers.
                    ot = diff_spk_overlaps[i]
                    if len(last_utt_end_times) > 1 and not allow_3fold_overlap:
                        # second term for ensuring same speaker's utterances do not overlap.
                        # third term for ensuring the maximum number of overlaps is two.
                        ot = min(
                            ot,
                            cur_offset - last_utt_end[cur_spk],
                            cur_offset - last_utt_end_times[1],
                        )
                    else:
                        ot = min(ot, cur_offset - last_utt_end[cur_spk])
                    ot = -ot

            cur_offset += ot
            offsets.append(cur_offset)
            cur_offset += utterances[i].duration

            # Update last_utt_end and last_utt_end_times.
            last_utt_end[cur_spk] = cur_offset
            last_utt_end_times = sorted(list(last_utt_end.values()), reverse=True)
            cur_offset = last_utt_end_times[0]

        # Group utterances (and offsets) by speaker.
        spk_tracks = defaultdict(list)
        for utt, offset in zip(utterances, offsets):
            spk_tracks[utt.supervisions[0].speaker].append((utt, offset))

        # Now create speaker-wise tracks by mixing their utterances with silence padding.
        tracks = []
        for spk, spk_utts in spk_tracks.items():
            track, start = spk_utts[0]
            for utt, offset in spk_utts[1:]:
                track = mix(track, utt, offset=offset - start, allow_padding=True)
            track = MixTrack(cut=track, type=type(track), offset=start)
            tracks.append(track)

        return MixedCut(id=str(uuid4()), tracks=tracks)

    def simulate(
        self,
        cuts: CutSet,
        num_meetings: Optional[int] = None,
        num_repeats: Optional[int] = None,
        num_speakers_per_meeting: Union[int, List[int]] = 2,
        speaker_count_probs: Optional[List[float]] = None,
        max_duration_per_speaker: Optional[float] = 20.0,
        max_utterances_per_speaker: Optional[int] = 5,
        allow_3fold_overlap: bool = False,
        seed: int = 0,
        num_jobs: int = 1,
    ) -> CutSet:
        """
        Simulate the desired number of multi-speaker meetings.
        :param cuts: CutSet containing the MonoCut objects to be used for simulation.
        :param num_meetings: the number of meetings to simulate.
        :param num_repeats: the number of times to repeat the provided cuts. This means that
            the number of simulated meetings depends on how many cuts are available.
        :param num_speakers_per_meeting: the number of speakers per meeting. If a list is
            provided, the number of speakers per meeting is sampled from this list.
            [Default: 2]
        :param speaker_count_probs: the probability of each number of speakers per meeting.
            [Default: None]
        :param max_duration_per_speaker: the maximum duration of each speaker in a meeting.
            [Default: 20.0]
        :param max_utterances_per_speaker: the maximum number of utterances per speaker in a
            meeting. [Default: 5]
        :param allow_3fold_overlap: if True, allow 3-fold overlaps between speakers.
            [Default: False]
        :param seed: the random seed to be used for simulation. [Default: 0]
        :param num_jobs: the number of jobs to use for simulation. Use more jobs to speed up
            simulation when you have large number of source utterances. [Default: 1]
        """
        from scipy.stats import bernoulli

        if num_meetings is None and num_repeats is None:
            raise ValueError("Either num_meetings or num_repeats must be provided.")

        if num_meetings is not None:
            num_repeats = None

        if isinstance(num_speakers_per_meeting, int):
            num_speakers_per_meeting = [num_speakers_per_meeting]

        if speaker_count_probs is None:
            speaker_count_probs = [1.0 / len(num_speakers_per_meeting)] * len(
                num_speakers_per_meeting
            )

        # Initialize default distributions if not provided.
        if getattr(self, "same_spk_pause_dist", None) is None:
            self._init_defaults()

        # Create random number generators with the given seed.
        self.bernoulli = bernoulli

        # Create cuts sampler
        sampler = MeetingSampler(
            cuts,
            num_repeats=num_repeats,
            num_meetings=num_meetings,
            max_duration_per_speaker=max_duration_per_speaker,
            max_utterances_per_speaker=max_utterances_per_speaker,
            num_speakers_per_meeting=num_speakers_per_meeting,
            speaker_count_probs=speaker_count_probs,
            seed=seed,
        )
        sampler_iter = iter(sampler)

        global _simulate_worker

        def _simulate_worker(utterances):
            return self._create_mixture(
                utterances, allow_3fold_overlap=allow_3fold_overlap
            )

        mixtures = []
        if num_jobs == 1:
            # Don't use multiprocessing if num_jobs == 1.
            for mixture in tqdm(
                map(_simulate_worker, sampler_iter),
                total=num_meetings,
                desc="Simulating meetings",
            ):
                mixtures.append(mixture)
        else:
            for mixture in tqdm(
                parallel_map(
                    _simulate_worker,
                    sampler_iter,
                    num_jobs=num_jobs,
                    queue_size=num_jobs * MAX_TASKS_WAITING,
                ),
                total=num_meetings,
                desc="Simulating meetings",
            ):
                mixtures.append(mixture)

        return CutSet.from_cuts(mixtures)

    def reverberate(self, cuts: CutSet, *rirs: RecordingSet) -> CutSet:
        return reverberate_cuts(cuts, *rirs)
