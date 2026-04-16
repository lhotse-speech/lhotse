import math

import pytest
from pytest import approx

from lhotse.cut import CutSet
from lhotse.testing.dummies import dummy_cut, dummy_supervision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cut(duration: float, start: float = 0.0, n_supervisions: int = 0):
    """Return a MonoCut whose recording exactly covers the requested duration."""
    supervisions = [
        dummy_supervision(i, start=0.0, duration=duration)
        for i in range(n_supervisions)
    ]
    return dummy_cut(
        0,
        start=start,
        duration=duration,
        recording_duration=start + duration,
        supervisions=supervisions,
    )


# ---------------------------------------------------------------------------
# Short-cut passthrough (duration <= max_duration)
# ---------------------------------------------------------------------------


def test_short_cut_returned_unchanged():
    """A cut that already fits in max_duration should be returned as-is."""
    cut = _make_cut(duration=30.0)
    result = list(cut.cut_into_windows_balanced(min_duration=30, max_duration=40))
    assert len(result) == 1
    assert result[0] == cut


def test_exactly_max_duration_returned_unchanged():
    cut = _make_cut(duration=40.0)
    result = list(cut.cut_into_windows_balanced(min_duration=30, max_duration=40))
    assert len(result) == 1
    assert result[0] == cut


# ---------------------------------------------------------------------------
# Basic windowing behaviour
# ---------------------------------------------------------------------------


def test_windows_cover_full_duration():
    """Every sample of the original cut should be reachable via the sub-cuts.

    For duration=95s, min=30, max=40, overlap=1s:
    best_duration=33 (hop=32) yields 3 chunks with last chunk=31s (the maximum).
    Windows: [0,33), [32,65), [64,95).
    """
    duration = 95.0
    overlap = 1.0
    cut = _make_cut(duration=duration)
    windows = list(
        cut.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=overlap)
    )

    assert len(windows) == 3
    assert windows[0].start == approx(0.0)
    assert windows[0].duration == approx(33.0)
    assert windows[1].start == approx(32.0)
    assert windows[1].duration == approx(33.0)
    assert windows[2].start == approx(64.0)
    assert windows[2].duration == approx(31.0)


def test_window_durations_are_uniform():
    """All windows except possibly the last should have the same chosen duration."""
    cut = _make_cut(duration=95.0)
    windows = list(
        cut.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=1.0)
    )
    first_dur = windows[0].duration
    # First window sets the template duration
    for w in windows[:-1]:
        assert w.duration == approx(first_dur)


def test_window_duration_within_bounds():
    """Chosen window size must be within [min_duration, max_duration]."""
    cut = _make_cut(duration=150.0)
    windows = list(
        cut.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=1.0)
    )
    for w in windows[:-1]:
        assert 30 <= w.duration <= 40


def test_consecutive_windows_overlap():
    """Consecutive windows should overlap by approximately `overlap` seconds."""
    overlap = 1.0
    cut = _make_cut(duration=95.0)
    windows = list(
        cut.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=overlap)
    )
    for i in range(len(windows) - 1):
        gap = windows[i + 1].start - windows[i].start
        hop = windows[i].duration - overlap
        assert gap == approx(hop)


# ---------------------------------------------------------------------------
# Custom metadata stamped on sub-cuts
# ---------------------------------------------------------------------------


def test_source_cut_id_is_set():
    """Every sub-cut must carry source_cut_id equal to the parent cut id."""
    cut = _make_cut(duration=95.0)
    windows = list(
        cut.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=1.0)
    )
    for w in windows:
        assert w.custom is not None
        assert w.custom["source_cut_id"] == cut.id


def test_source_cut_start_is_set():
    """Every sub-cut must carry source_cut_start equal to the parent cut start."""
    parent_start = 5.0
    cut = _make_cut(duration=95.0, start=parent_start)
    windows = list(
        cut.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=1.0)
    )
    for w in windows:
        assert w.custom["source_cut_start"] == approx(parent_start)


def test_sub_cut_ids_are_indexed():
    """Sub-cut IDs should follow the pattern ``{parent_id}-{index}``."""
    cut = _make_cut(duration=95.0)
    windows = list(
        cut.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=1.0)
    )
    for i, w in enumerate(windows):
        assert w.id == f"{cut.id}-{i}"


def test_existing_custom_attrs_are_preserved():
    """Pre-existing custom attributes on the cut should survive windowing."""
    cut = _make_cut(duration=95.0)
    # dummy_cut already adds custom_attribute
    assert cut.custom.get("custom_attribute") == "dummy-value"
    windows = list(
        cut.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=1.0)
    )
    for w in windows:
        assert w.custom.get("custom_attribute") == "dummy-value"


# ---------------------------------------------------------------------------
# Window-count optimisation (maximise last-chunk fill)
# ---------------------------------------------------------------------------


def test_chosen_duration_maximises_last_chunk():
    """
    Verify that the chosen window duration actually maximises the last-chunk
    duration compared to all other integer durations in [min, max].
    """
    total = 95.0
    min_d, max_d, overlap = 30, 40, 1.0
    cut = _make_cut(duration=total)
    windows = list(
        cut.cut_into_windows_balanced(
            min_duration=min_d, max_duration=max_d, overlap=overlap
        )
    )
    chosen = windows[0].duration

    def last_chunk_len(d):
        hop = d - overlap
        n = math.ceil(total / hop)
        return total - hop * (n - 1)

    chosen_last = last_chunk_len(chosen)
    for d in range(min_d, max_d + 1):
        assert (
            chosen_last >= last_chunk_len(d) - 1e-9
        ), f"Duration {chosen} was chosen but {d} yields a longer last chunk"


# ---------------------------------------------------------------------------
# CutSet variant
# ---------------------------------------------------------------------------


def test_cutset_balanced_produces_same_result_as_single_cut():
    """CutSet.cut_into_windows_balanced should match per-cut results."""
    cut = _make_cut(duration=95.0)
    cutset = CutSet.from_cuts([cut])
    cs_windows = list(
        cutset.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=1.0)
    )
    direct_windows = list(
        cut.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=1.0)
    )
    assert len(cs_windows) == len(direct_windows)
    for a, b in zip(cs_windows, direct_windows):
        assert a.id == b.id
        assert a.start == approx(b.start)
        assert a.duration == approx(b.duration)


def test_cutset_short_cuts_pass_through():
    """Short cuts (≤ max_duration) in a CutSet must pass through unchanged."""
    short = _make_cut(duration=25.0)
    long_ = _make_cut(duration=95.0)
    # Give them distinct ids
    short = short.with_id("short")
    long_ = long_.with_id("long")
    cutset = CutSet.from_cuts([short, long_])
    windows = list(
        cutset.cut_into_windows_balanced(min_duration=30, max_duration=40, overlap=1.0)
    )
    ids = [w.id for w in windows]
    assert "short" in ids  # passed through unchanged
    assert "long" not in ids  # was split; children are long-0, long-1, …
    assert any(w_id.startswith("long-") for w_id in ids)
