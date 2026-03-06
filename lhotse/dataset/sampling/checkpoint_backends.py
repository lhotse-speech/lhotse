import random
from typing import Any, Callable, Optional

from lhotse.dataset.sampling.base import EpochDiagnostics


def _is_cutset(obj: Any) -> bool:
    from lhotse.cut import CutSet

    return isinstance(obj, CutSet)


def _all_sources_indexed(sampler: Any) -> bool:
    return all(
        getattr(cs, "is_indexed", False)
        for cs in getattr(sampler, "cuts", ())
        if _is_cutset(cs)
    )


def _has_cuts_state(cuts_state: Optional[list]) -> bool:
    return cuts_state is not None and any(state is not None for state in cuts_state)


class IndexedCheckpointBackend:
    """
    O(1) checkpoint backend for indexed datasets.

    This backend is strict: missing state or restore failures are treated
    as hard errors and never fall back to O(N) replay.
    """

    def __init__(
        self,
        *,
        has_required_state: bool,
        restore_fn: Callable[[], None],
        missing_state_message: str,
        failure_message: str,
    ) -> None:
        self.has_required_state = has_required_state
        self.restore_fn = restore_fn
        self.missing_state_message = missing_state_message
        self.failure_message = failure_message

    def restore(self) -> None:
        if not self.has_required_state:
            raise RuntimeError(self.missing_state_message)
        try:
            self.restore_fn()
        except Exception as e:
            raise RuntimeError(f"{self.failure_message} Error: {e}") from e


class ReplayCheckpointBackend:
    """
    O(N) replay backend.

    Replays ``num_steps`` batches after rebuilding the sampler iterator state.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        reset_for_replay_fn: Callable[[], None],
        initialize_iterator_fn: Callable[[], None],
        replay_step_fn: Callable[[], None],
        post_restore_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self.num_steps = num_steps
        self.reset_for_replay_fn = reset_for_replay_fn
        self.initialize_iterator_fn = initialize_iterator_fn
        self.replay_step_fn = replay_step_fn
        self.post_restore_fn = post_restore_fn

    def restore(self) -> None:
        self.reset_for_replay_fn()
        self.initialize_iterator_fn()
        for _ in range(self.num_steps):
            self.replay_step_fn()
        if self.post_restore_fn is not None:
            self.post_restore_fn()


def build_dynamic_cut_checkpoint_backend(
    sampler: Any, *, current_epoch: int, num_batches_to_iter: int
) -> Any:
    cuts_state = getattr(sampler, "_cuts_state", None)
    has_state = _has_cuts_state(cuts_state)
    replay_backend = _build_dynamic_cut_replay_backend(
        sampler=sampler,
        current_epoch=current_epoch,
        num_batches_to_iter=num_batches_to_iter,
    )

    if _all_sources_indexed(sampler):
        return IndexedCheckpointBackend(
            has_required_state=has_state,
            restore_fn=lambda: _restore_dynamic_cut_indexed(sampler, cuts_state),
            missing_state_message=(
                "O(1) indexed restore is missing required checkpoint state for "
                "DynamicCutSampler. This is a bug — indexed datasets should never "
                "use O(N) fast-forward. "
                f"State flags: has_cuts_state={has_state}, "
                f"num_batches_to_iter={num_batches_to_iter}."
            ),
            failure_message=(
                "O(1) indexed restore failed for indexed datasets. This is a bug — "
                "indexed datasets should never use O(N) fast-forward."
            ),
        )

    return replay_backend


def _build_dynamic_cut_replay_backend(
    *, sampler: Any, current_epoch: int, num_batches_to_iter: int
) -> ReplayCheckpointBackend:
    def _reset_diagnostics_for_replay() -> None:
        sampler.diagnostics.stats_per_epoch[current_epoch] = EpochDiagnostics(
            epoch=current_epoch
        )

    return ReplayCheckpointBackend(
        num_steps=num_batches_to_iter,
        reset_for_replay_fn=_reset_diagnostics_for_replay,
        initialize_iterator_fn=sampler._initialize_replay_iterator,
        replay_step_fn=sampler._replay_step,
        post_restore_fn=lambda: setattr(sampler, "_just_restored_state", True),
    )


def _restore_dynamic_cut_indexed(sampler: Any, cuts_state: list) -> None:
    for cs, cs_state in zip(sampler.cuts, cuts_state):
        if _is_cutset(cs) and cs_state is not None:
            cs.load_state_dict(cs_state)

    sampler._just_restored_state = False
    sampler._cuts_state = None
    sampler._skip_diagnostics_reset_once = True
    iter(sampler)
    sampler._restore_transforms_state()
    sampler._just_restored_state = True


def build_dynamic_bucketing_checkpoint_backend(
    sampler: Any, *, current_epoch: int, num_batches_to_iter: int
) -> Any:
    cuts_state = getattr(sampler, "_cuts_state", None)
    rng_state = getattr(sampler, "_rng_state", None)
    bucketer_state = getattr(sampler, "_bucketer_state", None)
    has_cuts_state = _has_cuts_state(cuts_state)
    has_full_state = (
        has_cuts_state and rng_state is not None and bucketer_state is not None
    )

    replay_backend = _build_dynamic_bucketing_replay_backend(
        sampler=sampler,
        current_epoch=current_epoch,
        num_batches_to_iter=num_batches_to_iter,
    )

    if _all_sources_indexed(sampler):
        if has_full_state:
            return IndexedCheckpointBackend(
                has_required_state=True,
                restore_fn=lambda: _restore_dynamic_bucketing_full(
                    sampler,
                    cuts_state=cuts_state,
                    rng_state=rng_state,
                    bucketer_state=bucketer_state,
                ),
                missing_state_message="",
                failure_message=(
                    "O(1) indexed restore failed for indexed datasets. This is a bug — "
                    "indexed datasets should never use O(N) fast-forward."
                ),
            )
        if num_batches_to_iter == 0:
            return IndexedCheckpointBackend(
                has_required_state=True,
                restore_fn=lambda: _restore_dynamic_bucketing_pre_yield(sampler),
                missing_state_message="",
                failure_message=(
                    "O(1) indexed restore (pre-yield) failed for indexed datasets. "
                    "This is a bug — indexed datasets should never use O(N) fast-forward."
                ),
            )
        return IndexedCheckpointBackend(
            has_required_state=False,
            restore_fn=lambda: None,
            missing_state_message=(
                "O(1) indexed restore is missing required checkpoint state for "
                "DynamicBucketingSampler. This is a bug — indexed datasets should "
                "never use O(N) fast-forward. "
                f"State flags: has_cuts_state={has_cuts_state}, "
                f"has_rng_state={rng_state is not None}, "
                f"has_bucketer_state={bucketer_state is not None}, "
                f"num_batches_to_iter={num_batches_to_iter}."
            ),
            failure_message="",
        )

    return replay_backend


def _build_dynamic_bucketing_replay_backend(
    *, sampler: Any, current_epoch: int, num_batches_to_iter: int
) -> ReplayCheckpointBackend:
    def _reset_diagnostics_for_replay() -> None:
        sampler.diagnostics.stats_per_epoch[current_epoch] = EpochDiagnostics(
            epoch=current_epoch
        )

    return ReplayCheckpointBackend(
        num_steps=num_batches_to_iter,
        reset_for_replay_fn=_reset_diagnostics_for_replay,
        initialize_iterator_fn=sampler._initialize_replay_iterator,
        replay_step_fn=sampler._replay_step,
        post_restore_fn=lambda: setattr(sampler, "_just_restored_state", True),
    )


def _restore_dynamic_bucketing_full(
    sampler: Any, *, cuts_state: list, rng_state: list, bucketer_state: dict
) -> None:
    from lhotse.checkpoint import _rng_state_from_json

    sampler.rng = random.Random()
    sampler.rng.setstate(_rng_state_from_json(rng_state))

    for cs, cs_state in zip(sampler.cuts, cuts_state):
        if _is_cutset(cs) and cs_state is not None:
            cs.load_state_dict(cs_state)

    sampler._just_restored_state = False
    sampler._cuts_state = None
    sampler._rng_state = None
    sampler._bucketer_state = None
    sampler._skip_diagnostics_reset_once = True
    iter(sampler)
    sampler._bucketer.set_state(bucketer_state)
    sampler._restore_transforms_state()
    sampler._just_restored_state = True


def _restore_dynamic_bucketing_pre_yield(sampler: Any) -> None:
    sampler._just_restored_state = False
    sampler._cuts_state = None
    sampler._rng_state = None
    sampler._bucketer_state = None
    sampler._skip_diagnostics_reset_once = True
    iter(sampler)
    sampler._restore_transforms_state()
    sampler._just_restored_state = True
