import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from multiprocessing.context import BaseContext
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

CaseT = TypeVar("CaseT")
PredictT = TypeVar("PredictT")
ResultT = TypeVar("ResultT")


DoWork = Callable[[CaseT, Callable[[CaseT], PredictT], Any], ResultT]


class ProcessWorker(Generic[CaseT, PredictT, ResultT]):
    """A wrapper for a function that does the actual work in a multiprocessing context."""

    models: Dict[Optional[int], Callable[[CaseT], PredictT]] = {}

    def __init__(
        self,
        gen_model: Callable[[], Callable[[CaseT], PredictT]],
        do_work: DoWork[CaseT, PredictT, ResultT],
        # "error", "ignore", "always", "default", "module", "once"
        warnings_mode: Optional[str] = None,
    ):
        self._gen_model = gen_model
        self._do_work = do_work
        self._warnings_mode = warnings_mode

    def _get_model(self) -> Callable[[CaseT], PredictT]:
        pid = multiprocessing.current_process().pid
        model = self.models.get(pid)
        if model is None:
            model = self._gen_model()
            self.models[pid] = model
        return model

    def __call__(self, obj: CaseT, **kwargs: Any) -> ResultT:
        if self._warnings_mode is not None:
            warnings.simplefilter(self._warnings_mode)  # type: ignore
        return self._do_work(obj, model=self._get_model(), **kwargs)

    def clear(self):
        self.models.clear()


class Processor(Generic[CaseT, PredictT, ResultT]):
    """Multiprocessing wrapper for ProcessWorker."""

    def __init__(
        self,
        worker: ProcessWorker[CaseT, PredictT, ResultT],
        *,
        num_jobs: int,
        verbose: bool = False,
        mp_context: Optional[Union[BaseContext, str]] = None,
    ):
        self._worker = worker
        self._num_jobs = num_jobs
        self._verbose = verbose

        if mp_context is None:
            self._mp_context = None
        else:
            if isinstance(mp_context, str):
                mp_context = multiprocessing.get_context(mp_context)
            self._mp_context = mp_context

    @contextmanager
    def handle_errors(self, pool: ProcessPoolExecutor):
        with pool as executor:
            try:
                yield executor
            except KeyboardInterrupt as exc:  # pragma: no cover
                pool.shutdown(wait=False)
                if self._verbose:
                    print("Processing interrupted by the user.")
                raise exc
            except Exception as exc:  # pragma: no cover
                pool.shutdown(wait=False)
                raise RuntimeError("Processing failed.") from exc
            finally:
                self._worker.clear()

    def __call__(
        self,
        sequence: Sequence[CaseT],
        **kwargs: Any,
    ) -> Iterable[ResultT]:
        """Iterate over the results of processing the sequence with the worker."""
        pool = ProcessPoolExecutor(
            max_workers=self._num_jobs,
            # initializer=...,
            mp_context=self._mp_context,
        )

        with self.handle_errors(pool) as executor:
            factory = partial(self._worker, **kwargs)
            results = executor.map(factory, sequence)

            if self._verbose:
                from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

                results = tqdm(
                    results,
                    total=len(sequence),
                    desc="Multiprocessing",
                    unit="task",
                )

            yield from results
