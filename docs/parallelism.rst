Executing tasks in parallel
===========================

In this section we will explain how Lhotse uses a generic interface called ``Executor`` to
parallelize some tasks (mostly feature extraction).

There are multiple ways we can parallelize execution of a Python method:

- using multi-threading (single node, single process);
- using multi-processing (single node, multiplce processes);
- using distributed processing (multiple nodes, multiple processes).

The ``Executor`` API, introduced in Python's standard library in :mod:`concurrent.futures` module,
allows us to use any of these methods, while writing the code independently of how it is going to be parallelized.
This module defines two types of *executors*, i.e. :class:`concurrent.futures.ProcessPoolExecutor`
and :class:`concurrent.futures.ThreadPoolExecutor`.
We refer the reader to `the official documentation of concurrent.futures`_ for details.
On a high level, these executors accepts tasks in the form of a Python function and an iterable of arguments,
and then distribute the tasks among workers, automatically balancing the load
(no manual splitting into chunks/batches is necessary).

Some methods in Lhotse (notably: :meth:`lhotse.CutSet.compute_and_store_features`) have a parameter called ``executor``,
which is set to ``None`` by default.
It means that by default, they are going to run everything in a single thread and process.
The user can pass an object satisfying the ``Executor`` API instead, and these methods will
automatically parallelize the underlying tasks.

**Multi-processing**: This is the recommended way to parallelize the execution for most users.
An example of use to extract features on a :class:`lhotse.CutSet`:

.. code-block::

    from concurrent.futures import ProcessPoolExecutor
    from lhotse import CutSet, Fbank, LilcomFilesWriter
    num_jobs = 8
    with ProcessPoolExecutor(num_jobs) as ex:
        cuts: CutSet = cuts.compute_and_store_features(
            extractor=Fbank(),
            storage=LilcomFilesWriter('feats'),
            executor=ex
        )

**Distributed processing**: This is the recommended way for more advanced users that have the access and desire to
leverage high-performance clusters (e.g. at universities).
A library called `Dask`_ offers multiple powerful Python interfaces for distributed execution.
One of them is called `Dask.distributed`_.
It implements the ``Executor`` API via class ``distributed.Client``, that can be connected to an existing
Dask cluster.
The setup of Dask clusters is beyond the scope of this documentation, however you can find a working
implementation for the `CLSP Sun Grid Engine system here`_.

.. caution::

    Dask is an optional dependency for Lhotse and has to be installed separately.
    You can install it with ``pip install dask distributed``.

**Multi-threading**: We discourage using multi-threading with Python.
Python is well known for its issues with multi-threading due to global interpreter lock (GIL), which
prohibits most "typical" multi-threaded code from running in parallel. Therefore, usually concurrent tasks
have to be executed in separate processes (each with its own interpreter), or use threading at the native
(C or C++) level. Lhotse currently does not implement any native components, so we rely on Python-level parallelism.

If you are sure that you want to use multi-threading, you can use ``concurrent.futures.ThreadPoolExecutor``.
We use it sometimes in Lhotse when we expect the operations to be I/O bound rather than CPU bound
(like scanning the filesystem for multiple files).

.. _Dask: https://dask.org
.. _the official documentation of concurrent.futures: https://docs.python.org/3.8/library/concurrent.futures.html
.. _Dask.distributed: https://distributed.dask.org/en/latest/
.. _CLSP Sun Grid Engine system here: https://github.com/pzelasko/plz