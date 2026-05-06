Indexed Manifests and IteratorNodes
===================================

Indexed manifests are the foundation for exact O(1) restore of Lhotse's
dataloading pipeline. This page explains what they are, how they compose through
lazy iterator graphs, how checkpointing uses them, and what contract new
``IteratorNode`` implementations must satisfy.

What an indexed manifest is
---------------------------

An indexed manifest is a lazy manifest backed by an auxiliary binary ``.idx``
file that lets Lhotse jump directly to a specific example instead of scanning
from the beginning.

Typical examples are:

* an uncompressed ``.jsonl`` manifest with ``cuts.jsonl.idx``
* an uncompressed Shar manifest shard such as ``cuts.000000.jsonl`` together
  with ``cuts.000000.jsonl.idx``
* an uncompressed tar shard together with ``recording.000000.tar.idx``

When the underlying data is indexed, Lhotse can reconstruct a buffered example
directly during checkpoint restore rather than replaying earlier batches.

Creating indexes
----------------

For standalone manifests:

.. code-block:: bash

   lhotse index jsonl /path/to/cuts.jsonl
   lhotse index tar /path/to/recording.tar

For Shar:

.. code-block:: bash

   lhotse index shar /path/to/shar_dir/

When writing Shar from Python, keep the cuts manifest uncompressed and enable
index creation:

.. code-block:: python

   from lhotse.shar import SharWriter

   writer = SharWriter(
       "data/",
       fields={"recording": "wav"},
       shard_size=1000,
       compress_jsonl=False,
       create_index=True,
   )

.. note::

   Indexed access requires **uncompressed, seekable** data sources.
   ``.jsonl.gz`` and ``pipe:...`` inputs are valid for sequential streaming,
   but they do not provide constant-time reconstruction. Local files and
   supported remote/object-store URIs can be indexed as long as the storage
   backend supports indexed reads.

Reading indexed data
--------------------

Use ``indexed=True`` when reading plain manifests:

.. code-block:: python

   from lhotse import CutSet

   cuts = CutSet.from_file("cuts.jsonl", indexed=True)

For Shar:

.. code-block:: python

   cuts = CutSet.from_shar(in_dir="data/", indexed=True)

``CutSet.from_shar(..., indexed=None)`` will auto-detect indexed mode when all
requested field shards are uncompressed, indexable, and have matching indexes
available.

How iterator composition works
------------------------------

Lhotse builds a lazy iterator graph underneath ``CutSet``. The concrete nodes
in that graph are subclasses of :class:`lhotse.lazy.IteratorNode`.

Examples:

* ``CutSet.from_file(..., indexed=True)`` creates a
  :class:`lhotse.lazy.LazyIndexedManifestIterator`
* ``cuts.filter(...)`` wraps the current iterator with
  :class:`lhotse.lazy.LazyFilter`
* ``cuts.map(...)`` wraps it with :class:`lhotse.lazy.LazyMapper`
* ``CutSet.mux(a, b)`` creates :class:`lhotse.lazy.LazyIteratorMultiplexer`
* ``cuts.mix(...)`` creates :class:`lhotse.cut.set.LazyCutMixer`

``CutSet`` itself is just a manifest wrapper. Graph-building code should work on
the underlying iterator stored in ``CutSet.data``. The helper
``resolve_iterator_source()`` does exactly that.

Three important capabilities
----------------------------

Every ``IteratorNode`` exposes three related but distinct properties:

* ``is_checkpointable``: the node can save and restore its internal iteration
  state.
* ``is_indexed``: the node is backed by indexed data.
* ``has_constant_time_access``: the node can reconstruct a specific output item
  directly through ``__getitem__``.

``has_constant_time_access`` is the crucial property for exact O(1) restore.
It does **not** mean the node has a dense integer index. It only means the node
can take a restore token and rebuild the matching output item directly.

For example:

* a plain indexed manifest may use integer tokens such as ``17``
* a multiplexer may use ``(source_idx, child_token)``
* a repeater may use ``(repeat_idx, child_token)``
* an indexed Shar iterator may use ``(global_idx, shar_epoch)``

Property summary for built-in iterators
---------------------------------------

The following table summarizes the three properties for the iterator nodes
shipped with Lhotse. ``checkpointable`` means ``state_dict`` /
``load_state_dict`` work; ``indexed`` means the node is backed by
random-access data; ``O(1)`` means ``__getitem__`` reconstructs the exact
item directly from a graph token without replay.

"delegates" means the property is a Python ``@property`` that returns the
value of the same property on the wrapped source — so the answer depends on
what you compose the transform with. To check at runtime, use
``getattr(node, "is_indexed", False)`` etc., or the helper
``supports_graph_restore(node)`` for a combined check.

================================  ================  ===========  ===========
Iterator                          Checkpointable    Indexed      O(1)
================================  ================  ===========  ===========
``LazyJsonlIterator``             no                no           no
``LazyManifestIterator``          yes               no           no
``LazyIndexedManifestIterator``   yes               delegates    delegates
``LazySharIterator``              yes               no           no
``LazyIndexedSharIterator``       yes               delegates    delegates
``LazyIteratorChain``             yes               delegates    delegates
``LazyIteratorMultiplexer``       yes               delegates    delegates
``LazyMapper``                    yes               delegates    delegates
``LazyFilter``                    yes               delegates    delegates
``LazyShuffler``                  delegates         delegates    delegates
``LazyFlattener``                 delegates         delegates    delegates
``LazyCutMixer``                  delegates         delegates    delegates
``LazyRepeater``                  yes               delegates    delegates
================================  ================  ===========  ===========

The leaf manifest iterators (``LazyJsonlIterator``, ``LazyManifestIterator``,
``LazySharIterator``) are streaming-only. To get indexed / O(1) behavior,
construct them via ``CutSet.from_file(..., indexed=True)`` /
``CutSet.from_shar(..., indexed=True)``, which build
``LazyIndexedManifestIterator`` / ``LazyIndexedSharIterator`` instead.

Checkpointing: graph tokens
---------------------------

Indexed restore uses ``_graph_origin`` tokens attached to yielded items:

1. the sampler saves a token for every buffered item
2. on restore it calls ``source[token]``
3. each iterator node consumes its part of the token and delegates the rest to
   its child
4. the graph reconstructs the same output item without replay

Why this enables O(1) restore
-----------------------------

Replay-based restore starts from the beginning and skips already-consumed data.
For large blends that is expensive.

Graph-token restore avoids replay:

* the sampler restores its own RNG and buffer state
* buffered items are rebuilt from their saved tokens
* iterator nodes restore their internal cursor/RNG state
* iteration continues from the next batch

For indexed datasets, Lhotse treats this as a strict contract:

* missing indexed checkpoint state is a hard error
* missing ``_graph_origin`` on graph-restorable buffered items is a hard error
* Lhotse does not silently downgrade indexed exact restore to replay

Worker-process restore
----------------------

With ``torchdata.stateful_dataloader.StatefulDataLoader``, each worker process
stores its own iterator graph state. Indexed restore works in workers for the
same reason it works in the main process: workers also rebuild buffered items
from graph tokens rather than replaying from the start.

Implementing a new IteratorNode
-------------------------------

Start with this checklist.

1. Derive from :class:`lhotse.lazy.IteratorNode`.
2. Store children in ``self.source`` or ``self.sources``.
3. Call ``resolve_iterator_source()`` in ``__init__`` so ``CutSet`` wrappers do
   not leak into the graph.
4. Set ``is_checkpointable`` correctly.
5. Delegate child checkpoint state from ``state_dict()`` / ``load_state_dict()``
   when the child is checkpointable.
6. If the node supports exact reconstruction, implement ``__getitem__`` and
   ``has_constant_time_access``.
7. Propagate graph tokens through iteration and reconstruction.
8. If the node has mutable iteration state, include that state in
   ``state_dict()`` and ``load_state_dict()``.

Checkpointable does not imply O(1)
----------------------------------

``is_checkpointable`` and ``has_constant_time_access`` are different contracts.

Examples of checkpointable but non-O(1) nodes in the codebase:

* ``LazyManifestIterator``
* ``LazySharIterator``

Set ``is_checkpointable = True`` when the node can resume exactly, even if that
resume path is sequential rather than random-access. Set
``has_constant_time_access = True`` only when ``__getitem__`` can rebuild a
specific output item directly from a token.

Minimal stateless transform node
--------------------------------

This is the simplest useful pattern for a transform that preserves exact O(1)
restore when its source supports it. The important detail is that a
checkpointable parent must delegate child state explicitly; checkpoint graph
traversal does not recurse into checkpointable children of a checkpointable
parent.

.. code-block:: python

   from lhotse.lazy import (
       IteratorNode,
       attach_graph_origin,
       get_graph_origin,
       maybe_attach_graph_origin,
       normalize_graph_token,
       resolve_iterator_source,
       supports_graph_restore,
   )


   class MyTransform(IteratorNode):
       is_checkpointable = True

       def __init__(self, source):
           self.source = resolve_iterator_source(source)

       @property
       def is_indexed(self):
           return getattr(self.source, "is_indexed", False)

       @property
       def has_constant_time_access(self):
           return supports_graph_restore(self.source)

       def __getitem__(self, token):
           token = normalize_graph_token(token)
           item = self.source[token]
           item = transform(item)
           return attach_graph_origin(item, token)

       def __iter__(self):
           for item in self.source:
               yield maybe_attach_graph_origin(transform(item), get_graph_origin(item))

       def state_dict(self):
           state = {}
           if getattr(self.source, "is_checkpointable", False):
               state["source"] = self.source.state_dict()
           return state

       def load_state_dict(self, state):
           if "source" in state:
               self.source.load_state_dict(state["source"])

Stateful node with RNG or cursor state
--------------------------------------

If the node has its own position, RNG state, or per-epoch behavior, that state
must be part of the checkpoint. Typical examples are:

* multiplexer choice RNG
* iterator position within a shard
* current repeat epoch
* per-iteration seed used to derive deterministic item-level randomness

State restoration must satisfy this rule:

    after ``load_state_dict()``, the node must yield the same remaining outputs
    as an uninterrupted run

When a node also supports exact reconstruction, its token must contain all
information needed to rebuild the exact output. Indexed Shar is a good example:
the token includes both the global item index and the Shar epoch metadata that
was attached when the item was first produced.

When a node should not support exact restore
--------------------------------------------

Some iterator shapes are intentionally not checkpointable or not exact:

* infinite approximate multiplexers
* transforms whose output cannot be reconstructed exactly and whose in-flight
  state cannot be serialized compactly

For these nodes:

* keep ``is_checkpointable = False`` only when exact resumption of any kind is
  not implementable
* do not claim ``has_constant_time_access = True``
* prefer an explicit exception over an implicit fallback

Nodes such as ``LazyShuffler`` and ``LazyFlattener`` can still be exact for
indexed outer sources by saving compact local state:

* ``LazyShuffler`` saves its shuffle buffer and RNG state
* ``LazyFlattener`` saves the outer token and the local offset inside the
  current inner collection

Transforms that change cardinality
**********************************

Transforms whose output cardinality depends on the input data — i.e. one
input row produces a variable number of output rows — *can* still preserve
indexedness and O(1) restore, but only when they implement the composite-
token contract.

``LazyFlattener`` is the canonical example. Each input row contains a
collection that may have anywhere between 0 and N items, and the flattener
emits ``(outer_token, inner_token)`` pairs as graph tokens. Its
``__getitem__((outer_token, inner_token))`` rebuilds the right inner item
by indexing into the outer source first and then into the materialized
collection. So a ``LazyFlattener`` over an indexed source remains indexed,
checkpointable, and O(1) — there is no replay degradation.

The pitfall is with **custom** cardinality-changing transforms that don't
implement that contract. If your transform yields more than one item per
input row but does not expose ``__getitem__`` taking a composite token (and
does not propagate graph origins on yielded items), the resulting iterator
can no longer reconstruct a specific output item from a saved token. In an
indexed pipeline this surfaces as a hard error from Lhotse's strict
graph-token contract above.

Two safe patterns for custom cardinality-changing transforms:

* **Materialize offline.** Run the transform once and write the expanded
  manifest. Downstream iterators then see one row per output item and the
  whole graph stays indexed end-to-end. This is the simplest option when
  the expansion is deterministic and the storage cost is acceptable.
* **Implement composite tokens** following the ``LazyFlattener`` pattern:
  emit ``(outer_token, inner_token)`` graph origins from ``__iter__``,
  implement ``__getitem__`` to dispatch on them, and delegate
  ``is_indexed`` / ``has_constant_time_access`` to the source.

Runtime metadata rules
----------------------

Checkpoint metadata such as ``_graph_origin`` is runtime-only. Do not attach it
through normal custom fields on cuts, because that would serialize it into
manifests.

Use ``attach_graph_origin(...)``. This helper bypasses cut serialization hooks
and keeps checkpoint metadata process-local.

Testing new IteratorNodes
-------------------------

A new node should have tests for:

* uninterrupted iteration vs checkpoint/restore equality
* main-process restore
* worker-process restore when supported
* graph-token propagation if it wraps another indexed node
* failure behavior when reconstruction is impossible
* strict errors when the node claims graph restore support but emitted items are
  missing ``_graph_origin``

The checkpoint matrix in ``test/test_iterator_node_e2e_checkpoint.py`` is the
main coverage gate for production ``IteratorNode`` subclasses.
