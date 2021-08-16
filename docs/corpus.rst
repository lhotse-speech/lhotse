Representing a corpus
=====================

In Lhotse, we represent the data using a small number of Python classes, enhanced with methods for solving common data manipulation tasks, that can be stored as JSON or JSONL manifests.
For most audio corpora, we will need two types of manifests to fully describe them:
a recording manifest and a supervision manifest.

Recording manifest
------------------

.. autoclass:: lhotse.audio.Recording
  :no-members:
  :no-special-members:
  :noindex:

.. autoclass:: lhotse.audio.RecordingSet
  :no-members:
  :no-special-members:
  :noindex:

Supervision manifest
--------------------

.. autoclass:: lhotse.supervision.SupervisionSegment
  :no-members:
  :no-special-members:
  :noindex:

.. autoclass:: lhotse.supervision.SupervisionSet
  :no-members:
  :no-special-members:
  :noindex:

Standard data preparation recipes
---------------------------------

We provide a number of standard data preparation recipes. By that, we mean a collection of a Python function +
a CLI tool that create the manifests given a corpus directory.

.. list-table:: Currently supported corpora
  :widths: 30 50
  :header-rows: 1

  * - Corpus name
    - Function
  * - Aishell
    - :func:`lhotse.recipes.prepare_aishell`
  * - AMI
    - :func:`lhotse.recipes.prepare_ami`
  * - BABEL
    - :func:`lhotse.recipes.prepare_single_babel_language`
  * - CallHome Egyptian
    - :func:`lhotse.recipes.prepare_callhome_egyptian`
  * - CallHome English
    - :func:`lhotse.recipes.prepare_callhome_english`
  * - CMU Arctic
    - :func:`lhotse.recipes.prepare_cmu_arctic`
  * - CMU Kids
    - :func:`lhotse.recipes.prepare_cmu_kids`
  * - CSLU Kids
    - :func:`lhotse.recipes.prepare_cslu_kids`
  * - DIHARD III
    - :func:`lhotse.recipes.prepare_dihard3`
  * - English Broadcast News 1997
    - :func:`lhotse.recipes.prepare_broadcast_news`
  * - Fisher English Part 1, 2
    - :func:`lhotse.recipes.prepare_fisher_english`
  * - Fisher Spanish
    - :func:`lhotse.recipes.prepare_fisher_spanish`
  * - GALE Arabic Broadcast Speech
    - :func:`lhotse.recipes.prepare_gale_arabic`
  * - GALE Mandarin Broadcast Speech
    - :func:`lhotse.recipes.prepare_gale_mandarin`
  * - GigaSpeech
    - :func:`lhotse.recipes.prepare_gigaspeech`
  * - Heroico
    - :func:`lhotse.recipes.prepare_heroico`
  * - L2 Arctic
    - :func:`lhotse.recipes.prepare_l2_arctic`
  * - LibriSpeech (including "mini")
    - :func:`lhotse.recipes.prepare_librispeech`
  * - LibriTTS
    - :func:`lhotse.recipes.prepare_libritts`
  * - LJ Speech
    - :func:`lhotse.recipes.prepare_ljspeech`
  * - MiniLibriMix
    - :func:`lhotse.recipes.prepare_librimix`
  * - MTEDx
    - :func:`lhotse.recipes.prepare_mtdex`
  * - MobvoiHotWord
    - :func:`lhotse.recipes.prepare_mobvoihotwords`
  * - Multilingual LibriSpeech (MLS)
    - :func:`lhotse.recipes.prepare_mls`
  * - MUSAN
    - :func:`lhotse.recipes.prepare_musan`
  * - National Speech Corpus (Singaporean English)
    - :func:`lhotse.recipes.prepare_nsc`
  * - Switchboard
    - :func:`lhotse.recipes.prepare_switchboard`
  * - TED-LIUM v3
    - :func:`lhotse.recipes.prepare_tedlium`
  * - VCTK
    - :func:`lhotse.recipes.prepare_vctk`


Adding new corpora
------------------

.. hint::
    **Python data preparation recipes.** Each corpus has a dedicated Python file in ``lhotse/recipes``,
    which you can use as the basis for your own recipe.

.. hint::
    **(optional) Downloading utility.** For publicly available corpora that can be freely downloaded,
    we usually define a function called ``download_<corpus-name>()``.

.. hint::
    **Data preparation Python entry-point.** Each data preparation recipe should expose a single function
    called ``prepare_<corpus-name>``,
    that produces dicts like: ``{'recordings': <RecordingSet>, 'supervisions': <SupervisionSet>}``.


.. hint::
    **CLI recipe wrappers.** We provide a command-line interface that wraps the ``download`` and ``prepare``
    functions -- see ``lhotse/bin/modes/recipes`` for examples of how to do it.

.. hint::
    **Pre-defined train/dev/test splits.** When a corpus defines standard split (e.g. train/dev/test),
    we return a dict with the following structure:
    ``{'train': {'recordings': <RecordingSet>, 'supervisions': <SupervisionSet>}, 'dev': ...}``

.. hint::
    **Isolated utterance corpora.** Some corpora (like LibriSpeech) come with pre-segmented recordings.
    In these cases, the :class:`~lhotse.supervision.SupervisionSegment` will exactly match the
    :class:`~lhotse.recording.Recording` duration
    (and there will likely be exactly one segment corresponding to any recording).

.. hint::
    **Conversational corpora.** Corpora with longer recordings (e.g. conversational, like Switchboard) should have exactly one
    :class:`~lhotse.audio.Recording` object corresponding to a single conversation/session,
    that spans its whole duration.
    Each speech segment in that recording should be represented as a :class:`~lhotse.supervision.SupervisionSegment`
    with the same ``recording_id`` value.

.. hint::
    **Multi-channel corpora.** Corpora with multiple channels for each session (e.g. AMI) should have a single
    :class:`~lhotse.audio.Recording` with multiple :class:`~lhotse.audio.AudioSource` objects --
    each corresponding to a separate channel.
    Remember to make the :class:`~lhotse.supervision.SupervisionSegment` objects correspond to the right channels!


.. _`pysoundfile`: https://pysoundfile.readthedocs.io/en/latest/
.. _`audioread`: https://github.com/beetbox/audioread
