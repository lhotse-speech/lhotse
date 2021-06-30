Representing a corpus
=====================

In Lhotse, we represent the data using YAML (more readable), JSON, or JSONL (faster) manifests.
For most audio corpora, we will need two types of manifests to fully describe them:
a recording manifest and a supervision manifest.

.. caution::
    We show all the examples in YAML format for improved readability. However, when processing medium/large datasets, we recommend to use JSON or JSONL, which are much quicker to load and save.

Recording manifest
------------------

.. autoclass:: lhotse.audio.Recording
  :no-members:
  :noindex:

.. autoclass:: lhotse.audio.RecordingSet
  :no-members:
  :noindex:

Supervision manifest
--------------------

The supervision manifest contains the supervision information that we have about the recordings.
In particular, it involves the segmentation - there might be a single segment for a single utterance recording,
and multiple segments for a recording of a converstion.

When coming from Kaldi, think of it as a *segments* file on steroids,
that also contains *utt2spk*, *utt2gender*, *utt2dur*, etc.

This is a YAML supervision manifest:

.. code-block:: yaml

    ---
    - id: 'segment-1'
      recording_id: 'recording-2'
      channel: 0
      start: 0.1
      duration: 0.3
      text: 'transcript of the first segment'
      language: 'english'
      speaker: 'Norman Dyhrentfurth'

    - id: 'segment-2'
      recording_id: 'recording-2'
      start: 0.5
      duration: 0.4

Each segment is characterized by the following attributes:

- a unique id,
- a corresponding recording id,
- start time in seconds, relative to the beginning of the recording,
- the duration in seconds

Each segment may be assigned optional supervision information. In this example, the first segment
contains the transcription text, the language of the utterance and a speaker name.
The second segment contains only the minimal amount of information, which should be interpreted as:
"this is some area of interest in the recording that we know nothing else about."

Python
******

In Python, the supervision manifest is represented by classes :class:`SupervisionSet` and :class:`SupervisionSegment`.
Example usage:

.. code-block:: python

    supervisions = SupervisionSet.from_segments([
        SupervisionSegment(
            id='segment-1',
            recording_id='recording-1',
            start=0.5,
            duration=10.7,
            text='quite a long utterance'
        )
    ])
    print(f'There is {len(supervisions)} supervision in the set.')


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

General pointers:

* Each corpus has a dedicated Python file in ``lhotse/recipes``.
* For publicly available corpora that can be freely downloaded, we usually define a function called ``download``, ``download_and_untar``, etc.
* Each data preparation recipe should expose a single function called ``prepare_X``, with X being the name of the corpus, that produces dicts like: ``{'recordings': <RecordingSet>, 'supervisions': <SupervisionSet>}`` for the data in that corpus.
* When a corpus defines standard split (e.g. train/dev/test), we return a dict with the following structure: ``{'train': {'recordings': <RecordingSet>, 'supervisions': <SupervisionSet>}, 'dev': ...}``
* Some corpora (like LibriSpeech) come with pre-segmented recordings. In these cases, the :class:`SupervisionSegment` will exactly match the :class:`Recording` duration (and there will likely be exactly one segment corresponding to any recording).
* Corpora with longer recordings (e.g. conversational, like Switchboard) should have exactly one :class:`Recording` object corresponding to a single conversation/session, that spans its whole duration. Each speech segment in that recording should be represented as a :class:`SupervisionSegment` with the same ``recording_id`` value.
* Corpora with multiple channels for each session (e.g. AMI) should have a single :class:`Recording` with multiple :class:`AudioSource` objects - each corresponding to a separate channel.


.. _`pysoundfile`: https://pysoundfile.readthedocs.io/en/latest/
.. _`audioread`: https://github.com/beetbox/audioread
