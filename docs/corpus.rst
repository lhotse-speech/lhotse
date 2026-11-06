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

Every recipe follows the same rules, so once you have used one, you know how to use all of them.

Running a recipe
****************

From Python, call ``prepare_<corpus>`` with the directory where the corpus is stored.
If you also pass ``output_dir``, the manifests are saved there::

    >>> from lhotse.recipes import prepare_librispeech
    >>> manifests = prepare_librispeech("/data/LibriSpeech", output_dir="data/manifests")

From the command line, the same recipe is::

    lhotse prepare librispeech /data/LibriSpeech data/manifests

Many public corpora can also be downloaded first with ``download_<corpus>`` (or ``lhotse download <corpus>``).

What you get back
*****************

The result is a dictionary with one entry per **part** of the corpus (its train, dev and test splits).
Each part holds three manifests:

.. code-block:: python

    {
        "train-clean-100": {
            "recordings": RecordingSet,      # the audio files
            "supervisions": SupervisionSet,  # what is said, by whom, and when
            "cuts": CutSet,                  # recordings and supervisions together, ready for training
        },
        "dev-clean": {...},
        ...
    }

A few details:

* Parts use the corpus's own split names. If a corpus has no splits, the only part is called ``"all"``.
* Recipes that prepare several languages in one call add one more level on top: ``manifests[language][part]``.
* Noise, music and room impulse response corpora have no transcripts, so they have no ``"supervisions"``.
* A few corpora add an extra manifest next to these three, for example ``"uem"`` in DIHARD III.

The cuts are usually what you want. There is **one cut per recording**, covering the whole recording,
with all of that recording's supervisions attached. For corpora of short utterances (like LibriSpeech),
that means one cut per utterance. For long recordings (meetings, phone calls, lectures, podcasts),
each cut is a full session; split it into smaller pieces when you need to:

.. code-block:: python

    cuts = manifests["train"]["cuts"]
    utterances = cuts.trim_to_supervisions()   # one cut per supervision segment
    windows = cuts.cut_into_windows(duration=30.0)  # or fixed-length chunks

Files on disk
*************

With ``output_dir``, each manifest is saved as a compressed JSONL file named::

    <corpus>_<manifest>_<part>.jsonl.gz

* ``<corpus>`` is the corpus name in lowercase, with words joined by ``-``. Variants chosen when running
  the recipe, such as the microphone type or the language, are added with more ``-``:
  ``ami-ihm``, ``mls-polish``, ``cv-en``.
* ``<manifest>`` is ``recordings``, ``supervisions`` or ``cuts`` (or the name of an extra manifest).
* ``<part>`` is the same part name you see in the returned dictionary.

For example, ``librispeech_cuts_train-clean-100.jsonl.gz`` or ``ami-ihm_supervisions_dev.jsonl.gz``.
All files go directly into ``output_dir``. Load them later with
:func:`lhotse.load_manifest` or ``CutSet.from_file(...)``. The cuts are read lazily from disk,
so even very large corpora don't have to fit in memory.

If you run a recipe again with the same ``output_dir``, parts that are already saved are usually loaded
from disk instead of being prepared again. To start over, delete the files.

.. list-table:: Currently supported audio corpora
  :widths: 30 50
  :header-rows: 1

  * - Corpus name
    - Function
  * - ADEPT
    - :func:`lhotse.recipes.prepare_adept`
  * - Aidatatang_200zh
    - :func:`lhotse.recipes.prepare_aidatatang_200zh`
  * - Aishell
    - :func:`lhotse.recipes.prepare_aishell`
  * - Aishell-3
    - :func:`lhotse.recipes.prepare_aishell3`
  * - AISHELL-4
    - :func:`lhotse.recipes.prepare_aishell4`
  * - AliMeeting
    - :func:`lhotse.recipes.prepare_alimeeting`
  * - AMI
    - :func:`lhotse.recipes.prepare_ami`
  * - ASpIRE
    - :func:`lhotse.recipes.prepare_aspire`
  * - ATCOSIM
    - :func:`lhotse.recipes.prepare_atcosim`
  * - AudioMNIST
    - :func:`lhotse.recipes.prepare_audio_mnist`
  * - BABEL
    - :func:`lhotse.recipes.prepare_single_babel_language`
  * - Bengali.AI Speech
    - :func:`lhotse.recipes.prepare_bengaliai_speech`
  * - BUT ReverbDB
    - :func:`lhotse.recipes.prepare_but_reverb_db`
  * - BVCC / VoiceMOS Challenge
    - :func:`lhotse.recipes.bvcc`
  * - CallHome Egyptian
    - :func:`lhotse.recipes.prepare_callhome_egyptian`
  * - CallHome English
    - :func:`lhotse.recipes.prepare_callhome_english`
  * - Chinese Dysarthric Speech Database
    - :func:`lhotse.recipes.prepare_cdsd`
  * - CHiME-6
    - :func:`lhotse.recipes.prepare_chime6`
  * - CMU Arctic
    - :func:`lhotse.recipes.prepare_cmu_arctic`
  * - CMU Indic
    - :func:`lhotse.recipes.prepare_cmu_indic`
  * - CMU Kids
    - :func:`lhotse.recipes.prepare_cmu_kids`
  * - CommonVoice
    - :func:`lhotse.recipes.prepare_commonvoice`
  * - Corpus of Spontaneous Japanese
    - :func:`lhotse.recipes.prepare_csj`
  * - CSLU Kids
    - :func:`lhotse.recipes.prepare_cslu_kids`
  * - DailyTalk
    - :func:`lhotse.recipes.prepare_daily_talk`
  * - DIHARD III
    - :func:`lhotse.recipes.prepare_dihard3`
  * - DiPCo
    - :func:`lhotse.recipes.prepare_dipco`
  * - Earnings'21
    - :func:`lhotse.recipes.prepare_earnings21`
  * - Earnings'22
    - :func:`lhotse.recipes.prepare_earnings22`
  * - EARS
    - :func:`lhotse.recipes.prepare_ears`
  * - The Edinburgh International Accents of English Corpus
    - :func:`lhotse.recipes.prepare_edacc`
  * - English Broadcast News 1997
    - :func:`lhotse.recipes.prepare_broadcast_news`
  * - Fisher English Part 1, 2
    - :func:`lhotse.recipes.prepare_fisher_english`
  * - Fisher Spanish
    - :func:`lhotse.recipes.prepare_fisher_spanish`
  * - FLEURS
    - :func:`lhotse.recipes.prepare_fleurs`
  * - Fluent Speech Commands
    - :func:`lhotse.recipes.slu`
  * - GALE Arabic Broadcast Speech
    - :func:`lhotse.recipes.prepare_gale_arabic`
  * - GALE Mandarin Broadcast Speech
    - :func:`lhotse.recipes.prepare_gale_mandarin`
  * - GigaSpeech
    - :func:`lhotse.recipes.prepare_gigaspeech`
  * - GigaSpeech 2
    - :func:`lhotse.recipes.prepare_gigaspeech2`
  * - GigaST
    - :func:`lhotse.recipes.prepare_gigast`
  * - Heroico
    - :func:`lhotse.recipes.prepare_heroico`
  * - HiFiTTS
    - :func:`lhotse.recipes.prepare_hifitts`
  * - HI-MIA (including HI-MIA-CW)
    - :func:`lhotse.recipes.prepare_himia`
  * - ICMC-ASR
    - :func:`lhotse.recipes.prepare_icmcasr`
  * - ICSI
    - :func:`lhotse.recipes.prepare_icsi`
  * - IWSLT22_Ta
    - :func:`lhotse.recipes.prepare_iwslt22_ta`
  * - KeSpeech
    - :func:`lhotse.recipes.prepare_kespeech`
  * - KsponSpeech
    - :func:`lhotse.recipes.prepare_ksponspeech`
  * - L2 Arctic
    - :func:`lhotse.recipes.prepare_l2_arctic`
  * - LibriCSS
    - :func:`lhotse.recipes.prepare_libricss`
  * - LibriLight
    - :func:`lhotse.recipes.prepare_librilight`
  * - LibriMix
    - :func:`lhotse.recipes.prepare_librimix`
  * - LibriSpeech (including "mini")
    - :func:`lhotse.recipes.prepare_librispeech`
  * - LibriSpeechMix
    - :func:`lhotse.recipes.prepare_librispeechmix`
  * - LibriTTS
    - :func:`lhotse.recipes.prepare_libritts`
  * - LibriTTS-R
    - :func:`lhotse.recipes.prepare_librittsr`
  * - LJ Speech
    - :func:`lhotse.recipes.prepare_ljspeech`
  * - MDCC
    - :func:`lhotse.recipes.prepare_mdcc`
  * - Medical
    - :func:`lhotse.recipes.prepare_medical`
  * - MiniLibriMix
    - :func:`lhotse.recipes.prepare_librimix_mini`
  * - MTEDx
    - :func:`lhotse.recipes.prepare_mtdex`
  * - MobvoiHotWord
    - :func:`lhotse.recipes.prepare_mobvoihotwords`
  * - Multilingual LibriSpeech (MLS)
    - :func:`lhotse.recipes.prepare_mls`
  * - MUSAN
    - :func:`lhotse.recipes.prepare_musan`
  * - MuST-C
    - :func:`lhotse.recipes.prepare_must_c`
  * - NOTSOFAR
    - :func:`lhotse.recipes.prepare_notsofar1`
  * - National Speech Corpus (Singaporean English)
    - :func:`lhotse.recipes.prepare_nsc`
  * - otoSpeech
    - :func:`lhotse.recipes.prepare_oto_speech`
  * - People's Speech
    - :func:`lhotse.recipes.prepare_peoples_speech`
  * - ReazonSpeech
    - :func:`lhotse.recipes.prepare_reazonspeech`
  * - RIRs and Noises Corpus (OpenSLR 28)
    - :func:`lhotse.recipes.prepare_rir_noise`
  * - SBCSAE
    - :func:`lhotse.recipes.prepare_sbcsae`
  * - Seamless Interaction
    - :func:`lhotse.recipes.prepare_seamless_interaction`
  * - Spatial-LibriSpeech
    - :func:`lhotse.recipes.prepare_spatial_librispeech`
  * - Speech Commands
    - :func:`lhotse.recipes.prepare_speechcommands`
  * - SpeechIO
    - :func:`lhotse.recipes.prepare_speechio`
  * - SPGISpeech
    - :func:`lhotse.recipes.prepare_spgispeech`
  * - Switchboard
    - :func:`lhotse.recipes.prepare_switchboard`
  * - TED-LIUM v2
    - :func:`lhotse.recipes.prepare_tedlium2`
  * - TED-LIUM v3
    - :func:`lhotse.recipes.prepare_tedlium`
  * - TIMIT
    - :func:`lhotse.recipes.prepare_timit`
  * - This American Life
    - :func:`lhotse.recipes.prepare_this_american_life`
  * - UWB-ATCC
    - :func:`lhotse.recipes.prepare_uwb_atcc`
  * - VCTK
    - :func:`lhotse.recipes.prepare_vctk`
  * - VoxCeleb
    - :func:`lhotse.recipes.prepare_voxceleb`
  * - VoxConverse
    - :func:`lhotse.recipes.prepare_voxconverse`
  * - VoxPopuli
    - :func:`lhotse.recipes.prepare_voxpopuli`
  * - WenetSpeech
    - :func:`lhotse.recipes.prepare_wenet_speech`
  * - WenetSpeech4TTS
    - :func:`lhotse.recipes.prepare_wenetspeech4tts`
  * - WHAM
    - :func:`lhotse.recipes.prepare_wham`
  * - YesNo
    - :func:`lhotse.recipes.prepare_yesno`
  * - Emilia
    - :func:`lhotse.recipes.prepare_emilia`
  * - Eval2000
    - :func:`lhotse.recipes.prepare_eval2000`
  * - MGB2
    - :func:`lhotse.recipes.prepare_mgb2`
  * - XBMU-AMDO31
    - :func:`lhotse.recipes.xbmu_amdo31`


.. list-table:: Currently supported video corpora
  :widths: 30 50
  :header-rows: 1

  * - Corpus name
    - Function
  * - Grid Audio-Visual Speech Corpus
    - :func:`lhotse.recipes.prepare_grid`

Adding new corpora
------------------

A new recipe should behave exactly like the existing ones. Pick a similar corpus in ``lhotse/recipes``
and use it as a starting point. Then check your recipe against this list:

1. **One file, two functions.** Add ``lhotse/recipes/<corpus>.py`` with ``prepare_<corpus>(corpus_dir, output_dir=None, ...)``.
   If the corpus can be downloaded freely, also add ``download_<corpus>(target_dir, ...)``.
   Export both in ``lhotse/recipes/__init__.py`` and list the corpus in the table above.

2. **Return the standard dictionary.** Return ``{part: {"recordings": ..., "supervisions": ..., "cuts": ...}}``
   as described in `What you get back`_. Use the corpus's own split names, or ``"all"`` if it has none.
   Always return every part, including parts that were loaded from disk instead of prepared again.

3. **Make the cuts with the helper.** Build the cuts with
   :func:`lhotse.recipes.utils.create_cuts_per_recording`. It makes one cut per recording with all of its
   supervisions, and when given an ``output_path`` it writes the cuts one by one and reads them back lazily.
   Don't split recordings into segments; leave that to the user.

4. **Name the files by the rule.** Save each manifest as ``output_dir / f"{corpus}_{manifest}_{part}.jsonl.gz"``.
   Use ``-`` inside the corpus name, never ``_``, because ``_`` separates the three fields.
   Put the files directly in ``output_dir``, not in subdirectories.

5. **Make output_dir optional.** With ``output_dir=None``, write nothing and return the manifests in memory.
   The only exception is a recipe that has to create new audio files; it should say so and raise a clear error.

6. **Keep memory use low for big corpora.** For corpora with millions of recordings, write the recordings,
   supervisions and cuts to disk as you create them, using ``RecordingSet.open_writer``,
   ``SupervisionSet.open_writer`` and ``CutSet.open_writer``, and return lazily opened manifests.

7. **Add a command.** Add a CLI wrapper in ``lhotse/bin/modes/recipes`` so users can run
   ``lhotse prepare <corpus>`` (and ``lhotse download <corpus>``). Copy the pattern of an existing wrapper.

8. **Check imports.** Import optional packages inside the functions that need them, and tell the user how
   to install them if they are missing.

When modelling the data itself, follow these guidelines:

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
