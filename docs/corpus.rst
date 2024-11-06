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
  * - LibriSpeech (including "mini")
    - :func:`lhotse.recipes.prepare_librispeech`
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
    - :func:`lhotse.recipes.prepare_librimix`
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
  * - National Speech Corpus (Singaporean English)
    - :func:`lhotse.recipes.prepare_nsc`
  * - People's Speech
    - :func:`lhotse.recipes.prepare_peoples_speech`
  * - ReazonSpeech
    - :func:`lhotse.recipes.prepare_reazonspeech`
  * - RIRs and Noises Corpus (OpenSLR 28)
    - :func:`lhotse.recipes.prepare_rir_noise`
  * - SBCSAE
    - :func:`lhotse.recipes.prepare_sbcsae`
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
    **Manifest naming convention.** The default naming convention is ``<corpus-name>_<manifest-type>_<split>.jsonl.gz``,
    i.e., we save the manifests in a compressed JSONL file. Here, ``<manifest-type>`` can be ``recordings``,
    ``supervisions``, etc., and ``<split>`` can be ``train``, ``dev``, ``test``, etc. In case the corpus
    has no such split defined, we can use ``all`` as default. Other information, e.g., mic type, language, etc. may
    be included in the ``<corpus-name>``. Some examples are: ``cmu-indic_recordings_all.jsonl.gz``,
    ``ami-ihm_supervisions_dev.jsonl.gz``, ``mtedx-english_recordings_train.jsonl.gz``.

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
