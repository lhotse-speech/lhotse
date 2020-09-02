Representing a corpus
=====================

In Lhotse, we represent the data using YAML manifests.
For most audio corpora, we will need two types of manifests to fully describe them:
a recording manifest and a supervision manifest.

Recording manifest
------------------

The recording manifest describes the recordings in a given corpus.
It only contains information about the recording itself - this manifest does not specify any segmentation information
or supervision such as the transcript or the speaker.
It means that when a recording is a 1 hour long file, it is a single item in this manifest.

When coming from Kaldi, think of it as *wav.scp* on steroids, that also contains *reco2dur*, *reco2num_samples* and
some extra information.

This is a YAML manifest for a corpus with two recordings:

.. code-block:: yaml

    ---
    - id: 'recording-1'
      sampling_rate: 8000
      num_samples: 4000
      duration_seconds: 0.5
      sources:
        - type: file
          channel_ids: [0]
          source: 'test/fixtures/mono_c0.wav'
        - type: file
          channel_ids: [1]
          source: 'test/fixtures/mono_c1.wav'
    - id: 'recording-2'
      sampling_rate: 8000
      num_samples: 8000
      duration_seconds: 1.0
      sources:
        - type: file
          channel_ids: [0, 1]
          source: 'test/fixtures/stereo.wav'

Each recording is described by:

- a unique id,
- its sampling rate,
- the number of samples,
- the duration in seconds,
- a list of audio sources.

Audio source is a useful abstraction for cases when the user has an audio format not supported by the library,
or wants to use shell tools such as SoX to perform some additional preprocessing.
An audio source has the following properties:

- type: either `file` or `command`
- channel_ids: a list of integer identifiers for each channel in the recording
- source: in case of a `file`, it's a path; in case of a `command`, its a shell command that will be expected to write a WAVE file to stdout.

Python
******

In Python, the recording manifest is represented by classes :class:`RecordingSet`, :class:`Recording`, and :class:`AudioSource`.
Example usage:

.. code-block:: python

    recordings = RecordingSet.from_yaml('audio.yml')
    for recording in recordings:
        if recording.duration >= 1.0:
            samples = recording.load_audio(
                channels=0,
                offset_seconds=0.3,
                duration_seconds=0.5
            )
            # Further sample processing


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
      channel_id: 0
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

Currently supported corpora:

- AMI :func:`lhotse.recipes.prepare_ami`
- English Broadcast News 1997 :func:`lhotse.recipes.prepare_broadcast_news`
- Mini LibriMix :func:`lhotse.recipes.prepare_librimix`
- Mini LibriSpeech :func:`lhotse.recipes.prepare_mini_librispeech`
- Switchboard :func:`lhotse.recipes.prepare_switchboard`
