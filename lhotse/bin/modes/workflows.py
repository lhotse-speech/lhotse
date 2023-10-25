# pylint: disable=C0415,R0913,R0914
import sys
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.bin.modes.cli_base import cli
from lhotse.parallel import ParallelExecutor
from lhotse.serialization import load_manifest_lazy_or_eager
from lhotse.utils import PythonLiteralOption, exactly_one_not_null


@cli.group()
def workflows():
    """Workflows using corpus creation tools."""
    pass


@workflows.command()
@click.argument("out_cuts", type=click.Path(allow_dash=True))
@click.option(
    "-m",
    "--recordings-manifest",
    type=click.Path(exists=True, dir_okay=False, allow_dash=True),
    help="Path to an existing recording manifest.",
)
@click.option(
    "-r",
    "--recordings-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Directory with recordings. We will create a RecordingSet for it automatically.",
)
@click.option(
    "-c",
    "--cuts-manifest",
    type=click.Path(exists=True, dir_okay=False, allow_dash=True),
    help="Path to an existing cuts manifest.",
)
@click.option(
    "-e",
    "--extension",
    default="wav",
    help="Audio file extension to search for. Used with RECORDINGS_DIR.",
)
@click.option(
    "-n",
    "--model-name",
    default="base",
    help="One of Whisper variants (base, medium, large, etc.)",
)
@click.option(
    "-l",
    "--language",
    help="Language spoken in the audio. Inferred by default.",
)
@click.option(
    "-d", "--device", default="cpu", help="Device on which to run the inference."
)
@click.option("-j", "--jobs", default=1, help="Number of jobs for audio scanning.")
@click.option(
    "--force-nonoverlapping/--keep-overlapping",
    default=False,
    help="If True, the Whisper segment time-stamps will be processed to make sure they are non-overlapping.",
)
def annotate_with_whisper(
    out_cuts: str,
    recordings_manifest: Optional[str],
    recordings_dir: Optional[str],
    cuts_manifest: Optional[str],
    extension: str,
    model_name: str,
    language: Optional[str],
    device: str,
    jobs: int,
    force_nonoverlapping: bool,
):
    """
    Use OpenAI Whisper model to annotate either RECORDINGS_MANIFEST, RECORDINGS_DIR, or CUTS_MANIFEST.
    It will perform automatic segmentation, transcription, and language identification.

    RECORDINGS_MANIFEST, RECORDINGS_DIR, and CUTS_MANIFEST are mutually exclusive. If CUTS_MANIFEST
    is provided, its supervisions will be overwritten with the results of the inference.

    Note: this is an experimental feature of Lhotse, and is not guaranteed to yield
    high quality of data.
    """
    from lhotse import annotate_with_whisper as annotate_with_whisper_

    assert exactly_one_not_null(recordings_manifest, recordings_dir, cuts_manifest), (
        "Options RECORDINGS_MANIFEST, RECORDINGS_DIR, and CUTS_MANIFEST are mutually exclusive "
        "and at least one is required."
    )

    if recordings_manifest is not None:
        manifest = RecordingSet.from_file(recordings_manifest)
    elif recordings_dir is not None:
        manifest = RecordingSet.from_dir(
            recordings_dir, pattern=f"*.{extension}", num_jobs=jobs
        )
    else:
        manifest = CutSet.from_file(cuts_manifest).to_eager()

    with CutSet.open_writer(out_cuts) as writer:
        for cut in tqdm(
            annotate_with_whisper_(
                manifest,
                language=language,
                model_name=model_name,
                device=device,
                force_nonoverlapping=force_nonoverlapping,
            ),
            total=len(manifest),
            desc="Annotating with Whisper",
        ):
            writer.write(cut, flush=True)


@workflows.command()
@click.argument(
    "in_cuts", type=click.Path(exists=True, dir_okay=False, allow_dash=True)
)
@click.argument("out_cuts", type=click.Path(allow_dash=True))
@click.option(
    "-n",
    "--bundle-name",
    default="WAV2VEC2_ASR_BASE_960H",
    help="One of torchaudio pretrained 'bundle' variants (see: https://pytorch.org/audio/stable/pipelines.html)",
)
@click.option(
    "-d", "--device", default="cpu", help="Device on which to run the inference."
)
@click.option(
    "--normalize-text/--dont-normalize-text",
    default=True,
    help="By default, we'll try to normalize the text by making it uppercase and discarding symbols "
    "outside of model's character level vocabulary. If this causes issues, "
    "turn the option off and normalize the text yourself.",
)
@click.option(
    "-j",
    "--num-jobs",
    default=1,
    help="Number of parallel jobs to run.",
)
@click.option(
    "--check-language/--dont-check-language",
    default=True,
    help="If `False`, warnings about non-existent language tags in supervisions will be suppressed.",
)
def align_with_torchaudio(
    in_cuts: str,
    out_cuts: str,
    bundle_name: str,
    device: str,
    normalize_text: bool,
    num_jobs: int = 1,
    check_language: bool = True,
):
    """
    Use a pretrained ASR model from torchaudio to force align IN_CUTS (a Lhotse CutSet)
    and write the results to OUT_CUTS.
    It will attach word-level alignment information (start, end, and score) to the
    supervisions in each cut.

    This is based on a tutorial from torchaudio:
    https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

    In order to use a multilingual alignment model, use `--bundle_name MMS_FA`.
    (based on the multilingual tutorial: https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html)

    Note: this is an experimental feature of Lhotse, and is not guaranteed to yield
    high quality of data.
    """
    from lhotse import align_with_torchaudio as align_with_torchaudio_

    cuts = load_manifest_lazy_or_eager(in_cuts)

    with CutSet.open_writer(out_cuts) as writer:
        for cut in tqdm(
            align_with_torchaudio_(
                cuts,
                bundle_name=bundle_name,
                device=device,
                normalize_text=normalize_text,
                num_jobs=num_jobs,
                verbose=False,
                check_language=check_language,
            ),
            total=len(cuts),
            desc="Aligning",
        ):
            writer.write(cut, flush=True)


@workflows.command()
@click.argument(
    "in_cuts", type=click.Path(exists=True, dir_okay=False, allow_dash=True)
)
@click.argument("out_cuts", type=click.Path(allow_dash=True))
@click.option(
    "--method",
    type=click.Choice(["independent", "conversational"]),
    default="independent",
    help="The simulation method to use: "
    "independent - each speaker is simulated independently, "
    "conversational - the speakers are simulated as a group, "
    "using overall silence/overlap statistics.",
)
# Options used with the "independent" method
@click.option(
    "--loc",
    type=float,
    default=0.0,
    help="The minimum silence duration between two consecutive utterances from the same speaker.",
    show_default=True,
)
@click.option(
    "--scale",
    type=float,
    default=2.0,
    help="The scale parameter of the exponential distribution used to sample the silence "
    "duration between two consecutive utterances from a speaker.",
    show_default=True,
)
# Options used with the "conversational" method


@click.option(
    "--same-spk-pause",
    type=float,
    default=1.0,
    help="The mean pause duration between utterances of the same speaker",
    show_default=True,
)
@click.option(
    "--diff-spk-pause",
    type=float,
    default=1.0,
    help="The mean pause duration between utterances of different speakers",
    show_default=True,
)
@click.option(
    "--diff-spk-overlap",
    type=float,
    default=2.0,
    help="The mean overlap duration between utterances of different speakers",
    show_default=True,
)
@click.option(
    "--prob-diff-spk-overlap",
    type=float,
    default=0.5,
    help="The probability of overlap between utterances of different speakers",
    show_default=True,
)
# Common options
@click.option(
    "--fit-to-supervisions",
    "-f",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a supervision set to learn the distributions for simulation.",
)
@click.option(
    "--reverberate/--dont-reverberate",
    default=False,
    help="If True, the simulated meetings will be reverberated.",
)
@click.option(
    "--rir-recordings",
    "--rir",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="Path to a recording set containing RIRs. If provided, the simulated meetings will be "
    "reverberated using the RIRs from this set. A directory containing recording sets can also "
    "be provided, in which case each meeting will use a recording set sampled from this directory.",
)
@click.option(
    "--num-meetings",
    "-n",
    type=int,
    default=None,
    help="Number of meetings to simulate. Either this of `num_repeats` must be provided.",
)
@click.option(
    "--num-repeats",
    "-r",
    type=int,
    default=1,
    help="Number of times to repeat each input cut. The resulting cuts will be used as a finite "
    "set of utterances to use for simulation. Either this of `num_meetings` must be provided.",
)
@click.option(
    "--num-speakers-per-meeting",
    "-s",
    cls=PythonLiteralOption,
    default="2",
    help="Number of speakers per meeting. One or more integers can be provided (comma-separated). "
    "In this case, the number of speakers will be sampled uniformly from the provided list, "
    "or using the distribution provided in `speaker-count-probs`.",
)
@click.option(
    "--speaker-count-probs",
    "-p",
    cls=PythonLiteralOption,
    default=None,
    help="A list of probabilities for each speaker count. The length of the list must be "
    "equal to the number of elements in `num-speakers-per-meeting`.",
)
@click.option(
    "--max-duration-per-speaker",
    "-d",
    type=float,
    default=20.0,
    help="Maximum duration of a single speaker in a meeting.",
)
@click.option(
    "--max-utterances-per-speaker",
    "-u",
    type=int,
    default=5,
    help="Maximum number of utterances per speaker in a meeting.",
)
@click.option(
    "--allow-3fold-overlap/--no-3fold-overlap",
    default=False,
    help="If True, the simulated meetings will allow more than 2 speakers to overlap. This "
    "is only relevant for the `conversational` method.",
)
@click.option(
    "--seed",
    type=int,
    default=1234,
    help="Random seed for reproducibility.",
)
@click.option(
    "--num-jobs",
    "-j",
    type=int,
    default=1,
    help="Number of parallel jobs to run.",
)
def simulate_meetings(
    in_cuts: str,
    out_cuts: str,
    method: str,
    loc: float,
    scale: float,
    same_spk_pause: float,
    diff_spk_pause: float,
    diff_spk_overlap: float,
    prob_diff_spk_overlap: float,
    fit_to_supervisions: Optional[str],
    reverberate: bool,
    rir_recordings: Optional[str],
    **kwargs,
):
    """
    Simulate meeting-style mixtures using a provided CutSet containing single-channel
    cuts. Different simulation techniques can be selected using the `--method` option.
    Currently, the following methods are supported:

    - independent: each speaker is simulated independently, using the provided cuts as a finite
        set of utterances.

    - conversational: the speakers are simulated as a group, using overall silence/overlap
        statistics.

    The number of speakers per meeting is sampled uniformly from the range provided in
    `--num-speakers-per-meeting`.

    The number of meetings to simulate is controlled by either `--num-meetings` or
    `--num-repeats`. If the former is provided, the same number of meetings will be
    simulated. If the latter is provided, the provided cuts will be repeated `num_repeats`
    times, and the resulting cuts will be used as a finite set of utterances to use for simulation.

    The simulated meetings can be optionally reverberated using the RIRs from a provided
    recording set. If no RIRs are provided, we will use a fast random approximation technique
    to simulate the reverberation. The RIRs can be provided as a single recording set, or as
    a directory containing multiple recording sets. In the latter case, the RIRs will be sampled
    from the provided directory.

    """
    if method == "independent":
        from lhotse.workflows.meeting_simulation import (
            SpeakerIndependentMeetingSimulator,
        )

        simulator = SpeakerIndependentMeetingSimulator(loc=loc, scale=scale)

        # Remove options that are not relevant for the independent method.
        kwargs.pop("allow_3fold_overlap")
    elif method == "conversational":
        from lhotse.workflows.meeting_simulation import ConversationalMeetingSimulator

        simulator = ConversationalMeetingSimulator(
            same_spk_pause=same_spk_pause,
            diff_spk_pause=diff_spk_pause,
            diff_spk_overlap=diff_spk_overlap,
            prob_diff_spk_overlap=prob_diff_spk_overlap,
        )
    else:
        raise ValueError(f"Unknown meeting simulation method: {method}")

    if fit_to_supervisions is not None:
        print("Fitting the meeting simulator to the provided supervisions...")
        sups = load_manifest_lazy_or_eager(
            fit_to_supervisions, manifest_cls=SupervisionSet
        )
        simulator.fit(sups)

    cuts = load_manifest_lazy_or_eager(in_cuts)
    print("Simulating meetings...")
    mixed_cuts = simulator.simulate(
        cuts,
        **kwargs,
    )

    if reverberate:
        print("Reverberating the simulated meetings...")
        if rir_recordings:
            if rir_recordings.is_file():
                rirs = [
                    load_manifest_lazy_or_eager(
                        rir_recordings, manifest_cls=RecordingSet
                    )
                ]
            else:
                rirs = [
                    load_manifest_lazy_or_eager(p)
                    for p in rir_recordings.glob("*.jsonl.gz")
                ]
            mixed_cuts = simulator.reverberate(mixed_cuts, *rirs)
        else:
            mixed_cuts = simulator.reverberate(mixed_cuts)

    print("Saving the simulated meetings...")
    mixed_cuts.to_file(out_cuts)


@workflows.command()
@click.option(
    "-r",
    "--recordings-manifest",
    type=click.Path(exists=True, dir_okay=False, allow_dash=True),
    help="Path to an existing recording manifest.",
)
@click.option(
    "-o",
    "--output-supervisions-manifest",
    type=click.Path(exists=False, dir_okay=True, allow_dash=True),
    help="Path to the output supervisions manifest or a directory where it will be saved.",
)
@click.option(
    "-m",
    "--model-name",
    default="silero-vad-16k",
    help="One of activity detector: silero_vad_16k, silero_vad_8k.",
)
@click.option(
    "-d",
    "--device",
    default="cpu",
    help="Device on which to run the inference.",
)
@click.option(
    "-j",
    "--jobs",
    default=1,
    help="Number of jobs for audio scanning.",
)
@click.option(
    "--force_download",
    is_flag=True,
    help="Forced cache clearing and model downloading",
)
def activity_detection(
    recordings_manifest: str,
    output_supervisions_manifest: Optional[str],
    model_name: str,
    device: str,
    jobs: int,
    force_download: bool,
):
    """
    Use activity detection methods (e.g., Silero VAD) to detect and annotate
    the segmentation of Lhotse RecordingSets and save the results in the
    SupervisionSet manifest. The output manifest will be saved in the path
    specified by OUTPUT_SUPERVISIONS_MANIFEST. If OUTPUT_SUPERVISIONS_MANIFEST
    is not provided, the output manifest will be saved in the same directory
    as RECORDINGS_MANIFEST.

    Note: this is an experimental feature and it does not guarantee
    high-quality performance and data annotation.
    """

    import warnings

    from lhotse.workflows.activity_detection import SileroVAD8k, SileroVAD16k

    warnings.filterwarnings("ignore")

    detectors = {
        "silero_vad_8k": SileroVAD8k,
        "silero_vad_16k": SileroVAD16k,
    }
    detector_kls = detectors.get(model_name)

    if detector_kls is None:
        print(
            f"Unknown activity detector: {model_name}. "
            f"Supported detectors: {list(detectors)}"
        )
        sys.exit()

    # prepare paths and input data
    recs_path = Path(recordings_manifest).expanduser().absolute()
    if not recs_path.exists() or not recs_path.is_file():
        print(f"Recordings manifest not found: {str(recs_path)}")
        sys.exit()

    sups_path = (
        recs_path.parent
        if output_supervisions_manifest is None
        else Path(output_supervisions_manifest).expanduser().absolute()
    )
    if sups_path.is_dir():
        name = Path(recs_path).name
        for ext in [".gz", ".jsonl", ".json", ".yaml"]:
            if name.endswith(ext):  # .remove_suffix(ext) in Python 3.9
                name = name[: -len(ext)]
        name += f"_supervisions_{model_name}.jsonl.gz"
        sups_path = sups_path / name

    if not sups_path.parent.exists():
        print(f"Parent directory for output manifest does not exist: {str(sups_path)}")
        sys.exit()

    print(f"Loading recordings from {str(recordings_manifest)}...")
    recordings = RecordingSet.from_file(str(recordings_manifest))

    # run activity detection
    if force_download:  # pragma: no cover
        print("Removing model state from cache...")
        detector_kls.force_download()
    else:
        print("Checking model state in cache...")
        detector_kls("cpu")

    print(f"Making activity detection processor for {model_name!r}...")
    detector_init_fn = partial(detector_kls, device=device)
    processor = ParallelExecutor(
        init_fn=detector_init_fn,
        num_jobs=jobs,
        verbose=True,
        description="Running VAD",
    )
    print(f"Running activity detection using {model_name!r}...")
    supervisions = SupervisionSet.from_segments(
        chain.from_iterable(processor(recordings))
    )

    print(f"Saving {model_name!r} results ...")
    supervisions.to_file(str(sups_path))

    print("Results saved to:", str(sups_path), sep="\n")
