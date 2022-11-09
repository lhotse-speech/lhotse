from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from lhotse import CutSet, RecordingSet
from lhotse.bin.modes.cli_base import cli
from lhotse.serialization import load_manifest_lazy_or_eager
from lhotse.utils import exactly_one_not_null


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
def align_with_torchaudio(
    in_cuts: str,
    out_cuts: str,
    bundle_name: str,
    device: str,
    normalize_text: bool,
):
    """
    Use a pretrained ASR model from torchaudio to force align IN_CUTS (a Lhotse CutSet)
    and write the results to OUT_CUTS.
    It will attach word-level alignment information (start, end, and score) to the
    supervisions in each cut.

    This is based on a tutorial from torchaudio:
    https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

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
    "-d", "--device", default="cpu", help="Device on which to run the inference."
)
@click.option(
    "--num-speakers",
    type=int,
    default=None,
    help="Number of clusters to use for speaker diarization. Will use threshold if not provided.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Threshold for speaker diarization. Will use num-speakers if not provided.",
)
def diarize_segments_with_speechbrain(
    in_cuts: str,
    out_cuts: str,
    device: str = "cpu",
    num_speakers: Optional[int] = None,
    threshold: Optional[float] = None,
):
    """
    This workflow uses SpeechBrain's pretrained speaker embedding model to compute speaker embeddings
    for each cut in the CutSet. The cuts for the same recording are then clustered using
    agglomerative hierarchical clustering, and the resulting cluster indices are used to create new cuts
    with the speaker labels.

    Please refer to https://huggingface.co/speechbrain/spkrec-xvect-voxceleb for more details
    about the speaker embedding extractor.
    """
    from lhotse.workflows import diarize_segments_with_speechbrain

    assert exactly_one_not_null(
        num_speakers, threshold
    ), "Exactly one of --num-speakers and --threshold must be provided."

    cuts = load_manifest_lazy_or_eager(in_cuts)
    cuts_with_spk_id = diarize_segments_with_speechbrain(
        cuts, device=device, num_speakers=num_speakers, threshold=threshold
    )
    cuts_with_spk_id.to_file(out_cuts)
