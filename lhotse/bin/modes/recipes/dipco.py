import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.dipco import download_dipco, prepare_dipco
from lhotse.utils import Pathlike

__all__ = ["dipco"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--mic",
    type=click.Choice(["ihm", "mdm"], case_sensitive=False),
    default="mdm",
    help="DiPCo microphone setting.",
)
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "upper", "kaldi"], case_sensitive=False),
    default="kaldi",
    help="Text normalization method.",
    show_default=True,
)
@click.option(
    "--use-chime7-offset",
    is_flag=True,
    default=False,
    help="If True, offset session IDs (from CHiME-7 challenge).",
)
def dipco(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    mic: str,
    normalize_text: str,
    use_chime7_offset: bool,
):
    """DiPCo data preparation."""
    prepare_dipco(
        corpus_dir,
        output_dir=output_dir,
        mic=mic,
        normalize_text=normalize_text,
        use_chime7_offset=use_chime7_offset,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    type=bool,
    default=False,
    help="If True, download even if file is present.",
)
def dipco(
    target_dir: Pathlike,
    force_download: bool,
):
    """DiPCo download."""
    download_dipco(
        target_dir,
        force_download=force_download,
    )
