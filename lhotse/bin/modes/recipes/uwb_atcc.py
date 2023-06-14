import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.uwb_atcc import download_uwb_atcc, prepare_uwb_atcc
from lhotse.utils import Pathlike

__all__ = ["uwb_atcc"]


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def uwb_atcc(target_dir: Pathlike):
    """UWB-ATCC download."""
    download_uwb_atcc(target_dir)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--silence-sym", type=str, default="")
@click.option("--breath-sym", type=str, default="")
@click.option("--noise-sym", type=str, default="")
@click.option("--foreign-sym", type=str, default="<unk>")
@click.option("--partial-sym", type=str, default="<unk>")
@click.option("--unintelligble-sym", type=str, default="<unk>")
@click.option("--unknown-sym", type=str, default="<unk>")
def uwb_atcc(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    silence_sym: str,
    breath_sym: str,
    noise_sym: str,
    foreign_sym: str,
    partial_sym: str,
    unintelligble_sym: str,
    unknown_sym: str,
):
    """UWB-ATCC data preparation."""
    prepare_uwb_atcc(
        corpus_dir,
        output_dir=output_dir,
        silence_sym=silence_sym,
        breath_sym=breath_sym,
        noise_sym=noise_sym,
        foreign_sym=foreign_sym,
        partial_sym=partial_sym,
        unintelligble_sym=unintelligble_sym,
        unknown_sym=unknown_sym,
    )
