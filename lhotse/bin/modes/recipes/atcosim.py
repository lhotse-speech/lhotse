import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.atcosim import download_atcosim, prepare_atcosim
from lhotse.utils import Pathlike

__all__ = ["atcosim"]


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def atcosim(target_dir: Pathlike):
    """ATCOSIM download."""
    download_atcosim(target_dir)


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--silence-sym", type=str, default="")
@click.option("--breath-sym", type=str, default="")
@click.option("--foreign-sym", type=str, default="<unk>")
@click.option("--partial-sym", type=str, default="<unk>")
@click.option("--unknown-sym", type=str, default="<unk>")
def atcosim(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    silence_sym: str,
    breath_sym: str,
    foreign_sym: str,
    partial_sym: str,
    unknown_sym: str,
):
    """ATCOSIM data preparation."""
    prepare_atcosim(
        corpus_dir,
        output_dir=output_dir,
        silence_sym=silence_sym,
        breath_sym=breath_sym,
        foreign_sym=foreign_sym,
        partial_sym=partial_sym,
        unknown_sym=unknown_sym,
    )
