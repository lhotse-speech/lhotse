import click

from lhotse.bin.modes import obtain, prepare
from lhotse.recipes.heroico import download_and_untar, prepare_heroico_answers, prepare_heroico_recitations, prepare_usma
from lhotse.utils import Pathlike

__all__ = ['heroico']


@prepare.command(context_settings=dict(show_default=True))
@click.argument('speech_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('transcript_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def heroico_answers(
        speech_dir: Pathlike,
        transcript_dir: Pathlike,
        output_dir: Pathlike
):
    """heroico Answers ASR data preparation."""
    prepare_heroico_answers(speech_dir, transcript_dir, output_dir)


@prepare.command(context_settings=dict(show_default=True))
@click.argument('speech_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('transcript_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def heroico_recitations(
        speech_dir: Pathlike,
        transcript_dir: Pathlike,
        output_dir: Pathlike
):
    """heroico Answers ASR data preparation."""
    prepare_heroico_recitations(speech_dir, transcript_dir, output_dir)


@prepare.command(context_settings=dict(show_default=True))
@click.argument('speech_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('transcript_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def usma(
        speech_dir: Pathlike,
        transcript_dir: Pathlike,
        output_dir: Pathlike
):
    """usma ASR data preparation."""
    prepare_usma(speech_dir, transcript_dir, output_dir)


@obtain.command(context_settings=dict(show_default=True))
@click.argument('target_dir', type=click.Path())
def heroico(
        target_dir: Pathlike
):
    """heroico download."""
    download_and_untar(target_dir)
