import click

from .cli_base import cli
from ...tools.env import default_tools_cachedir
from ...tools.sph2pipe import SPH2PIPE_URL


@cli.command(context_settings=dict(show_default=True))
@click.option(
    "--install-dir",
    type=click.Path(),
    default=default_tools_cachedir(),
    help="Directory where sph2pipe will be downloaded and installed.",
)
@click.option(
    "--url", default=SPH2PIPE_URL, help="URL from which to download sph2pipe."
)
def install_sph2pipe(install_dir: str, url: str):
    """
    Install the sph2pipe program to handle sphere (.sph) audio files with
    "shorten" codec compression (needed for older LDC data).

    It downloads an archive and then decompresses and compiles the contents.
    """
    from lhotse.tools.sph2pipe import install_sph2pipe

    install_sph2pipe(where=install_dir, download_from=url)
