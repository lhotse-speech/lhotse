import logging
import subprocess
import tarfile
from pathlib import Path

from lhotse.tools.env import default_tools_cachedir
from lhotse.utils import Pathlike, urlretrieve_progress

SPH2PIPE_URL = "https://github.com/burrmill/sph2pipe/archive/2.5.tar.gz"


def install_sph2pipe(
    where: Pathlike = default_tools_cachedir(),
    download_from: str = SPH2PIPE_URL,
    force: bool = False,
) -> None:
    """
    Install the sph2pipe program to handle sphere (.sph) audio files with
    "shorten" codec compression (needed for older LDC data).

    It downloads an archive and then decompresses and compiles the contents.
    """
    where = Path(where)
    # Download
    download_and_untar_sph2pipe(where, url=download_from, force_download=force)
    # Compile
    subprocess.run([f'make -C {where / "sph2pipe-2.5"}'], shell=True, check=True)
    logging.info("Finished installing sph2pipe.")


def download_and_untar_sph2pipe(
    target_dir: Pathlike,
    url: str,
    force_download: bool = False,
) -> Path:
    target_dir = Path(target_dir)
    sph2pipe_dir = target_dir / "sph2pipe-2.5"
    if (sph2pipe_dir / "Makefile").is_file() and not force_download:
        return sph2pipe_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_name = "sph2pipe-2.5.tar.gz"
    tar_path = target_dir / tar_name
    if force_download or not tar_path.is_file():
        urlretrieve_progress(url, filename=tar_path, desc=f"Downloading {tar_name}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    return sph2pipe_dir
