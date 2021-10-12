import os
import sys
from pathlib import Path


def default_tools_cachedir() -> Path:
    d = Path.home() / ".lhotse/tools"
    d.mkdir(exist_ok=True, parents=True)
    return d


def add_tools_to_path():
    sph2pipe_path = str(default_tools_cachedir() / "sph2pipe-2.5")
    sys.path.append(sph2pipe_path)
    os.environ["PATH"] += os.pathsep + sph2pipe_path  # platform-agnostic
