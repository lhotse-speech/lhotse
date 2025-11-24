import logging
import os
import platform
import sys
from pathlib import Path


def default_tools_cachedir(force_mkdir: bool = False) -> Path:
    d = Path.home() / ".lhotse/tools"
    try:
        d.mkdir(exist_ok=True, parents=True)
    except OSError:
        if force_mkdir:
            raise
        else:
            logging.warning(
                f"We couldn't create lhotse utilities directory: {d} (not enough space/no permissions?)"
            )
    return d


def add_tools_to_path():
    sph2pipe_path = str(default_tools_cachedir() / "sph2pipe-2.5")
    sys.path.append(sph2pipe_path)
    os.environ["PATH"] += os.pathsep + sph2pipe_path  # platform-agnostic


def add_macos_homebrew_lib_paths():
    if platform.system() == "Darwin":  # macOS
        HOMEBREW_LIB_PATHS = ["/opt/homebrew/lib", "/usr/local/lib"]
        for path in HOMEBREW_LIB_PATHS:
            dyld_library_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            if os.path.exists(path) and path not in dyld_library_path.split(":"):
                os.environ["DYLD_LIBRARY_PATH"] = dyld_library_path + f":{path}"
