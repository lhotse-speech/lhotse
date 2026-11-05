"""Build-time version discovery without importing Lhotse or its dependencies."""

import os
from email.parser import Parser
from pathlib import Path
from subprocess import DEVNULL, PIPE, CalledProcessError, run


def discover_lhotse_version(project_root: Path) -> str:
    """Preserve an sdist's version, or identify a release/development checkout."""
    metadata_path = project_root / "PKG-INFO"
    if metadata_path.is_file():
        metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
        version = metadata["Version"]
        if version is None:
            raise ValueError(f"Missing Version in {metadata_path}")
        return version

    version = (project_root / "VERSION").read_text(encoding="utf-8").strip()
    if os.environ.get("LHOTSE_PREPARING_RELEASE"):
        return version

    if (project_root / ".git").exists():
        try:
            commit = run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=project_root,
                check=True,
                stdout=PIPE,
                stderr=DEVNULL,
                text=True,
            ).stdout.strip()
            dirty = run(
                ["git", "diff", "--shortstat"],
                cwd=project_root,
                check=True,
                stdout=PIPE,
                stderr=DEVNULL,
                text=True,
            ).stdout.strip()
            state = "dirty" if dirty else "clean"
            return f"{version}.dev0+git.{commit}.{state}"
        except (OSError, CalledProcessError):
            pass
    return f"{version}.dev0+unknownsource"


def write_lhotse_version(project_root: Path, version: str) -> None:
    """Embed the resolved version in wheels and editable installations."""
    version_path = project_root / "lhotse" / "version.py"
    content = f'__version__ = "{version}"\n'
    if (
        not version_path.is_file()
        or version_path.read_text(encoding="utf-8") != content
    ):
        version_path.write_text(content, encoding="utf-8")
