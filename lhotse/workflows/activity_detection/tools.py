from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def resolve_path(path: Optional[PathLike]) -> Optional[Path]:
    if path is None or (isinstance(path, str) and path == ""):
        return None
    return Path(path).expanduser().resolve().absolute()


def assert_output_dir(path: Optional[PathLike], name: str) -> Optional[Path]:
    path = resolve_path(path)
    if path is None:
        return None
    if path.is_file():
        msg = f"Path {name}={path} is a file."
        msg += " Please provide a directory path."
        raise ValueError(msg)
    if not path.exists():
        msg = f"Directory {name}={path} does not exist."
        msg += " Please create {path} first or provide a different path."
        raise ValueError(msg)

    return path


def assert_output_file(path: Optional[PathLike], name: str) -> Optional[Path]:
    path = resolve_path(path)
    if path is None:
        return None
    if path.is_dir():
        msg = f"Path {name}={path} is a directory."
        msg += " Please provide a file path."
        raise ValueError(msg)

    if not path.parent.exists():
        msg = f"Parent directory of {name}={path} does not exist."
        msg += " Please create {path.parent} first or provide a different path."
        raise ValueError(msg)
    return path
