"""Containment checks around stdlib tar extraction, including older Python versions."""

import errno
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from tarfile import TarFile
from typing import Iterator, List, Set, Union


def _resolve_path(path: Path) -> Path:
    """Resolve links, tolerating missing paths but no other filesystem errors."""
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        pass

    # Unlike resolve(strict=False) on older Python versions, this must not
    # suppress errors such as ELOOP, EACCES or ENAMETOOLONG.
    path = Path.cwd() / path
    if not path.is_absolute():
        raise ValueError(f"Cannot resolve archive path: {path}")
    resolved = Path(path.anchor)
    pending: List[Union[str, Path]] = list(reversed(path.parts[1:]))
    resolving: Set[Path] = set()
    while pending:
        part = pending.pop()
        if isinstance(part, Path):
            resolving.remove(part)
            continue
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        try:
            target = Path(os.readlink(candidate))
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.EINVAL):
                raise
            resolved = candidate
            continue
        if candidate in resolving:
            raise ValueError(f"Symlink loop in archive path: {candidate}")
        resolving.add(candidate)
        pending.append(candidate)
        if target.is_absolute():
            resolved = Path(target.anchor)
            pending.extend(reversed(target.parts[1:]))
        elif target.drive:
            raise ValueError(f"Cannot resolve archive link: {target}")
        else:
            pending.extend(reversed(target.parts))
    return resolved


@contextmanager
def guarded_tar_extraction(tar: TarFile, path: Path) -> Iterator[None]:
    """Guard actual writes without changing tarfile's filters or metadata policy."""
    root = _resolve_path(path)
    symlinks: Set[Path] = set()

    def check_path(target: Path, allow_root: bool = True) -> None:
        try:
            resolved = _resolve_path(target)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"Cannot safely resolve archive path: {target}") from error
        if root not in resolved.parents and not (allow_root and root == resolved):
            # ValueError is deliberately outside tarfile's suppressible errors:
            # containment failures must abort even when tar.errorlevel == 0.
            raise ValueError(f"Attempted Path Traversal in Tar File: {target}")

    # Retain the existing whole-archive path validation, including when only a
    # subset of members is selected. Later checks also see newly created links.
    for member in tar.getmembers():
        check_path(path / member.name, allow_root=member.isdir())

    def guard(method, extracting: bool = False):
        @wraps(method)
        def checked(member, targetpath, *args, **kwargs):
            target = Path(targetpath)
            check_path(target)
            if extracting:
                # makedirs() visits lexical parents, even when the final path
                # resolves inside root (e.g. ../outside/../destination/file).
                # Only the caller's destination itself may need outside parents.
                parent = target.parent
                while (
                    target not in (path, root)
                    and parent not in (path, root)
                    and parent != parent.parent
                ):
                    try:
                        parent.stat()
                    except FileNotFoundError:
                        check_path(parent)
                        parent = parent.parent
                    except OSError as error:
                        raise ValueError(
                            f"Cannot safely resolve archive path: {parent}"
                        ) from error
                    else:
                        break
                if member.issym():
                    check_path(target.parent / member.linkname)
                    symlinks.add(target)
                elif member.islnk():
                    check_path(
                        Path(getattr(member, "_link_target", path / member.linkname))
                    )
            return method(member, targetpath, *args, **kwargs)

        return checked

    methods = ("_extract_member", "chown", "chmod", "utime")
    overrides = {name: tar.__dict__[name] for name in methods if name in tar.__dict__}
    originals = {name: getattr(tar, name) for name in methods}
    try:
        # Guard the instance so recursive link fallback uses the same checks.
        # TarFile already requires exclusive access during extraction.
        for name, method in originals.items():
            setattr(tar, name, guard(method, extracting=name == "_extract_member"))
        yield
        # A later link can change the meaning of an earlier dangling target.
        for target in symlinks:
            check_path(target)
    finally:
        for name in methods:
            if name in overrides:
                setattr(tar, name, overrides[name])
            else:
                delattr(tar, name)
