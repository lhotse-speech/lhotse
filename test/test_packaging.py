from subprocess import CalledProcessError, CompletedProcess

import pytest

import _lhotse_version
from _lhotse_version import discover_lhotse_version, write_lhotse_version


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    monkeypatch.delenv("LHOTSE_PREPARING_RELEASE", raising=False)
    (tmp_path / "VERSION").write_text("2.0.0a6\n", encoding="utf-8")
    (tmp_path / "lhotse").mkdir()
    return tmp_path


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("git_directory", [False, True])
def test_development_version(source_tree, monkeypatch, dirty, git_directory):
    git_path = source_tree / ".git"
    if git_directory:
        git_path.mkdir()
    else:
        git_path.write_text("gitdir: elsewhere\n", encoding="utf-8")

    def run_git(command, **kwargs):
        assert kwargs["cwd"] == source_tree
        output = "abc123\n" if command[1] == "rev-parse" else "changed" if dirty else ""
        return CompletedProcess(command, returncode=0, stdout=output)

    monkeypatch.setattr(_lhotse_version, "run", run_git)
    expected = f"2.0.0a6.dev0+git.abc123.{'dirty' if dirty else 'clean'}"
    assert discover_lhotse_version(source_tree) == expected


def test_release_version(source_tree, monkeypatch):
    monkeypatch.setenv("LHOTSE_PREPARING_RELEASE", "1")
    assert discover_lhotse_version(source_tree) == "2.0.0a6"


@pytest.mark.parametrize("preparing_release", [False, True])
def test_sdist_preserves_embedded_version(source_tree, monkeypatch, preparing_release):
    # An sdist may be unpacked inside another checkout, with a different release flag.
    (source_tree / ".git").mkdir()
    expected = "2.0.0a6.dev0+git.abc123.clean"
    (source_tree / "PKG-INFO").write_text(
        f"Metadata-Version: 2.4\nName: lhotse\nVersion: {expected}\n",
        encoding="utf-8",
    )
    if preparing_release:
        monkeypatch.setenv("LHOTSE_PREPARING_RELEASE", "1")

    def unexpected_git(*args, **kwargs):
        pytest.fail("Rebuilding an sdist must not query Git")

    monkeypatch.setattr(_lhotse_version, "run", unexpected_git)
    assert discover_lhotse_version(source_tree) == expected
    write_lhotse_version(source_tree, version=expected)
    namespace = {}
    exec((source_tree / "lhotse" / "version.py").read_text(), namespace)
    assert namespace["__version__"] == expected


@pytest.mark.parametrize(
    "error", [OSError("git unavailable"), CalledProcessError(1, "git")]
)
def test_version_without_working_git(source_tree, monkeypatch, error):
    (source_tree / ".git").mkdir()

    def fail_git(*args, **kwargs):
        raise error

    monkeypatch.setattr(_lhotse_version, "run", fail_git)
    assert discover_lhotse_version(source_tree) == "2.0.0a6.dev0+unknownsource"


def test_source_archive_does_not_use_parent_git_checkout(source_tree, monkeypatch):
    def unexpected_git(*args, **kwargs):
        pytest.fail("A source archive must not use a parent Git repository")

    monkeypatch.setattr(_lhotse_version, "run", unexpected_git)
    assert discover_lhotse_version(source_tree) == "2.0.0a6.dev0+unknownsource"
