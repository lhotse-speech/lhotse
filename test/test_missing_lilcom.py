import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from lhotse.features.io import default_features_storage_backend_name
from lhotse.utils import is_module_available

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_without_lilcom(tmp_path: Path, script: str) -> subprocess.CompletedProcess:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        textwrap.dedent(
            """
            import builtins
            import importlib.util

            _real_import = builtins.__import__


            def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "lilcom":
                    raise ModuleNotFoundError("No module named 'lilcom'")
                return _real_import(name, globals, locals, fromlist, level)


            builtins.__import__ = _blocked_import

            _real_find_spec = importlib.util.find_spec


            def _blocked_find_spec(name, package=None):
                if name == "lilcom":
                    return None
                return _real_find_spec(name, package)


            importlib.util.find_spec = _blocked_find_spec
            """
        )
    )

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(tmp_path)
        if not pythonpath
        else os.pathsep.join([str(tmp_path), pythonpath])
    )

    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_lhotse_imports_and_loads_numpy_features_without_lilcom(tmp_path):
    result = run_without_lilcom(
        tmp_path,
        textwrap.dedent(
            """
            import lhotse

            assert "lilcom_chunky" not in lhotse.available_storage_backends()
            cut = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json")[0]
            feats = cut.load_features()
            assert feats.shape == (1000, 40)

            try:
                lhotse.LilcomFilesWriter("unused")
            except ImportError:
                pass
            else:
                raise AssertionError("LilcomFilesWriter should fail when lilcom is unavailable.")
            """
        ),
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_default_feature_storage_falls_back_to_numpy_without_lilcom(tmp_path):
    result = run_without_lilcom(
        tmp_path,
        textwrap.dedent(
            """
            from tempfile import TemporaryDirectory

            import lhotse

            recording = lhotse.Recording.from_file("test/fixtures/mono_c0.wav")
            cut = recording.to_cut()
            extractor = lhotse.Fbank(lhotse.FbankConfig(sampling_rate=recording.sampling_rate))

            with TemporaryDirectory() as d:
                cuts = lhotse.CutSet.from_cuts([cut]).compute_and_store_features(
                    extractor=extractor,
                    storage_path=d,
                )
                features = cuts[0].features
                assert features.storage_type == "numpy_files"
                assert features.storage_key.endswith(".npy")

            with TemporaryDirectory() as d:
                copied = lhotse.CutSet.from_file("test/fixtures/libri/cuts.json").copy_data(
                    d, verbose=False
                )
                copied_features = copied[0].features
                assert copied_features.storage_type == "numpy_files"
            """
        ),
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.skipif(not is_module_available("lilcom"), reason="Requires lilcom.")
def test_feature_storage_backend_env_override(monkeypatch):
    monkeypatch.setenv("LHOTSE_FEATURES_STORAGE_BACKEND", "lilcom_chunky")
    assert default_features_storage_backend_name() == "lilcom_chunky"
