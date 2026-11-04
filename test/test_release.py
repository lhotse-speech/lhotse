import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("bash") is None, reason="Release script uses Bash")
@pytest.mark.parametrize("stale_lock", [False, True])
def test_release_script_uses_its_project_directory(tmp_path, stale_lock):
    project = tmp_path / "release project"
    caller = tmp_path / "another project"
    (project / "tools").mkdir(parents=True)
    caller.mkdir()
    script = project / "tools" / "make_release.sh"
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "tools/make_release.sh", script
    )
    for root in (project, caller):
        for name in ("dist", "build", "lhotse.egg-info"):
            (root / name).mkdir()
            (root / name / "sentinel").touch()

    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    uv = binary_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\t%s\\t%s\\n" "$PWD" "$*" "${LHOTSE_PREPARING_RELEASE-}" >> "$LHOTSE_RELEASE_LOG"\n'
        'if [[ "$1" == lock && "$LHOTSE_FAIL_LOCK" == 1 ]]; then exit 1; fi\n'
        'if [[ "$1" == build ]]; then mkdir -p dist; touch dist/new.whl dist/new.tar.gz; fi\n'
    )
    uv.chmod(0o755)
    log = tmp_path / "commands.log"
    env = {
        **os.environ,
        "PATH": str(binary_dir) + os.pathsep + os.environ["PATH"],
        "LHOTSE_RELEASE_LOG": str(log),
        "LHOTSE_FAIL_LOCK": "1" if stale_lock else "0",
    }
    env.pop("LHOTSE_PREPARING_RELEASE", None)
    result = subprocess.run(
        [shutil.which("bash"), str(script)],
        cwd=caller,
        env=env,
        capture_output=True,
        text=True,
    )

    for name in ("dist", "build", "lhotse.egg-info"):
        assert (caller / name / "sentinel").is_file()

    commands = [line.split("\t") for line in log.read_text().splitlines()]
    if stale_lock:
        assert result.returncode != 0
        assert commands == [[str(project), "lock --check", ""]]
        for name in ("dist", "build", "lhotse.egg-info"):
            assert (project / name / "sentinel").is_file()
    else:
        assert result.returncode == 0, result.stderr
        assert commands == [
            [str(project), "lock --check", ""],
            [str(project), "build --no-sources", "1"],
        ]
        assert sorted(path.name for path in (project / "dist").iterdir()) == [
            "new.tar.gz",
            "new.whl",
        ]
        assert not (project / "build").exists()
        assert not (project / "lhotse.egg-info").exists()
        assert "uv publish" in result.stdout
