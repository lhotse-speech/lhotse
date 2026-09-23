import subprocess
import sys


def test_lhotse_importable_without_urllib3():
    # urllib3 only arrives with the optional aistore extra, so a plain
    # install of lhotse must not import it at module level.
    code = "import sys; sys.modules['urllib3'] = None; import lhotse"
    subprocess.run([sys.executable, "-c", code], check=True)
