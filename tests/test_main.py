import subprocess
import sys


def test_main_compiles():
    subprocess.check_call([sys.executable, "-m", "py_compile", "main.py"])
