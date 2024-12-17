#!/usr/bin/env python3
"""
Use this script like: https://lhotse.readthedocs.io/en/latest/cli.html
"""

# Note: we import all the CLI modes here so they get auto-registered
#       in Lhotse's main CLI entry-point. Then, setuptools is told to
#       invoke the "cli()" method from this script.
from lhotse.bin.modes import *
