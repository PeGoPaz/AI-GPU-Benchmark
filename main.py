"""Runs the app from a checkout without installing it: `python main.py`."""
import sys

from gpu_bench.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
