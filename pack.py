#!/usr/bin/env python3
"""Run the packing engine with explicit options; see `python pack.py new --help`."""
import sys

from packing.cli import main


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
