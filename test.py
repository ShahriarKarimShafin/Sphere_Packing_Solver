#!/usr/bin/env python3
"""Run numerical and workflow tests; optionally compare an original package."""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import json
from pathlib import Path
import platform
import sys
import time
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Original flat solver directory for differential tests.")
    parser.add_argument("--report", type=Path, help="Write a JSON test summary and observed timings.")
    parser.add_argument("--pattern", default="test_*.py", help="unittest discovery pattern.")
    parser.add_argument("--work", type=Path, help="Temporary-fixture parent; defaults to the system temp directory.")
    args = parser.parse_args()
    if args.baseline:
        os.environ["PACKING_BASELINE_DIR"] = str(args.baseline.resolve())
    if args.work:
        args.work.mkdir(parents=True, exist_ok=True)
        os.environ["PACKING_TEST_TMP"] = str(args.work.resolve())
    root = Path(__file__).resolve().parent
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=args.pattern, top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    from packing.state import signature
    from tests.test_regression import RESULTS
    report = dict(
        passed=result.wasSuccessful(), tests_run=result.testsRun,
        failures=[(str(test), detail) for test, detail in result.failures],
        errors=[(str(test), detail) for test, detail in result.errors],
        skipped=[(str(test), reason) for test, reason in result.skipped],
        elapsed_seconds=time.perf_counter() - started,
        python=sys.version.split()[0], platform=platform.platform(),
        implementation_sha256=signature(), baseline=str(args.baseline.resolve()) if args.baseline else None,
        differential_cases=RESULTS, global_optimality_proved=False,
        timing_scope="Short native regression runs on this machine; not a full-PSD speedup benchmark.",
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
