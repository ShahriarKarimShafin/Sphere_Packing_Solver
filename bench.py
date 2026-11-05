"""Compare refinement methods from the same start under matched CPU budgets."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--psd", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--box-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--methods", nargs="+", choices=["none", "repair", "slp", "hybrid"],
        default=["none", "hybrid"],
    )
    parser.add_argument("--cpu-seconds", type=float, default=30.)
    parser.add_argument("--proposals-per-round", type=int, default=50000)
    parser.add_argument("--relax-iterations", type=int, default=400)
    parser.add_argument("--refine-every", type=int, default=1)
    parser.add_argument("--slp-steps", type=int, default=2)
    parser.add_argument("--lp-seconds", type=float, default=10.)
    parser.add_argument("--backend", choices=["native", "python", "auto"], default="native")
    args = parser.parse_args()
    if (
        not args.cpu_seconds > 0
        or min(args.proposals_per_round, args.relax_iterations, args.refine_every, args.slp_steps) < 1
        or not args.lp_seconds > 0
        or any(not 0 <= seed < 2**64 for seed in args.seeds)
        or len(set(args.seeds)) != len(args.seeds)
        or len(set(args.methods)) != len(args.methods)
    ):
        parser.error("Require positive budgets and distinct valid seeds/methods")
    for path in (args.psd, args.initial, args.box_file):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    # Run methods consecutively so competing children do not share CPU time.
    for seed in args.seeds:
        for method in args.methods:
            folder = output / f"seed{seed}_{method}"
            folder.mkdir()
            command = [
                sys.executable, "-u", str(HERE / "pack.py"), "new",
                "--psd", str(args.psd.resolve()), "--initial", str(args.initial.resolve()),
                "--box-file", str(args.box_file.resolve()), "--starts", "0", "--seed", str(seed),
                "--run", str(folder / "run"), "--export", str(folder / "result"),
                "--box", "rectangular", "--rounds", "1000000000",
                "--cpu-seconds", str(args.cpu_seconds),
                "--proposals-per-round", str(args.proposals_per_round),
                "--relax-iterations", str(args.relax_iterations), "--refinement", method,
                "--refine-every", str(args.refine_every), "--slp-steps", str(args.slp_steps),
                "--lp-seconds", str(args.lp_seconds), "--backend", args.backend,
            ]
            started = time.perf_counter()
            with (folder / "run.log").open("w") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Benchmark child failed: {folder}/run.log")
            summary = json.loads((folder / "run/last_command.json").read_text())
            state = json.loads((folder / "run/state.json").read_text())
            verification = json.loads((folder / "result/verification.json").read_text())
            if not verification["passed"] or not verification["original_radii_and_counts_exact"]:
                raise RuntimeError("An exported candidate failed independent verification")
            row = dict(
                seed=seed,
                method=method,
                nominal_cpu_seconds=args.cpu_seconds,
                actual_search_cpu_seconds=summary["command_cpu_seconds"],
                startup_export_inclusive_wall_seconds=time.perf_counter() - started,
                final_percent=summary["best_packing_percent"],
                rounds=state["total_rounds"],
                counters=state["counters"],
                psd_fingerprint=state["psd_fingerprint"],
                implementation_sha256=state["implementation_sha256"],
                exact_overlaps=verification["overlapping_pairs_exact"],
                exact_self_image_violations=verification["self_image_violations_exact"],
                command=command,
                result=str((folder / "result").relative_to(output)),
            )
            rows.append(row)
            report = dict(
                results=rows,
                global_optimality_proved=False,
                note="Same nominal CPU caps checked between rounds; last rounds can overrun. Compare actual CPU too. These are search trajectories, not global density bounds.",
            )
            (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
