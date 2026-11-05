"""Run independent serial optimizations, retaining one checkpoint per seed.

    python multi.py new --campaign study --psd input.csv --seeds 1 2 3 --rounds 100
    python multi.py resume --campaign study
    python multi.py extend --campaign study --rounds 100
    python multi.py status --campaign study

Extra options after new are forwarded to pack.py new. Each seed keeps its own
trajectory; child runs execute consecutively without migration.
"""

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from packing.cli import parser as engine_parser
from packing.geometry import read_psd, write_psd
from packing.state import (
    atomic_json, exclusive_run, export_copy, read_state, sha, signature,
)


def collect(folder, settings, target):
    """Summarize committed child states and select the smallest verified cell."""
    rows = []
    best = None
    for seed in settings["seeds"]:
        run = folder / "runs" / f"seed_{seed:020d}"
        if not (run / "state.json").exists():
            rows.append(dict(seed=seed, completed_rounds=0, phase="not_initialized"))
            continue
        with exclusive_run(run):
            state, psd, _, _, _ = read_state(run)
        if psd.fingerprint != settings["psd_fingerprint"]:
            raise ValueError("Child PSD differs from the campaign")
        convergence = state.get("convergence", {})
        plateau = convergence.get("enabled", False) and convergence.get("met", False)
        phase = (
            "plateau" if plateau
            else "complete" if state["total_rounds"] >= target
            else "incomplete"
        )
        rows.append(dict(
            seed=seed,
            completed_rounds=state["total_rounds"],
            target_rounds=target,
            best_packing_fraction=state["best_packing_fraction"],
            counters=state["counters"],
            run=str(run.relative_to(folder)),
            phase=phase,
        ))
        # Compare the recorded decimal cell volumes without floating-point rounding.
        if best is None or Fraction(state["best_volume_m3"]) < Fraction(best[1]["best_volume_m3"]):
            best = run, state
    complete = all(row.get("phase") in ("complete", "plateau") for row in rows)
    report = dict(
        target_rounds_per_seed=target,
        seeds=len(rows),
        complete=complete,
        independent_optimized_runs=True,
        parallel=False,
        runs=rows,
        completed_rounds=sum(row["completed_rounds"] for row in rows),
        global_optimality_proved=False,
    )
    if best is not None:
        run, state = best
        source = run / state["best"]
        report.update(
            best_packing_fraction=state["best_packing_fraction"],
            best_run=str(run.relative_to(folder)),
            coordinates=str((source / "x_y_z_radius.csv").resolve()),
            box=str((source / "box.json").resolve()),
            verification=str((source / "verification.json").resolve()),
        )
        atomic_json(folder / "BEST.json", dict(
            packing_fraction=state["best_packing_fraction"],
            run=str(run.relative_to(folder)),
            coordinates=str((source / "x_y_z_radius.csv").relative_to(folder)),
            box=str((source / "box.json").relative_to(folder)),
            verification=str((source / "verification.json").relative_to(folder)),
            paths_relative_to_campaign=True,
            global_optimality_proved=False,
        ))
    atomic_json(folder / "progress.json", report)
    return report, best


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=["new", "resume", "extend", "status", "export"])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--psd", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--export", "--out", type=Path)
    args, engine_args = parser.parse_known_args()
    folder = args.campaign.resolve()
    with exclusive_run(folder):
        if args.command == "new":
            if any(path.name != ".writer.lock" for path in folder.iterdir()):
                raise ValueError("New campaign requires a new/empty directory")
            if args.psd is None or not args.seeds or args.rounds is None or args.rounds < 0:
                raise ValueError("new requires --psd, distinct --seeds and nonnegative --rounds")
            if (len(set(args.seeds)) != len(args.seeds)
                    or any(not 0 <= seed < 2**64 for seed in args.seeds)):
                raise ValueError("Seeds must be distinct uint64 integers")
            protected = {
                "--run", "--psd", "--seed", "--rounds", "--export", "--initial",
                "--box-file", "--cpu-seconds", "--allow-code-change",
            }
            if any(token.split("=", 1)[0] in protected for token in engine_args):
                raise ValueError("Campaign owns paths, seeds and budgets; use fresh independent initializations")
            # Validate forwarded options before creating campaign input/results.
            engine_parser().parse_args([
                "new", "--psd", str(args.psd), "--run", str(folder / "validation_only"),
                "--rounds", str(args.rounds), "--seed", str(args.seeds[0]), *engine_args,
            ])
            psd = read_psd(args.psd)
            write_psd(folder / "psd.csv", psd)
            settings = dict(
                schema=1,
                seeds=args.seeds,
                engine_args=engine_args,
                psd_fingerprint=psd.fingerprint,
                implementation_sha256=signature(),
            )
            atomic_json(folder / "settings.json", settings)
            campaign = dict(
                schema=1,
                settings_sha256=sha(folder / "settings.json"),
                target_rounds=args.rounds,
            )
            atomic_json(folder / "campaign.json", campaign)
        else:
            if engine_args or args.psd is not None or args.seeds is not None:
                raise ValueError("Existing campaign settings are immutable; only extend accepts additional --rounds")
            campaign = json.loads((folder / "campaign.json").read_text())
            if sha(folder / "settings.json") != campaign["settings_sha256"]:
                raise ValueError("Campaign settings checksum mismatch")
            settings = json.loads((folder / "settings.json").read_text())
            if settings["implementation_sha256"] != signature():
                raise ValueError("Campaign source changed; restore the same package")
            if read_psd(folder / "psd.csv").fingerprint != settings["psd_fingerprint"]:
                raise ValueError("Campaign PSD was changed")
            if args.command == "extend":
                if args.rounds is None or args.rounds < 0:
                    raise ValueError("extend requires nonnegative additional --rounds per seed")
                campaign["target_rounds"] += args.rounds
                atomic_json(folder / "campaign.json", campaign)
            elif args.rounds is not None:
                raise ValueError("--rounds is only accepted by new and extend")

        interrupted = False
        if args.command in ("new", "resume", "extend"):
            try:
                for seed in settings["seeds"]:
                    run = folder / "runs" / f"seed_{seed:020d}"
                    if (run / "state.json").exists():
                        state, _, _, _, _ = read_state(run)
                        convergence = state.get("convergence", {})
                        if convergence.get("enabled", False) and convergence.get("met", False):
                            continue
                        remaining = max(0, campaign["target_rounds"] - state["total_rounds"])
                        if not remaining:
                            continue
                        command = ["extend", "--run", str(run), "--rounds", str(remaining)]
                    else:
                        command = [
                            "new", "--psd", str(folder / "psd.csv"), "--run", str(run),
                            "--rounds", str(campaign["target_rounds"]), "--seed", str(seed),
                            *settings["engine_args"],
                        ]
                    (folder / "logs").mkdir(exist_ok=True)
                    print(json.dumps(dict(
                        seed=seed, phase="running", target_rounds=campaign["target_rounds"],
                    )), flush=True)
                    with (folder / "logs" / f"seed_{seed:020d}.log").open("a") as handle:
                        child = subprocess.Popen(
                            [sys.executable, "-u", str(Path(__file__).with_name("pack.py")), *command],
                            stdout=handle, stderr=subprocess.STDOUT,
                        )
                        try:
                            returncode = child.wait()
                        except KeyboardInterrupt:
                            if os.name == "posix" and child.poll() is None:
                                child.send_signal(signal.SIGINT)
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                child.terminate()
                                child.wait()
                            interrupted = True
                            break
                    if returncode == 130:
                        interrupted = True
                        break
                    if returncode:
                        raise RuntimeError(f"Seed {seed} failed; see its log. Completed checkpoints are retained.")
                    progress, _ = collect(folder, settings, campaign["target_rounds"])
                    print(json.dumps(dict(
                        seed=seed, phase="completed",
                        best_packing_fraction=progress.get("best_packing_fraction"),
                    )), flush=True)
            except KeyboardInterrupt:
                interrupted = True
        report, best = collect(folder, settings, campaign["target_rounds"])
        if args.export:
            if best is None:
                raise ValueError("No initialized result is available to export")
            export_copy(best[0], best[1], args.export)
            report["export_directory"] = str(args.export.resolve())
            atomic_json(args.export / "campaign_summary.json", report)
        elif args.command == "export":
            raise ValueError("export requires --out DESTINATION")
        report["interrupted"] = interrupted
        print(json.dumps(report, indent=2), flush=True)
        return 130 if interrupted else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
