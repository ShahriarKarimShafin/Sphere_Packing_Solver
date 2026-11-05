#!/usr/bin/env python3
"""Read saved progress without changing or stopping a running search.

    python progress.py --run run_refined
    python progress.py --run run_refined --window 100 --blocks 3 --threshold 0.0001

The threshold is a packing-fraction gain per window: 0.0001 is 0.01 percentage
points. A plateau describes the saved trajectory, not a global maximum.
"""

import argparse
import json
import math
from pathlib import Path

from packing.stop import diagnostic


def report(run, window=100, blocks=3, threshold=0.0001, prior_rounds=0):
    """Summarize recent solve outcomes and complete consecutive gain windows."""
    if window < 1 or blocks < 1 or prior_rounds < 0:
        raise ValueError("window/blocks must be positive; prior-rounds must be nonnegative")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    run = Path(run)
    # State and records are atomic. Exclude records committed after this snapshot.
    state = json.loads((run / "state.json").read_text())
    end = state["total_rounds"]
    rows = {}
    for path in sorted((run / "records").glob("*.json")):
        if path.stem.isdigit() and int(path.stem) <= end:
            row = json.loads(path.read_text())
            if row["round"] != int(path.stem):
                raise ValueError("Record filename and round do not match: " + str(path))
            rows[row["round"]] = row
    config = state["config"]
    recent = [rows[number] for number in sorted(rows) if end - window < number <= end]
    slp = [
        refinement for row in recent for refinement in row.get("refinement", [])
        if refinement.get("stage") == "full_periodic_slp"
    ]
    repair = [
        refinement for row in recent for refinement in row.get("refinement", [])
        if refinement.get("stage") == "bottleneck_repair"
    ]
    optimizations = [row.get("optimization", {}) for row in recent]
    windows = []
    for offset in range(blocks):
        stop = end - offset * window
        start = stop - window
        if start not in rows or stop not in rows:
            break
        if any(number not in rows for number in range(start, stop + 1)):
            break
        gain = rows[stop]["best_verified_fraction"] - rows[start]["best_verified_fraction"]
        if gain < -1e-12:
            raise ValueError("Best-density history decreased; inspect the input records")
        windows.append({
            "from_round": start,
            "to_round": stop,
            "gain_fraction": gain,
            "gain_percentage_points": 100 * gain,
            "below_threshold": gain < threshold,
        })
    windows.reverse()
    enough = len(windows) == blocks
    plateau = enough and all(row["below_threshold"] for row in windows)
    inner_limited = any("LIMIT" in solve.get("message", "").upper() for solve in optimizations)
    inner_limited |= any(refinement.get("lp_status") == 1 for refinement in slp + repair)
    return {
        "run": str(run.resolve()),
        "completed_rounds_in_this_run": end,
        "prior_rounds_supplied_by_user": prior_rounds,
        "displayed_cumulative_rounds": end + prior_rounds,
        "best_packing_percent": 100 * state["best_packing_fraction"],
        "saved_automatic_stopping": diagnostic(state.get("convergence")),
        "porosity_percent": 100 * (1 - state["best_packing_fraction"]),
        "recent_record_count": len(recent),
        "recent_relaxation_successes": sum(solve.get("success") is True for solve in optimizations),
        "recent_relaxation_limit_stops": sum(
            "LIMIT" in solve.get("message", "").upper() for solve in optimizations
        ),
        "recent_full_slp_scheduled": len(slp),
        "recent_full_slp_attempted": sum("lp_status" in refinement for refinement in slp),
        "recent_full_slp_accepted": sum(refinement.get("accepted", False) for refinement in slp),
        "recent_full_slp_time_limits": sum(
            refinement.get("lp_status") == 1
            and "time limit" in refinement.get("lp_message", "").lower()
            for refinement in slp
        ),
        "recent_local_repair_accepted": sum(refinement.get("accepted", False) for refinement in repair),
        "full_slp_particle_limit": config.get("slp_max_particles"),
        "lp_seconds_per_attempt": config.get("lp_seconds"),
        "relaxation_iteration_limit": config.get("relax_iterations"),
        "function_evaluation_limit": config.get("maxfun"),
        "window_rounds": window,
        "requested_consecutive_windows": blocks,
        "threshold_fraction_per_window": threshold,
        "threshold_percentage_points_per_window": 100 * threshold,
        "completed_windows": windows,
        "enough_history_for_plateau_rule": enough,
        "practical_plateau_observed": plateau,
        "inner_solve_limit_hit_in_recent_window": bool(inner_limited),
        "global_optimality_proved": False,
        "interpretation": (
            "A trajectory plateau is not proof of an optimum. "
            "This command reads saved reports; it does not independently verify geometry. "
            "Window endpoints require saved records; round-zero density is not invented. "
            "prior-rounds changes display labels only and does not add history."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0001)
    parser.add_argument("--prior-rounds", type=int, default=0)
    args = parser.parse_args()
    try:
        value = report(args.run, args.window, args.blocks, args.threshold, args.prior_rounds)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
