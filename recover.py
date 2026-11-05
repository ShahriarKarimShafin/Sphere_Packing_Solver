"""Roll back explicitly to a checked previous checkpoint after checkpoint damage.

    python recover.py --run run01

An intact state.json is required. The latest verified best is retained if its
files remain valid; work counters roll back with the working state.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import uuid

import numpy as np

from packing.state import (
    atomic_json, checkpoint, exclusive_run, inside, prune, read_state,
    sha, signature, summary, verify_best,
)
from packing.stop import new_monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--allow-code-change", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    with exclusive_run(run):
        live = json.loads((run / "state.json").read_text())
        if not live.get("previous_checkpoint"):
            raise ValueError("No previous checkpoint exists yet. Start a new run from the verified CSV.")
        previous = inside(run, live["previous_checkpoint"], "checkpoints")
        expected_hash = live["previous_checkpoint_sha256"]
        if sha(previous) != expected_hash:
            raise ValueError("The previous checkpoint is also damaged; use a new run from the verified CSV.")
        with np.load(previous, allow_pickle=False) as data:
            state = json.loads(str(data["metadata"]))
        state["checkpoint_sha256"] = expected_hash
        state, psd, x, bins, L = read_state(run, args.allow_code_change, state)
        preserved_latest_best = False
        try:
            verify_best(run, live)
            if Fraction(live["best_volume_m3"]) < Fraction(state["best_volume_m3"]):
                for key in ("best", "best_volume_m3", "best_packing_fraction", "best_verification_sha256"):
                    state[key] = live[key]
                preserved_latest_best = True
        except (ValueError, OSError, KeyError):
            pass
        # Create a new checkpoint rather than modify surviving immutable files.
        # Restart the plateau history because the working trajectory rolled back.
        if "convergence" in state:
            old_monitor = state["convergence"]
            state["convergence"] = new_monitor(
                state["total_rounds"], state["best_packing_fraction"],
                enabled=old_monitor["enabled"], window=old_monitor["window_rounds"],
                blocks=old_monitor["required_blocks"], tolerance=old_monitor["tolerance_fraction"],
            )
        state["implementation_sha256"] = signature()
        state = checkpoint(run, state, x, bins, L)
        prune(run, state)
        (run / "events").mkdir(exist_ok=True)
        event = dict(
            event="explicit_checkpoint_rollback",
            old_committed_rounds=live["total_rounds"],
            restored_committed_rounds=state["total_rounds"],
            retained_latest_verified_best=preserved_latest_best,
            convergence_history_reset="convergence" in state,
            note="Work counters represent the restored committed state. Replayed work is not an exhaustive search count.",
        )
        atomic_json(run / "events" / ("recovery_" + uuid.uuid4().hex + ".json"), event)
        print(json.dumps(dict(**event, result=summary(run, state)), indent=2))


if __name__ == "__main__":
    main()
