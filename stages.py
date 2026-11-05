#!/usr/bin/env python3
"""Build a bimodal crystal seed, then run none and hybrid to their plateaus.

Place beside refine.py. Ordinary solver options apply to both stages. Prefix an
option with --none- or --hybrid- to override only that stage, for example:
--proposals-per-round 50000000 --hybrid-convergence-blocks 2.
The stage names describe their defaults; --none-refinement and
--hybrid-refinement can select other supported refinement modes.
Use --initializer previous for the original FCC/random starts (10 by default).
Explicit --initial inputs retain the previous import behavior.
The default ordered initializer uses lattice_seeds/bimodal_lattice.py with
structure=best, the full cell-search list, cage=0.5 and no widening.
Use --initializer slabs for the older FCC slab/infill builder.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


DEFAULTS = [
    "--box", "rectangular", "--boundary", "periodic", "--seed", "2026",
    "--proposals-per-round", "30000000", "--relax-iterations", "5000",
    "--maxfun", "1000000", "--schedule", "portfolio", "--backend", "native",
    "--convergence", "--convergence-tolerance", "0.00001",
]
STAGE_DEFAULTS = {
    "none": ["--starts", "10", "--refinement", "none",
             "--convergence-window", "100", "--convergence-blocks", "3"],
    "hybrid": ["--refinement", "hybrid",
               "--convergence-window", "50", "--convergence-blocks", "2"],
}
INITIAL_OPTIONS = {"--initial", "--box-file", "--starts", "--init-mode", "--initial-iterations"}
INPUT_PATHS = {"--initial", "--box-file"}
RESERVED = {"--psd", "--run", "--export", "--from-export", "--help"}
ORDERED_OPTIONS = {
    "band_ratios": (float, "+"), "widths": (float, "+"),
    "refine_candidates": (int, None), "seed": (int, None),
    "domain_steps": (int, None), "pair_limit": (int, None), "lp_seconds": (float, None),
    "slab_scan": (int, None), "infill_targets": (float, "+"),
    "infill_phases": (int, None), "infill_restarts": (int, None),
    "infill_refine": (int, None), "infill_band_ratio": (float, None),
    "infill_max_groups": (int, None), "infill_max_grid_edge": (int, None),
    "infill_max_sites": (int, None), "infill_seconds": (float, None),
}
PREPARATION_OPTIONS = {
    "steps": int, "increment": float, "min_increment": float, "cage": float,
    "iterations": int, "maxfun": int, "cpu_seconds": float,
    "gtol": float, "ftol": float, "curvature": int,
}


def parser():
    options = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog=("All pack.py new search options are accepted, including their "
                "--none-OPTION and --hybrid-OPTION forms. Initializer options "
                "apply only to the first stage. Use python pack.py new --help "
                "for the solver's option names."),
    )
    options.add_argument("--psd", required=True, type=Path)
    options.add_argument("--name", required=True, help="New experiment label")
    options.add_argument("--output-dir", type=Path, help="Output parent; default: solver folder")
    options.add_argument("--dry-run", action="store_true", help="Validate and print both stages; do not run")
    options.add_argument("--initializer", choices=["ordered", "previous", "slabs"],
                         help="Default: bimodal crystal; previous: FCC/random; slabs: old FCC builder")
    options.add_argument("--lattice-structure", default=None)
    options.add_argument("--lattice-search", default=None, help="Default: 1-1,2-2,6-6,6-1,10-1,11-1")
    options.add_argument("--lattice-cage", type=float, default=None)
    options.add_argument("--lattice-widen", action=argparse.BooleanOptionalAction, default=None)
    options.add_argument("--lattice-starts", type=int, default=None, help="Crystal search starts; default: 30")
    options.add_argument("--lattice-seed", type=int, default=None, help="Crystal search seed; default: 1")
    options.add_argument("--lattice-lp-iters", type=int, default=None, help="Caged seed LP steps; default: 300")
    options.add_argument("--lattice-recenter-iters", type=int, default=None, help="Fixed-cell recentering LP steps; default: 30")
    options.add_argument("--lattice-ratio", type=float, default=None)
    options.add_argument("--lattice-split-radius", type=float, default=None)
    options.add_argument("--lattice-cache", type=Path, default=None, help="Shared crystal cache; default: OUTPUT/_lattice_cache")
    for name, (value_type, nargs) in ORDERED_OPTIONS.items():
        options.add_argument("--ordered-" + name.replace("_", "-"), type=value_type,
                             nargs=nargs, default=None, help="Ordered builder only; default comes from ordered.py")
    for name in ("slabs", "infill"):
        options.add_argument("--ordered-" + name, action=argparse.BooleanOptionalAction,
                             default=None, help="Enable/disable this ordered constructor")
    for name, value_type in PREPARATION_OPTIONS.items():
        options.add_argument("--prepare-" + name.replace("_", "-"), type=value_type,
                             default=None, help="Ordered shortlist preparation only")
    return options


def route_options(tokens):
    """Strip stage prefixes while preserving argv values, repeated options and order."""
    groups = []
    for token in tokens:
        if token.startswith("--"):
            if token == "--":
                raise ValueError("Pass solver options directly, without a -- separator")
            scope, option = "common", token
            for stage in STAGE_DEFAULTS:
                prefix = f"--{stage}-"
                if token.startswith(prefix):
                    scope, option = stage, "--" + token[len(prefix):]
                    break
            groups.append((scope, [option]))
        elif groups:
            groups[-1][1].append(token)
        else:
            raise ValueError(f"Unexpected value without an option: {token}")

    routed = {"common": [], "none": [], "hybrid": []}
    for scope, group in groups:
        key, separator, value = group[0].partition("=")
        if key in RESERVED:
            raise ValueError(f"{key} is managed by the launcher; use --psd, --name and --output-dir")
        if key in INITIAL_OPTIONS:
            if scope == "hybrid":
                raise ValueError(f"{key} cannot override the second stage's imported best packing")
            scope = "none"
        if key in INPUT_PATHS:
            # Resolve input paths before the child changes to the solver folder.
            if separator:
                group[0] = key + "=" + str(Path(value).expanduser().resolve())
            elif len(group) == 2:
                group[1] = str(Path(group[1]).expanduser().resolve())
        routed[scope].extend(group)
    return routed


def check_configuration(config, allow_generated_seed=False):
    """Check cross-option bounds also enforced by the current engine's CLI."""
    if config.convergence is False:
        raise ValueError("This workflow requires convergence stopping in both stages")
    if config.seed > 2**64 - 1 or config.curvature > 30 or config.swap_window > 2**31 - 1:
        raise ValueError("Seed must fit uint64; curvature <=30; swap-window <=2^31-1")
    if not config.min_step <= config.step < 1:
        raise ValueError("Require 0 < min-step <= step <1")
    if config.initial_step is not None and not config.min_step <= config.initial_step < 1:
        raise ValueError("Require min-step <= initial-step <1")
    if (max(config.slp_move, config.repair_move) > .5 or
            config.slp_strain > .1 or config.repair_max_active < 2):
        raise ValueError("Require move fractions <=0.5, strain <=0.1 and repair-max-active >=2")
    if config.box_file and len(config.initial) != 1:
        raise ValueError("--box-file requires exactly one --initial")
    if config.starts == 0 and not config.initial and not allow_generated_seed:
        raise ValueError("--starts 0 requires an imported initial configuration")


def make_plan(root, output, args, routed, allow_generated_seed=False):
    """Use the real engine parser and launcher presets to validate both stages."""
    from packing.cli import parser as engine_parser
    from refine import PRESET

    preset = [item for pair in PRESET.items() for item in pair]
    stages = []
    first_export = output / f"result_{args.name}_none"
    for stage, defaults in STAGE_DEFAULTS.items():
        run = output / f"run_{args.name}_{stage}"
        export = output / f"result_{args.name}_{stage}"
        settings = [*DEFAULTS, *defaults, *routed["common"], *routed[stage]]
        if stage == "none":
            inputs = ["--psd", str(args.psd)]
            engine_inputs = inputs
        else:
            inputs = ["--from-export", str(first_export)]
            # These files will be created by stage one. Parse their expected paths
            # now, without requiring the export to exist during a dry run.
            engine_inputs = [
                "--psd", str(first_export / "psd.csv"),
                "--initial", str(first_export / "packing_coordinates.csv"),
                "--box-file", str(first_export / "box.json"), "--starts", "0",
            ]
        destinations = ["--run", str(run), "--export", str(export)]
        config = engine_parser().parse_args([
            "new", *preset, *engine_inputs, *settings, *destinations,
        ])
        check_configuration(config, allow_generated_seed=allow_generated_seed and stage == "none")
        stages.append(dict(
            stage=stage, run=str(run), export=str(export),
            log=str(output / f"{args.name}_{stage}.log"),
            command=[sys.executable, "-u", str(root / "refine.py"), "new",
                     *inputs, *settings, *destinations],
            settings=vars(config),
        ))
    return dict(
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stages=stages, global_optimality_proved=False,
    )


def imported_seed_options(tokens, export):
    """Replace only the first stage's seed inputs; keep other overrides in order."""
    retained, skip = [], False
    for token in tokens:
        if token.startswith("--"):
            skip = token.split("=", 1)[0] in {"--initial", "--box-file", "--starts"}
        if not skip:
            retained.append(token)
    return [*retained, "--initial", str(export / "packing_coordinates.csv"),
            "--box-file", str(export / "box.json"), "--starts", "0"]


def slab_plan(root, output, args, routed, preliminary):
    """Resolve the ordered builder, preserving the two ordinary stage settings."""
    missing = [name for name in ("ordered.py", "prepare.py") if not (root / name).is_file()]
    if missing:
        raise RuntimeError("Ordered initialization needs " + ", ".join(missing) +
                           " beside stages.py; use --initializer previous for the old starts")
    try:
        from ordered import Settings
        from prepare import Settings as Preparation
    except ImportError as error:
        raise RuntimeError("Ordered initialization requires compatible ordered.py and prepare.py "
                           "beside stages.py; use --initializer previous for the old starts") from error
    source = preliminary["stages"][0]["settings"]
    if len(source["initial"]) > 1:
        raise ValueError("Ordered mode accepts at most one imported template; use --initializer previous for multiple imports")
    parameters = asdict(Settings())
    parameters["seed"] = source["seed"]
    for name in (*ORDERED_OPTIONS, "slabs", "infill"):
        value = getattr(args, "ordered_" + name)
        if value is not None:
            if name not in parameters:
                raise ValueError(f"Installed ordered.py does not support --ordered-{name.replace('_', '-')}")
            parameters[name] = value
    settings = Settings(**parameters)
    settings.check()
    preparation = asdict(Preparation())
    preparation.update(gtol=source["relax_gtol"], ftol=source["relax_ftol"], curvature=source["curvature"])
    for name in PREPARATION_OPTIONS:
        value = getattr(args, "prepare_" + name)
        if value is not None:
            preparation[name] = value
    Preparation(**preparation).check()
    export = output / f"seed_{args.name}"
    updated = {key: value.copy() for key, value in routed.items()}
    updated["none"] = imported_seed_options(updated["none"], export)
    plan = make_plan(root, output, args, updated)
    plan["initializer"] = dict(mode="slabs", psd=str(args.psd), export=str(export),
        log=str(output / f"{args.name}_ordered.log"), kind=source["box"], backend=source["backend"],
        initial=str(source["initial"][0]) if source["initial"] else None,
        box_file=str(source["box_file"]) if source["box_file"] else None,
        settings=asdict(settings), preparation=preparation,
        sources={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                 for name in ("ordered.py", "prepare.py")})
    return plan


def run_slab_builder(specification):
    """Advance to ordinary optimization only after an exact-checked seed export."""
    from ordered import Settings, build_seed
    from prepare import Settings as Preparation
    print(f"Building ordered seed; progress: {specification['log']}", flush=True)
    with Path(specification["log"]).open("x", encoding="utf-8") as stream:
        build_seed(specification["psd"], specification["export"],
            settings=Settings(**specification["settings"]),
            preparation=Preparation(**specification["preparation"]),
            kind=specification["kind"], backend=specification["backend"],
            initial=specification["initial"], box_file=specification["box_file"],
            progress=lambda row: print(json.dumps(row), file=stream, flush=True))
    export = Path(specification["export"])
    verification = json.loads((export / "verification.json").read_text(encoding="utf-8"))
    if (not verification.get("passed") or
            verification.get("overlapping_pairs_exact") != 0 or
            verification.get("self_image_violations_exact") != 0 or
            not verification.get("original_radii_and_counts_exact")):
        raise RuntimeError("Ordered seed failed exact export verification; no optimization stage started")
    box = json.loads((export / "box.json").read_text(encoding="utf-8"))
    print(f"Ordered seed verified: {100 * box['packing_fraction']:.8f}%; "
          f"first stage imports {export / 'packing_coordinates.csv'}", flush=True)


LATTICE_OPTIONS = {
    "structure": "best", "search": "1-1,2-2,6-6,6-1,10-1,11-1", "cage": 0.5,
    "widen": False, "starts": 30, "seed": 1, "lp_iters": 300,
    "ratio": None, "split_radius": None, "recenter_iters": 30,
}


def lattice_plan(root, output, args, routed, preliminary):
    """Make one noninteractive crystal seed for this PSD; do not change search settings."""
    from lattice_seeds.build_seed import check_settings

    sources = [root / "lattice_seeds" / name for name in
               ("binary_lattice.py", "bimodal_lattice.py", "build_seed.py")]
    for path in sources:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty lattice builder: {path}")
    source = preliminary["stages"][0]["settings"]
    if source["initial"] or source["box_file"]:
        raise ValueError("Explicit --initial/--box-file uses --initializer previous; "
                         "ordered mode builds a new seed from --psd")
    if source["box"] != "rectangular" or source["boundary"] != "periodic":
        raise ValueError("Bimodal crystal initialization requires --box rectangular --boundary periodic")
    if any(value is not None for key, value in vars(args).items()
           if key.startswith(("ordered_", "prepare_"))):
        raise ValueError("--ordered-* and --prepare-* belong to --initializer slabs; "
                         "use --lattice-* for the bimodal crystal")
    parameters = {name: (getattr(args, "lattice_" + name)
                        if getattr(args, "lattice_" + name) is not None else default)
                  for name, default in LATTICE_OPTIONS.items()}
    check_settings(**parameters)
    export = output / f"seed_{args.name}"
    cache = args.lattice_cache.expanduser().resolve() if args.lattice_cache else output / "_lattice_cache"
    updated = {key: value.copy() for key, value in routed.items()}
    updated["none"] = imported_seed_options(updated["none"], export)
    # The seed PSD omits empty bins, including extreme sentinels that the old
    # engine would otherwise include in its supported-radius-range check.
    seed_args = argparse.Namespace(**vars(args))
    seed_args.psd = export / "psd.csv"
    plan = make_plan(root, output, seed_args, updated)
    command = [sys.executable, "-u", str(sources[-1]), "--psd", str(args.psd),
               "--out", str(export), "--cache-dir", str(cache)]
    for name, value in parameters.items():
        if name == "widen":
            if value:
                command.append("--widen")
        elif value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    plan["initializer"] = dict(mode="ordered", builder="bimodal_lattice.py",
        psd=str(args.psd), export=str(export), cache=str(cache),
        log=str(output / f"{args.name}_ordered.log"), command=command,
        settings=parameters, sources={str(path.relative_to(root)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    return plan


def run_lattice_builder(root, specification):
    """Only import a crystal after its saved decimal geometry passes the exact verifier."""
    print(f"Building bimodal crystal seed; progress: {specification['log']}", flush=True)
    with Path(specification["log"]).open("x", encoding="utf-8") as stream:
        subprocess.run(specification["command"], cwd=root, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    export = Path(specification["export"])
    verification = json.loads((export / "verification.json").read_text(encoding="utf-8"))
    if (not verification.get("passed") or
            verification.get("overlapping_pairs_exact") != 0 or
            verification.get("self_image_violations_exact") != 0 or
            not verification.get("original_radii_and_counts_exact") or
            not verification.get("family_assignment_verified") or
            verification.get("family_assignment_mismatches") != 0):
        raise RuntimeError("Crystal seed failed exact verification; no optimization stage started")
    box = json.loads((export / "box.json").read_text(encoding="utf-8"))
    print(f"Crystal seed verified: {100 * box['packing_fraction']:.8f}%; "
          f"first stage imports {export / 'packing_coordinates.csv'}", flush=True)


def run_stage(root, stage):
    """A failed, interrupted or capped stage never triggers the following stage."""
    print(f"Starting {stage['stage']}; progress: {stage['log']}", flush=True)
    with Path(stage["log"]).open("x", encoding="utf-8") as stream:
        subprocess.run(stage["command"], cwd=root, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    with (Path(stage["run"]) / "last_command.json").open(encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("stop_reason") != "density_plateau" or not result.get("convergence_reached"):
        raise RuntimeError(f"Stage {stage['stage']} did not reach its plateau; no next stage started")
    verification_path = result.get("verification")
    if not isinstance(verification_path, str):
        raise RuntimeError(f"Stage {stage['stage']} has no verification report")
    with Path(verification_path).open(encoding="utf-8") as stream:
        verification = json.load(stream)
    if (not verification.get("passed") or
            verification.get("overlapping_pairs_exact") != 0 or
            verification.get("self_image_violations_exact") != 0 or
            not verification.get("original_radii_and_counts_exact")):
        raise RuntimeError(f"Stage {stage['stage']} failed export verification")
    print(f"Stage {stage['stage']} reached its plateau; verified export: {stage['export']}", flush=True)


def main(argv=None):
    options = parser()
    args, solver_options = options.parse_known_args(argv)
    root = Path(__file__).resolve().parent
    args.psd = args.psd.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() if args.output_dir else root
    if not (root / "refine.py").is_file() or not args.psd.is_file():
        raise ValueError("Place stages.py beside refine.py and supply an existing PSD CSV")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        raise ValueError("NAME must contain only letters, numbers, underscores or hyphens")
    routed = route_options(solver_options)
    preliminary = make_plan(root, output, args, routed,
                            allow_generated_seed=args.initializer != "previous")
    mode = args.initializer or ("previous" if preliminary["stages"][0]["settings"]["initial"] else "ordered")
    if mode == "ordered":
        plan = lattice_plan(root, output, args, routed, preliminary)
    elif mode == "slabs":
        if any(value is not None for key, value in vars(args).items() if key.startswith("lattice_")):
            raise ValueError("--lattice-* options require --initializer ordered")
        plan = slab_plan(root, output, args, routed, preliminary)
    else:
        if any(value is not None for key, value in vars(args).items()
               if key.startswith(("ordered_", "prepare_", "lattice_"))):
            raise ValueError("Builder options require --initializer ordered or slabs")
        plan = make_plan(root, output, args, routed)
        plan["initializer"] = dict(mode="previous")
    manifest = output / f"{args.name}_stages.json"
    paths = [manifest, *(Path(stage[key]) for stage in plan["stages"]
                         for key in ("run", "export", "log"))]
    if mode in ("ordered", "slabs"):
        paths.extend(Path(plan["initializer"][key]) for key in ("export", "log"))
    for path in paths:
        if path.exists():
            raise ValueError(f"Already exists: {path}. Use a new --name")
    print(json.dumps(plan, indent=2, default=str), flush=True)
    if args.dry_run:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, default=str)
        stream.write("\n")
    if mode == "ordered":
        run_lattice_builder(root, plan["initializer"])
    elif mode == "slabs":
        run_slab_builder(plan["initializer"])
    for stage in plan["stages"]:
        run_stage(root, stage)
    print(f"Both stages reached their plateaus. Final export: {plan['stages'][-1]['export']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted; no further stage will be started.", file=sys.stderr)
        raise SystemExit(130)
