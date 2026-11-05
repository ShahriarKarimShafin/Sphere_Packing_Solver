# Periodic sphere packing

A serial packing optimizer for a fixed particle-size distribution (PSD) in a
rectangular or cubic cell, periodic in all three directions. Radii and counts
are preserved; particle positions, radius assignments to sites, and cell lengths
are optimized. Cell shear is not included. This is a heuristic search, not a
certificate of the global maximum or of mechanical jamming.

## Workflow

1. Read the PSD and normalize lengths by its count-weighted median radius.
2. Select an FCC-based, random, or imported initial configuration.
3. Propose collision-checked particle translations and radius-assignment swaps.
4. Compress the cell and relax relative overlaps with L-BFGS-B. Optionally apply
   local LP repair and/or full-particle sequential linear programming (SLP).
5. Clear residual numerical overlaps. Keep a new best only after independently
   checking the saved decimal coordinates, original PSD, and periodic images.
6. Commit a recoverable checkpoint and update the density-plateau monitor.

The objective is `phi = sum((4*pi/3)*r_i**3) / (Lx*Ly*Lz)` with fixed radii.
Periodic neighbors use the minimum-image displacement. The relaxation energy is
`0.5 * sum(max(1 - d_ij/(r_i+r_j), 0)**2)` over relevant pairs. LP refinement
uses bounded translations and diagonal cell strains with conservative contact
constraints; candidate volume reductions are checked after clearance.
The search uses float64 arithmetic. Exact rational/integer checks apply to the
**saved decimal export**, not to the floating-point optimizer or LP solution.

## Files

| File | Purpose |
|---|---|
| `pack.py` | Main command-line entry point with ordinary engine defaults. |
| `refine.py` | Same engine with the original longer-solve presets; explicit flags override them. |
| `packing/cli.py` | Argument parsing, run lifecycle, and stopping conditions. |
| `packing/search.py` | Initialization, dependent rounds, compression schedule, and best acceptance. |
| `packing/state.py` | Atomic checkpoints, source hashes, locks, pruning, and exports. |
| `packing/geometry.py` | PSD handling, periodic clearance, L-BFGS-B, and initialization. |
| `packing/energy.py` | Periodic neighbor lists and overlap energy/gradient bindings. |
| `packing/moves.py` | Native/Python movement and swap implementations. |
| `packing/lp.py` | Sparse full SLP and local bottleneck repair. |
| `packing/check.py` | Independent exact verification of exported coordinates and PSD. |
| `packing/stop.py` | Checkpointed density-plateau criterion. |
| `native/energy.c`, `native/moves.c` | Serial float64 kernels for energy/gradient and movement proposals. |
| `build.py` | Compile the C kernels; GCC or Clang, without fast-math. |
| `multi.py` | Independent multistart campaign, executed serially. |
| `progress.py`, `recover.py` | Read saved progress or explicitly recover a previous checkpoint. |
| `bench.py` | Paired fixed-start density/CPU-budget experiments. |
| `test.py`, `tests/` | Numerical, workflow, and optional baseline regression tests. |

## Setup and compilation

Use Python 3.10 or later, NumPy, and SciPy. A C99 compiler is needed for the
native backend; C++ and CMake are not required. Python fallback is available
with `--backend python`, but is slower for large proposal budgets.

```bash
python -m pip install -r requirements.txt
python build.py --compiler gcc
python test.py --report validation/local.json
```

On Linux the build produces `native/energy.so` and `native/moves.so`; Windows
uses `.dll`, macOS `.dylib`. The included Windows binaries were built during
validation. Rebuild for your own platform. The loader resolves paths relative
to the package, including paths containing spaces.

`requirements-tested.txt` records the NumPy/SciPy versions used for validation.
Different compiler or numerical-library versions may change optimization
trajectories. No packages beyond NumPy/SciPy and the standard library are needed
to run or test the solver.

## Input and a short smoke run

The PSD CSV needs `Radius_m` and `Count` columns. Additional columns such as
`VolumeWeight` are ignored; solid volume is calculated from the radii/counts.
Counts are nonnegative integers and radii are positive finite decimal values
in metres. Put your files anywhere and pass their paths explicitly.

```bash
python pack.py new --psd examples/demo_psd.csv --run runs/demo --rounds 5 --starts 1 --proposals-per-round 2000 --relax-iterations 100 --backend native --export results/demo
```

Run and export folders must be new or empty when created. `--starts` chooses
among initializers; it does not perform that many complete optimization runs.
For independent full runs, use `multi.py` (see `python multi.py --help`).

## New search with automatic stopping

```bash
python -u refine.py new \
  --psd examples/psd/psd4025.csv \
  --run run_psd4025_none_auto \
  --box rectangular/cubic \
  --boundary periodic \
  --starts 10 \
  --seed 2014 \
  --proposals-per-round 30000000 \
  --relax-iterations 5000 \
  --maxfun 1000000 \
  --schedule portfolio \
  --refinement none/repair/slp/hybrid \
  --convergence-window 100 \
  --convergence-blocks 3 \
  --convergence-tolerance 0.00001 \
  --backend native \
  --export result_psd4025_none_auto
```

The default rule stops when each of **three consecutive, non-overlapping
100-round blocks** gains strictly less than **0.0001** in best verified packing
fraction (0.01 percentage points per block). Equality or a larger gain resets
the streak. Interrupted runs retain partial blocks. This measures a density
plateau under the saved settings; it does not prove an optimum.

* Set `--convergence-tolerance 0.000001` for a tighter absolute fraction threshold.
* Add `--rounds 1000` for an upper limit of 1,000 additional rounds.
* Use `--no-convergence --rounds 1000` for a fixed-round search.
* `--cpu-seconds` is a soft CPU-time limit checked between rounds.

`refine.py` retains its original presets: 1,200,000 proposals, 5,000 relaxation
iterations, `maxfun=1000000`, hybrid refinement every two rounds, one SLP step,
10,000-particle and 250,000-pair LP caps, and 120 seconds per LP. The example
above explicitly overrides the proposal budget and refinement mode.
Use `python pack.py new --help` for all engine defaults/options and
`python refine.py new ... --dry-run` to inspect the resolved command.

## Resume, inspect, and export

```bash
python -u refine.py extend --run runs/sample --export results/sample_later
python progress.py --run runs/sample
python pack.py verify --run runs/sample
python pack.py export --run runs/sample --out results/sample_copy
```

`extend` retains the saved optimization settings, including refinement mode.
An already plateaued run does no further rounds unless explicitly reset:

```bash
python -u refine.py extend --run runs/sample --reset-convergence --rounds 100 --export results/sample_extra100
```

Old schema-3 periodic checkpoints remain readable. When continuing one with
this reorganized code, pass `--allow-code-change` once. The geometry, PSD and
checkpoint integrity are still checked; convergence blocks restart at the saved
round because the engine's source fingerprint changed. A changed source file
also changes that fingerprint. Use one writer per run folder.

To change refinement mode, export the current best and start a new branch:

```bash
python -u refine.py new --from-export results/sample --run runs/hybrid --seed 2014 --proposals-per-round 24000000 --refinement hybrid --slp-max-particles 100000 --lp-pair-limit 2000000 --lp-seconds 100 --export results/hybrid
```

`none` uses MC/compression/L-BFGS-B, `repair` adds local LP repair, `slp` adds
full-particle SLP, and `hybrid` enables both LP stages. Full SLP runs only when
particle/pair caps permit it. LP time limits are per solve; inspect skip and
limit diagnostics before interpreting a plateau. Large LPs can dominate runtime.

Exports contain one wrapped center per particle in metres, exact original
radius text/counts, `box.json`, and `verification.json`. Reuse the actual cell
dimensions from `box.json`; sphere extents are not periodic cell bounds.

## Verification and review

See `validation/` for the test summary, paired baseline results and observed
timings. To compare an unmodified previous package yourself:

```bash
python test.py --baseline /path/to/previous_solver --report validation/comparison.json
```

The refactor changes layout, names and documentation, not search algorithms,
tolerances, proposal ordering, native ABI, or file schemas. Tests cover periodic
faces/edges/corners, gradients, native/Python parity, exact exports, resume and
recovery, LP refinement, stopping, and bitwise baseline trajectories. See
`REFACTOR.md` for the old-to-new file map and the limits of that evidence.


