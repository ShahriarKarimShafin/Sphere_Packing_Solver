#!/usr/bin/env python3
"""Precompress an imported structured seed before the ordinary packing search.

Radii and site assignments remain fixed. Bounded L-BFGS-B permits local
displacements around affinely moving reference sites; no MC moves or swaps run.
This produces an ordered-derived seed, not a certificate of crystalline order
or global optimality. The existing engine supplies energy, clearance and export
verification. Put this file beside the cleaned solver's pack.py.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

# Import the package first so its numerical-thread settings precede NumPy/SciPy.
from packing.energy import Packing, load_native, minimum_image, wrap
from packing.geometry import clear, export_verified, initial_from_csv, numeric_check, read_psd
import numpy as np
from scipy.optimize import Bounds, minimize


@dataclass(frozen=True)
class Settings:
    steps: int = 100
    increment: float = .002
    min_increment: float = .000005
    cage: float = .25
    iterations: int = 300
    maxfun: int = 10000
    cpu_seconds: float = 120.
    gtol: float = 1e-9
    ftol: float = 1e-18
    curvature: int = 7

    def check(self):
        if self.steps < 0 or self.iterations < 1 or self.maxfun < 1:
            raise ValueError("Require steps >=0 and positive iterations/maxfun")
        values = (self.increment, self.min_increment, self.cage, self.cpu_seconds,
                  self.gtol, self.ftol)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Preparation scales, time and tolerances must be positive and finite")
        if not self.min_increment <= self.increment < 1 or self.cage > .5:
            raise ValueError("Require min-increment <= increment <1 and cage <=0.5")
        if not 1 <= self.curvature <= 30:
            raise ValueError("Require curvature between 1 and 30")


def site_displacement(x, cell, reference_fraction, reference_cell, radii):
    """Componentwise displacement in reference-radius units on the periodic cell."""
    displacement = minimum_image(x - reference_fraction * cell, cell)
    return np.abs(displacement / (cell / reference_cell)) / radii[:, None]


def compact(psd, x, bins, cell, library, settings, reference=None, progress=None, kind="rectangular"):
    """Alternate coarse compression and cage-bounded local relaxation."""
    settings.check()
    if kind not in ("rectangular", "cubic"):
        raise ValueError("Require a rectangular or cubic cell")
    psd.check(bins)
    radii = psd.grid[bins]
    check = numeric_check(x, radii, cell)
    if check["overlapping_pairs"] or check["self_image_violations"]:
        raise ValueError("Preparation requires a feasible periodic seed")
    x, cell = x.copy(), cell.copy()
    reference_x, reference_cell = reference if reference is not None else (x, cell)
    reference_cell = np.asarray(reference_cell).copy()
    reference_fraction = reference_x / reference_cell
    if np.max(site_displacement(x, cell, reference_fraction, reference_cell, radii)) > settings.cage + 1e-10:
        raise ValueError("Initial positions are outside the supplied reference cages")

    initial = best = psd.solid / float(np.prod(cell))
    increment = settings.increment
    records = []
    stalled = 0
    reason = "step_limit"
    interrupted = False
    started = time.process_time()
    try:
        for index in range(settings.steps):
            if time.process_time() - started >= settings.cpu_seconds:
                reason = "cpu_limit"
                break
            target = min(best + increment, 1 - 1e-12)
            scale = max((best / target) ** (1 / 3),
                        float(np.max(2 * radii.max() * (1 + 1e-9) / cell)))
            if scale >= 1 - 16 * np.finfo(float).eps:
                reason = "self_image_floor"
                break
            trial_cell = cell * scale
            sites = reference_fraction * trial_cell
            # The reference boxes deform with the cell. For cage <=0.5, each
            # original cage lies within a ball of radius sqrt(3)*cage*r_i < r_i;
            # nonoverlapping reference spheres therefore have disjoint cages.
            halfwidth = settings.cage * radii[:, None] * (trial_cell / reference_cell)
            lower, upper = sites - halfwidth, sites + halfwidth
            start = sites + minimum_image(x * scale - sites, trial_cell)
            start = np.clip(start, lower, upper)
            problem = Packing(start, radii, trial_cell, library)
            result = minimize(
                problem.fun, start.ravel(), method="L-BFGS-B", jac=True,
                bounds=Bounds(lower.ravel(), upper.ravel()),
                options=dict(maxiter=settings.iterations, maxfun=settings.maxfun,
                             gtol=settings.gtol, ftol=settings.ftol,
                             maxcor=settings.curvature, maxls=30),
            )
            if not np.isfinite(result.x).all():
                raise RuntimeError("Nonfinite precompression result")
            candidate, candidate_cell = clear(
                wrap(result.x.reshape(-1, 3), trial_cell), radii, trial_cell, kind
            )
            displacement = site_displacement(candidate, candidate_cell, reference_fraction,
                                             reference_cell, radii)
            # This allowance concerns cage roundoff, not particle overlap.
            cage_ok = bool(np.max(displacement) <= settings.cage + 1e-10)
            fraction = psd.solid / float(np.prod(candidate_cell))
            old = best
            accepted = cage_ok and fraction > best + 1e-12
            if accepted:
                x, cell, best = candidate, candidate_cell, fraction
            if not accepted or fraction - old < .5 * increment:
                increment = max(settings.min_increment, .5 * increment)
            elif fraction - old > .9 * increment:
                increment = min(settings.increment, 1.15 * increment)
            stalled = stalled + 1 if increment == settings.min_increment and not accepted else 0
            record = dict(
                step=index + 1, fraction=best, trial_fraction=fraction,
                accepted=accepted, cage_ok=cage_ok, next_increment=increment,
                iterations=int(result.nit), evaluations=int(result.nfev),
                energy=float(result.fun), optimizer_success=bool(result.success),
                max_component_site_displacement_in_radii=float(np.max(displacement)),
                search_cpu_seconds=time.process_time() - started,
            )
            records.append(record)
            if progress:
                progress(record)
            if stalled >= 8:
                reason = "no_feasible_improvement"
                break
    except KeyboardInterrupt:
        interrupted = True
        reason = "interrupted"

    summary = dict(
        initial_fraction=initial, final_fraction_float64=best, steps=len(records),
        stop_reason=reason, interrupted=interrupted,
        search_cpu_seconds=time.process_time() - started, settings=asdict(settings),
        max_component_site_displacement_in_radii=float(np.max(
            site_displacement(x, cell, reference_fraction, reference_cell, radii))),
        moves_or_swaps_performed=0, records=records, global_optimality_proved=False,
    )
    return x, cell, summary


def prepare_seed(psd_path, coordinates, box_file, export, *, settings=None,
                 backend="native", reference_csv=None, reference_box=None, progress=None,
                 kind="rectangular"):
    """Load a structured seed and export its best independently verified result."""
    settings = settings or Settings()
    settings.check()
    psd_path, coordinates, export = map(Path, (psd_path, coordinates, export))
    if export.exists():
        raise ValueError(f"Preparation export already exists: {export}")
    if reference_box and not reference_csv:
        raise ValueError("A reference box requires a reference CSV")
    psd = read_psd(psd_path)
    x, cell, bins = initial_from_csv(coordinates, psd, kind, box_file)
    reference = None
    if reference_csv:
        reference_x, reference_cell, reference_bins = initial_from_csv(
            reference_csv, psd, kind, reference_box)
        if not np.array_equal(bins, reference_bins):
            raise ValueError("Reference CSV must use the same particle order and radius assignments")
        reference = reference_x, reference_cell
    library, label = load_native(backend == "python")
    if backend == "native" and library is None:
        raise RuntimeError("Native energy library unavailable; run python build.py")
    started = time.process_time()
    x, cell, summary = compact(psd, x, bins, cell, library, settings, reference, progress, kind)
    export.parent.mkdir(parents=True, exist_ok=True)
    certificate = export_verified(export, psd, x, bins, cell, kind)
    summary.update(
        final_fraction=certificate["packing_fraction"], particles=psd.n,
        radius_bins=len(psd.counts), backend=label, export=str(export.resolve()),
        exact_overlaps=certificate["overlapping_pairs_exact"],
        exact_self_image_violations=certificate["self_image_violations_exact"],
        original_radii_and_counts_preserved=certificate["original_radii_and_counts_exact"],
        source_csv_sha256=hashlib.sha256(coordinates.read_bytes()).hexdigest(),
        psd_sha256=hashlib.sha256(psd_path.read_bytes()).hexdigest(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        overall_cpu_seconds=time.process_time() - started,
        ordering="Fixed particle-to-site assignments with local cage-bounded displacements; crystal order is not independently classified",
    )
    (export / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--psd", required=True, type=Path)
    options.add_argument("--initial", required=True, type=Path)
    options.add_argument("--box-file", type=Path)
    options.add_argument("--export", required=True, type=Path)
    options.add_argument("--reference-csv", type=Path)
    options.add_argument("--reference-box", type=Path)
    options.add_argument("--box", choices=("rectangular", "cubic"), default="rectangular")
    options.add_argument("--backend", choices=("native", "python", "auto"), default="native")
    defaults = Settings()
    for name in ("steps", "increment", "min_increment", "cage", "iterations", "maxfun",
                 "cpu_seconds", "gtol", "ftol", "curvature"):
        options.add_argument("--" + name.replace("_", "-"),
                             type=int if name in ("steps", "iterations", "maxfun", "curvature") else float,
                             default=getattr(defaults, name))
    args = options.parse_args()
    settings = Settings(**{name: getattr(args, name) for name in asdict(defaults)})
    result = prepare_seed(
        args.psd, args.initial, args.box_file, args.export, settings=settings,
        backend=args.backend, reference_csv=args.reference_csv, reference_box=args.reference_box,
        kind=args.box,
        progress=lambda record: print(json.dumps(record), flush=True),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 130 if result["interrupted"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
