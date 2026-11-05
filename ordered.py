#!/usr/bin/env python3
"""Build and compare exact-PSD FCC-derived starting configurations.

The portfolio includes periodic FCC, size-banded domains, slabs and shifted
interstitial FCC lattices. Original PSD bin identities survive every constructor.
The result is the best tested seed, not a globally densest ordered packing.
"""

import argparse
import hashlib
from dataclasses import asdict, dataclass
from itertools import permutations
import json
import math
from pathlib import Path
import time

from packing.energy import load_native
from packing.geometry import (clear, export_verified, initial_from_csv,
                              make_start, numeric_check, read_psd)
import numpy as np
from prepare import Settings as Preparation, compact
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Settings:
    band_ratios: tuple = (1.05, 1.2, 1.5)
    widths: tuple = (.8, 1., 1.3)
    refine_candidates: int = 2
    seed: int = 2014
    domain_steps: int = 25
    pair_limit: int = 250000
    lp_seconds: float = 10.
    slabs: bool = True
    slab_scan: int = 70
    infill: bool = True
    infill_targets: tuple = (.62, .66, .70, .74)
    infill_phases: int = 12
    infill_restarts: int = 1
    infill_refine: int = 3
    infill_band_ratio: float = 1.05
    infill_max_groups: int = 80
    infill_max_grid_edge: int = 2000
    infill_max_sites: int = 5000000
    infill_seconds: float = 30.

    def check(self):
        if (not self.band_ratios or not self.widths or
                any(not math.isfinite(v) or v <= 1 for v in self.band_ratios) or
                any(not math.isfinite(v) or v <= 0 for v in self.widths)):
            raise ValueError('Band ratios must exceed 1; widths must be positive and finite')
        if self.refine_candidates < 0 or self.domain_steps < 0 or not 0 <= self.seed < 2**64:
            raise ValueError('Require nonnegative candidate count and uint64 seed')
        if self.pair_limit < 1 or not math.isfinite(self.lp_seconds) or self.lp_seconds <= 0:
            raise ValueError('Require positive domain pair limit and LP seconds')
        if (self.slab_scan < 8 or self.infill_phases < 1 or self.infill_restarts < 1 or
                self.infill_refine < 0 or self.infill_max_groups < 1 or
                self.infill_max_grid_edge < 2 or self.infill_max_sites < 1):
            raise ValueError('Invalid slab/infill scan, phase or resource limit')
        if (not self.infill_targets or any(not 0 < v < 1 for v in self.infill_targets) or
                not 1 < self.infill_band_ratio < 10 or
                not math.isfinite(self.infill_seconds) or self.infill_seconds <= 0):
            raise ValueError('Require targets in (0,1), band ratio in (1,10), and finite positive seconds')


def occupied_bins(psd):
    """Sort occupied bin IDs, retaining duplicate radii and zero-row identities."""
    ids = np.flatnonzero(psd.counts)
    return ids[np.argsort(-psd.grid[ids], kind='stable')]


def slab_plan(radii, counts, side, grid_limit, gap=1e-7):
    """Optimize consecutive radius groups within one finite slab construction."""
    dx = math.sqrt(2) * radii * (1 + gap)
    width = np.minimum(np.floor(side / dx), grid_limit).astype(np.int64)
    width -= width % 2
    if width[0] < 2:
        return None
    capacity = width * width // 2
    cumulative = np.r_[0, np.cumsum(counts)]
    seams = np.maximum(0., 2 * radii - dx) + 2 * gap * radii
    cost = np.full(len(radii) + 1, np.inf)
    predecessor = np.zeros(len(cost), dtype=int)
    cost[0] = 0.
    for end in range(1, len(cost)):
        for start in range(end):
            count = int(cumulative[end] - cumulative[start])
            layers = (count + int(capacity[start]) - 1) // int(capacity[start])
            candidate = cost[start] + layers * dx[start] + seams[start]
            if candidate < cost[end]:
                cost[end], predecessor[end] = candidate, start
    groups, end = [], len(radii)
    while end:
        start = int(predecessor[end])
        count = int(cumulative[end] - cumulative[start])
        layers = (count + int(capacity[start]) - 1) // int(capacity[start])
        groups.append((start, end, layers, int(width[start])))
        end = start
    groups.reverse()
    # Retain the supplied constructor's conservative one-segment convention.
    if len(groups) == 1 and groups[0][2] == 1:
        start, end, _, width = groups[0]
        groups = [(start, end, 2, width)]
    length = 0.
    for index, (start, end, layers, width) in enumerate(groups):
        following = groups[(index + 1) % len(groups)][0]
        interface = max(0., radii[start] + radii[following] - .5 * (dx[start] + dx[following]))
        interface += gap * (radii[start] + radii[following])
        length += layers * dx[start] + interface
    return groups, length


def slab_seed(psd, settings, kind):
    """Build variable-radius FCC slabs, preserving the original bin inventory."""
    ids = occupied_bins(psd)
    radii, counts = psd.grid[ids], psd.counts[ids]
    gap = 1e-7
    lower = 2.001 * math.sqrt(2) * radii[0] * (1 + gap)
    upper = max(lower * 2.6, (psd.solid / .65)**(1 / 3) * 2.1)
    best = None
    for side in np.geomspace(lower, upper, settings.slab_scan):
        plan = slab_plan(radii, counts, side, settings.infill_max_grid_edge, gap)
        if plan is not None:
            groups, length = plan
            volume = length * side * side
            if best is None or volume < best[0]:
                best = volume, side, groups
    if best is None:
        raise RuntimeError('No valid slab construction')
    _, side, groups = best
    x, bins = np.empty((psd.n, 3)), np.empty(psd.n, dtype=np.int32)
    row, cursor = 0, 0.
    for index, (start, end, layers, width) in enumerate(groups):
        radius = radii[start]
        dx, dy = math.sqrt(2) * radius * (1 + gap), side / width
        yy, zz = np.indices((width, width)).reshape(2, -1)
        group_bins = np.repeat(ids[start:end], counts[start:end])
        count = remaining = len(group_bins)
        for layer in range(layers):
            mask = (layer + yy + zz) % 2 == 0
            take = min(remaining, int(mask.sum()))
            if not take:
                break
            x[row:row + take, 0] = cursor + (layer + .5) * dx
            x[row:row + take, 1] = (yy[mask][:take] + .5) * dy
            x[row:row + take, 2] = (zz[mask][:take] + .5) * dy
            row, remaining = row + take, remaining - take
        if remaining:
            raise RuntimeError('Incomplete slab construction')
        bins[row - count:row] = group_bins
        following = groups[(index + 1) % len(groups)][0]
        next_dx = math.sqrt(2) * radii[following] * (1 + gap)
        interface = max(0., radius + radii[following] - .5 * (dx + next_dx))
        cursor += layers * dx + interface + gap * (radius + radii[following])
    cell = np.array([cursor, side, side])
    if kind == 'cubic':
        x *= cell.max() / cell
        cell = np.full(3, cell.max())
    psd.check(bins)
    return x, bins, cell


def radius_bands(x, radii, cell):
    """Periodic trees with bounded within-band size ratios for neighbor queries."""
    if not len(radii):
        return []
    labels = np.floor(np.log(radii / radii.min()) / math.log(1.6)).astype(int)
    result = []
    for label in np.unique(labels):
        mask = labels == label
        result.append((cKDTree(x[mask], boxsize=cell), x[mask], radii[mask], float(radii[mask].max())))
    return result


def allowed_sites(sites, radius, bands, cell, gap=1e-7):
    """Reject periodic collisions, including the entire relative clearance buffer."""
    alive = np.ones(len(sites), dtype=bool)
    for tree, old_x, old_r, maximum in bands:
        selected = np.flatnonzero(alive)
        for start in range(0, len(selected), 256):
            ids = selected[start:start + 256]
            near = tree.query_ball_point(sites[ids], (radius + maximum) * (1 + gap), workers=1)
            counts = np.fromiter(map(len, near), dtype=np.int64, count=len(ids))
            if not np.any(counts):
                continue
            a = np.repeat(np.arange(len(ids)), counts)
            b = np.concatenate([np.asarray(row, dtype=int) for row in near if row])
            delta = sites[ids[a]] - old_x[b]
            delta -= cell * np.rint(delta / cell)
            threshold = (radius + old_r[b]) * (1 + gap)
            rejected = np.einsum('ij,ij->i', delta, delta) <= threshold**2
            alive[ids[np.unique(a[rejected])]] = False
    return np.flatnonzero(alive)


def infill_seed(psd, density, seed, settings):
    """Place all particles on shifted FCC lattices, or return a failed attempt.

    Sites within one phase share a periodic even-parity FCC lattice. Later
    phases check all previously placed sites using conservative band maxima.
    A failed/limited attempt never enters candidate selection.
    """
    ids = occupied_bins(psd)
    ratio = settings.infill_band_ratio
    while True:
        groups, start = [], 0
        while start < len(ids):
            end = start + 1
            while end < len(ids) and psd.grid[ids[start]] / psd.grid[ids[end]] <= ratio:
                end += 1
            groups.append(ids[start:end])
            start = end
        if len(groups) <= settings.infill_max_groups:
            break
        ratio *= 1.1
    cell = np.full(3, (psd.solid / density)**(1 / 3))
    rng = np.random.default_rng(seed)
    coordinates, identities = [], []
    started, sites_queried = time.perf_counter(), 0
    for group in groups:
        radius = float(psd.grid[group[0]])
        group_bins = np.repeat(group, psd.counts[group]).astype(np.int32)
        width = int(math.floor(cell[0] / (math.sqrt(2) * radius * (1 + 1e-7))))
        width -= width % 2
        if not 2 <= width <= settings.infill_max_grid_edge:
            return None, dict(reason='grid_limit', group=int(group[0]))
        yz = np.indices((width, width)).reshape(2, -1).T
        old_x = np.vstack(coordinates) if coordinates else np.empty((0, 3))
        old_bins = np.concatenate(identities) if identities else np.empty(0, dtype=int)
        placed, remaining = [], len(group_bins)
        spacing = cell[0] / width
        for phase in range(settings.infill_phases):
            shift = np.zeros(3) if phase == 0 else rng.uniform(size=3)
            current_x = np.vstack([old_x, *placed]) if placed else old_x
            current_r = np.r_[psd.grid[old_bins], np.full(len(group_bins) - remaining, radius)]
            bands = radius_bands(current_x, current_r, cell)
            for layer in range(width):
                if time.perf_counter() - started >= settings.infill_seconds:
                    return None, dict(reason='time_limit', group=int(group[0]), sites=sites_queried)
                active = (layer + yz[:, 0] + yz[:, 1]) % 2 == 0
                plane = yz[active]
                sites_queried += len(plane)
                if sites_queried > settings.infill_max_sites:
                    return None, dict(reason='site_limit', group=int(group[0]), sites=sites_queried)
                sites = np.column_stack((np.full(len(plane), layer + .5 + shift[0]),
                                         plane[:, 0] + .5 + shift[1], plane[:, 1] + .5 + shift[2]))
                sites = (sites * spacing) % cell
                feasible = allowed_sites(sites, radius, bands, cell) if bands else np.arange(len(sites))
                take = min(remaining, len(feasible))
                if take:
                    placed.append(sites[feasible[:take]])
                    remaining -= take
                if not remaining:
                    break
            if not remaining:
                break
        if remaining:
            return None, dict(reason='insertion_failed', group=int(group[0]), unplaced=remaining)
        coordinates.append(np.vstack(placed))
        identities.append(group_bins)
    x, bins = np.vstack(coordinates), np.concatenate(identities)
    psd.check(bins)
    return (x, bins, cell), dict(reason='complete', sites=sites_queried, bands=len(groups))


def infill_candidates(psd, settings, attempts):
    """Finite density/phase search; successful targets do not establish bounds."""
    tried = {}
    def trials(density):
        success = False
        for restart in range(settings.infill_restarts):
            seed = settings.seed + restart + 1000 * int(round(density * 1000))
            result, info = infill_seed(psd, density, seed, settings)
            success = success or result is not None
            attempts.append(dict(method='fcc_infill', target=density, seed=seed,
                                 complete=result is not None, **info))
            if result is not None:
                yield f'fcc_infill_{density:g}_seed_{seed}', *result, None
        tried[density] = success
    for density in sorted(set(settings.infill_targets)):
        yield from trials(density)
    for _ in range(settings.infill_refine):
        passed = sorted(value for value, success in tried.items() if success)
        if not passed:
            break
        lower = passed[-1]
        failed = sorted(value for value, success in tried.items() if not success and value > lower)
        upper = failed[0] if failed else min(.98, lower + .03)
        density = (lower + upper) / 2
        if upper <= lower or density in tried:
            break
        yield from trials(density)


def brick(psd, bins):
    """A compact finite FCC block, padded to contain every complete sphere.

    Integer sites (i,j,k) of even parity have nearest spacing sqrt(2)*h.
    Set h=sqrt(2)*R, so all pair distances are >=2R >=r_i+r_j.
    No radius is rounded up in the exported PSD; R only sets site spacing.
    """
    count = len(bins)
    radius = float(psd.grid[bins].max())
    h = math.sqrt(2) * radius * (1 + 1e-7)
    bound = max(2, math.ceil((2 * count)**(1 / 3)) + 3)
    best = None
    for nx in range(1, bound + 1):
        for ny in range(1, bound + 1):
            nz = max(1, math.ceil(2 * count / (nx * ny)))
            while (nx * ny * nz + (nx % 2) * (ny % 2) * (nz % 2)) // 2 < count:
                nz += 1
            size = (np.array([nx, ny, nz]) - 1) * h + 2 * radius * (1 + 1e-7)
            score = (float(np.prod(size)), float(size.max() / size.min()))
            if best is None or score < best[0]:
                best = score, (nx, ny, nz)
    nx, ny, nz = best[1]
    grid = np.indices((nx, ny, nz)).reshape(3, -1).T
    grid = grid[grid.sum(axis=1) % 2 == 0]
    # Fill the most central sites first; larger particles occupy central sites.
    center = (np.array([nx, ny, nz]) - 1) / 2
    grid = grid[np.argsort(np.sum((grid - center)**2, axis=1), kind='stable')[:count]]
    grid -= grid.min(axis=0)
    bins = bins[np.argsort(-psd.grid[bins], kind='stable')]
    x = grid * h + radius * (1 + 1e-7)
    size = grid.max(axis=0) * h + 2 * radius * (1 + 1e-7)
    return x, bins, size


def domains(psd, ratio):
    """Partition by actual radius, independent of CSV row/family ordering."""
    bins = np.repeat(np.arange(len(psd.counts)), psd.counts)
    bins = bins[np.argsort(psd.grid[bins], kind='stable')]
    result = []
    start = 0
    while start < len(bins):
        limit = psd.grid[bins[start]] * ratio
        end = int(np.searchsorted(psd.grid[bins], limit, side='right'))
        result.append(brick(psd, bins[start:end]))
        start = end
    return result


def assemble(blocks, width_scale):
    """Greedy extreme-point cuboid placement, open in z and bounded in x/y.

    Cuboid interiors never intersect; each sphere is contained in its cuboid.
    This also makes the assembled outer cell safe under periodic repetition.
    The heuristic does not solve cuboid packing globally.
    """
    blocks = sorted(blocks, key=lambda b: -float(np.prod(b[2])))
    volume = sum(float(np.prod(b[2])) for b in blocks)
    edge = volume**(1 / 3) * width_scale
    largest = max(float(b[2].max()) for b in blocks)
    width, depth = max(edge, largest), max(edge, largest)
    placed, points, positions, identities = [], {(0., 0., 0.)}, [], []
    for x, bins, size in blocks:
        best = None
        for perm in permutations(range(3)):
            extent = size[list(perm)]
            for point in points:
                origin = np.asarray(point)
                top = origin + extent
                if top[0] > width + 1e-10 or top[1] > depth + 1e-10:
                    continue
                if any(np.all(origin < other_top) and np.all(top > other_origin)
                       for other_origin, other_top in placed):
                    continue
                score = (float(top[2]), float(origin[2]), float(top[1]), float(top[0]), perm)
                if best is None or score < best[0]:
                    best = score, origin, extent, perm
        if best is None:
            # A new floor above all previous blocks always admits this block.
            height = max((float(t[2]) for _, t in placed), default=0.)
            best = None, np.array([0., 0., height]), size, (0, 1, 2)
        _, origin, extent, perm = best
        top = origin + extent
        placed.append((origin, top))
        positions.append(x[:, list(perm)] + origin)
        identities.append(bins)
        for axis in range(3):
            point = origin.copy()
            point[axis] = top[axis]
            points.add(tuple(point))
        points = {p for p in points if not any(
            np.all(np.asarray(p) >= o) and np.all(np.asarray(p) < t) for o, t in placed)}
    labels = np.concatenate([np.full(len(b), i, dtype=int) for i, b in enumerate(identities)])
    return (np.concatenate(positions), np.concatenate(identities),
            np.max([t for _, t in placed], axis=0), labels)


def compact_domains(x, radii, cell, labels, steps, kind, pair_limit=250000, lp_seconds=10.):
    """Translate rigid FCC domains and shrink their containing orthorhombic cell.

    With n_ij=(x_i-x_j)/d_ij, impose
        n_ij dot (u_i-u_j) >= r_i+r_j-d_ij.
    This supporting-plane constraint implies true non-overlap for listed pairs.
    Query padding 2*sqrt(3)*move covers every pair that can meet in one step.
    Complete sphere containment makes all periodic images safe at cell faces.
    Each LP is local; it neither optimizes the global packing nor shears the cell.
    """
    count = int(labels.max()) + 1
    indices = [np.flatnonzero(labels == group) for group in range(count)]
    history = []
    x, cell = x.copy(), cell.copy()
    for step in range(steps):
        move = .02 * float(cell.min())
        padding = 2 * math.sqrt(3) * move
        trees = [cKDTree(x[ix]) for ix in indices]
        pairs = []
        queried = 0
        for a in range(count):
            ia = indices[a]
            for b in range(a + 1, count):
                ib = indices[b]
                cutoff = float(radii[ia].max() + radii[ib].max()) + padding
                queried += int(trees[a].count_neighbors(trees[b], cutoff))
                if queried > pair_limit:
                    history.append(dict(step=step + 1, accepted=False,
                                        stop_reason='domain_pair_limit', queried_pairs=queried))
                    return x, cell, history
                near = trees[a].sparse_distance_matrix(
                    trees[b], cutoff,
                    output_type='coo_matrix')
                i, j = ia[near.row], ib[near.col]
                mask = near.data <= radii[i] + radii[j] + padding
                if np.any(mask):
                    pairs.append(np.column_stack([i[mask], j[mask]]))
        rows, cols, values, rhs = [], [], [], []
        if pairs:
            pairs = np.concatenate(pairs)
            i, j = pairs.T
            delta = x[i] - x[j]
            distance = np.linalg.norm(delta, axis=1)
            normal = delta / distance[:, None]
            row = np.repeat(np.arange(len(pairs)), 3)
            rows.extend([row, row])
            cols.extend([(3 * labels[i, None] + np.arange(3)).ravel(),
                         (3 * labels[j, None] + np.arange(3)).ravel()])
            values.extend([-normal.ravel(), normal.ravel()])
            rhs.extend(distance - radii[i] - radii[j] - 1e-8 * np.median(radii))
        for group, ix in enumerate(indices):
            low = np.min(x[ix] - radii[ix, None], axis=0)
            high = np.max(x[ix] + radii[ix, None], axis=0)
            for axis in range(3):
                row = len(rhs)
                rows.append(np.array([row]))
                cols.append(np.array([3 * group + axis]))
                values.append(np.array([-1.]))
                rhs.append(low[axis])
                row = len(rhs)
                rows.append(np.array([row, row]))
                cols.append(np.array([3 * group + axis, 3 * count + axis]))
                values.append(np.array([1., -1.]))
                rhs.append(cell[axis] - high[axis])
        matrix = coo_matrix((np.concatenate(values),
                             (np.concatenate(rows), np.concatenate(cols))),
                            shape=(len(rhs), 3 * count + 3)).tocsr()
        objective = np.r_[np.zeros(3 * count), 1 / cell]
        equalities = None
        if kind == 'cubic':
            equalities = np.zeros((2, len(objective)))
            equalities[0, -3:] = [1, -1, 0]
            equalities[1, -3:] = [0, 1, -1]
        solution = linprog(objective, A_ub=matrix, b_ub=rhs,
            A_eq=equalities, b_eq=np.zeros(2) if equalities is not None else None,
            bounds=[(-move, move)] * (3 * count) + [(-.02 * side, .02 * side) for side in cell],
            method='highs', options={'time_limit': lp_seconds, 'primal_feasibility_tolerance': 1e-9})
        if not solution.success:
            break
        candidate = x + solution.x[:3 * count].reshape(count, 3)[labels]
        trial_cell = cell + solution.x[-3:]
        check = numeric_check(candidate, radii, trial_cell)
        if check['overlapping_pairs'] or check['self_image_violations']:
            break
        ratio = float(np.prod(trial_cell) / np.prod(cell))
        if ratio >= 1 - 1e-8:
            break
        x, cell = candidate, trial_cell
        history.append(dict(step=step + 1, volume_ratio=ratio, constraints=len(rhs)))
    return x, cell, history


def generate(psd, settings, kind='rectangular', attempts=None):
    """Yield finite candidates lazily, including a fully periodic FCC reference."""
    settings.check()
    if kind not in ('rectangular', 'cubic'):
        raise ValueError('Require rectangular or cubic cell')
    x, cell, bins = make_start(psd, np.random.default_rng(settings.seed), kind, ordered=True)
    yield 'periodic_fcc', x, bins, cell, None
    for ratio in settings.band_ratios:
        blocks = domains(psd, ratio)
        for width in settings.widths:
            x, bins, cell, labels = assemble(blocks, width)
            if kind == 'cubic':
                cell = np.full(3, cell.max())
            yield f'fcc_domains_ratio_{ratio:g}_width_{width:g}', x, bins, cell, labels
    if settings.slabs:
        x, bins, cell = slab_seed(psd, settings, kind)
        yield 'fcc_slabs', x, bins, cell, None
    if settings.infill:
        yield from infill_candidates(psd, settings, attempts if attempts is not None else [])


def build_seed(psd_path, export, *, settings=Settings(), preparation=Preparation(),
               kind='rectangular', backend='native', initial=None, box_file=None,
               progress=None):
    """Rank all seeds, precompress a bounded shortlist, export the verified best."""
    settings.check()
    preparation.check()
    export = Path(export)
    if export.exists():
        raise ValueError(f'Already exists: {export}')
    psd = read_psd(psd_path)
    started = time.process_time()
    candidates, records, attempts = [], [], []
    def consider(name, x, bins, cell, labels=None):
        psd.check(bins)
        raw_fraction = psd.solid / float(np.prod(cell))
        domain_history = []
        if labels is not None and settings.domain_steps:
            x, cell, domain_history = compact_domains(
                x, psd.grid[bins], cell, labels, settings.domain_steps, kind,
                settings.pair_limit, settings.lp_seconds)
        x, cell = clear(x, psd.grid[bins], cell, kind)
        check = numeric_check(x, psd.grid[bins], cell)
        if check['overlapping_pairs'] or check['self_image_violations']:
            raise RuntimeError(f'Invalid seed: {name}')
        record = dict(candidate=name, fraction=psd.solid / float(np.prod(cell)),
                      raw_fraction=raw_fraction, domain_steps=domain_history,
                      overlaps=0, self_image_violations=0)
        records.append(record)
        if progress:
            progress(record)
        candidates.append((record['fraction'], name, x, bins, cell))
        candidates.sort(key=lambda c: -c[0])
        del candidates[max(1, settings.refine_candidates):]
    for item in generate(psd, settings, kind, attempts):
        consider(*item)
    if initial is not None:
        x, cell, bins = initial_from_csv(initial, psd, kind, box_file)
        consider('imported_template', x, bins, cell)
    best = candidates[0]
    refinements = []
    if settings.refine_candidates:
        library, label = load_native(force_numpy=backend == 'python')
        if backend == 'native' and library is None:
            raise RuntimeError('Native backend unavailable; compile with python build.py')
        for _, name, x, bins, cell in candidates:
            refined_x, refined_cell, report = compact(
                psd, x, bins, cell, library, preparation, kind=kind,
                progress=(lambda row, name=name: progress(dict(candidate=name, preparation=row)))
                if progress else None)
            fraction = psd.solid / float(np.prod(refined_cell))
            refinements.append(dict(candidate=name, fraction=fraction, report=report))
            if fraction > best[0]:
                best = fraction, name, refined_x, bins, refined_cell
            if report['interrupted']:
                raise KeyboardInterrupt
    _, name, x, bins, cell = best
    verification = export_verified(export, psd, x, bins, cell, kind)
    report = dict(psd_fingerprint=psd.fingerprint,
                  sources={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (Path(__file__), Path(__file__).with_name('prepare.py'))},
                  selected=name, packing_fraction=psd.solid / float(np.prod(cell)),
                  particles=psd.n, original_PSD_preserved=True, candidates=records,
                  infill_attempts=attempts,
                  refinements=refinements, settings=asdict(settings),
                  preparation_settings=asdict(preparation), cpu_seconds=time.process_time()-started,
                  structure='FCC-derived; size domains and local site deformation allowed',
                  crystal_order_certified=False, global_optimality_proved=False,
                  verification_passed=verification['passed'])
    (export / 'ordered.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--psd', required=True)
    parser.add_argument('--export', required=True)
    parser.add_argument('--box', choices=['rectangular', 'cubic'], default='rectangular')
    parser.add_argument('--backend', choices=['native', 'python', 'auto'], default='native')
    parser.add_argument('--seed', type=int, default=2014)
    parser.add_argument('--band-ratios', type=float, nargs='+', default=[1.05, 1.2, 1.5])
    parser.add_argument('--widths', type=float, nargs='+', default=[.8, 1., 1.3])
    parser.add_argument('--refine-candidates', type=int, default=2)
    parser.add_argument('--domain-steps', type=int, default=25)
    parser.add_argument('--pair-limit', type=int, default=250000)
    parser.add_argument('--lp-seconds', type=float, default=10.)
    parser.add_argument('--cpu-seconds', type=float, default=120.)
    parser.add_argument('--initial')
    parser.add_argument('--box-file')
    parser.add_argument('--slabs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--slab-scan', type=int, default=70)
    parser.add_argument('--infill', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--infill-targets', type=float, nargs='+', default=[.62, .66, .70, .74])
    parser.add_argument('--infill-phases', type=int, default=12)
    parser.add_argument('--infill-restarts', type=int, default=1)
    parser.add_argument('--infill-refine', type=int, default=3)
    parser.add_argument('--infill-band-ratio', type=float, default=1.05)
    parser.add_argument('--infill-max-groups', type=int, default=80)
    parser.add_argument('--infill-max-grid-edge', type=int, default=2000)
    parser.add_argument('--infill-max-sites', type=int, default=5000000)
    parser.add_argument('--infill-seconds', type=float, default=30.)
    args = parser.parse_args()
    if args.box_file and not args.initial:
        parser.error('--box-file requires --initial')
    report = build_seed(args.psd, args.export,
                        settings=Settings(tuple(args.band_ratios), tuple(args.widths),
                                          args.refine_candidates, args.seed, args.domain_steps,
                                          args.pair_limit, args.lp_seconds,
                                          slabs=args.slabs, slab_scan=args.slab_scan,
                                          infill=args.infill, infill_targets=tuple(args.infill_targets),
                                          infill_phases=args.infill_phases,
                                          infill_restarts=args.infill_restarts,
                                          infill_refine=args.infill_refine,
                                          infill_band_ratio=args.infill_band_ratio,
                                          infill_max_groups=args.infill_max_groups,
                                          infill_max_grid_edge=args.infill_max_grid_edge,
                                          infill_max_sites=args.infill_max_sites,
                                          infill_seconds=args.infill_seconds),
                        preparation=Preparation(cpu_seconds=args.cpu_seconds),
                        kind=args.box, backend=args.backend, initial=args.initial,
                        box_file=args.box_file, progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({key: report[key] for key in ('selected', 'packing_fraction', 'particles')}))


if __name__ == '__main__':
    main()
