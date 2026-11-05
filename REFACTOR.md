# Refactor scope and review notes

# Reproducibility

`packing/state.py` fingerprints the reorganized executable sources. Moving or
renaming code therefore changes the source hash, even where calculations agree
bitwise. Loading an old run requires explicit `--allow-code-change`. The existing
PSD and file-integrity checks remain active; the saved geometry and global round
index continue, while stopping blocks restart at that index. Do not edit source
files while a search is running.

The native libraries now live in `native/`. Rebuild with `python build.py` rather
than copying old libraries under their old names. There is no added Python
dependency, worker pool, fast-math flag, particle growth, or cell shear.

# Evidence and limits

`validation/summary.json` records 38 passing tests and seven baseline scenarios:
none/SLP/hybrid for rectangular and cubic cells, plus a 10,000-particle imported
packing. The compared arrays, best volume, proposal/optimizer counters, stopping
state and scientific round fields match exactly; timings and source/file paths
are excluded from equality comparisons. Additional low-level checks are recorded
in `validation/native.json` and `validation/structure.json`.

`validation/performance.json` contains three serial, alternating paired repeats
on the same 10,000-particle seed (five rounds, 200,000 proposals per round and
300 relaxation iterations). Its samples support comparable performance on that
fixture, not a universal speedup. No long production-PSD optimization was rerun.

Tests were executed on Windows with Python 3.14.0, NumPy 2.3.4, SciPy 1.16.3,
and GCC. Linux/macOS build paths are retained, but were not runtime-tested here.
Agreement in regression cases is evidence of preservation, not a proof for every
possible input or a proof of global packing optimality.
