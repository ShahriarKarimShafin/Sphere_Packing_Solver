# Examples

`demo_psd.csv` and `demo_result` contain a 32-particle two-radius example.
`demo_checkpoint` is a historical four-round hybrid checkpoint from that example.
`latest_200` is a verified 10,000-particle export; `checkpoint_200` is its
historical 200-round run. They are regression fixtures, not universal packing
density targets or proofs of optimality.

`benchmark100_start` and `benchmark1000_start` contain saved starting exports
for optional small benchmarks. `psd/` contains additional PSD CSV examples.

Historic checkpoints retain their original source fingerprint. Inspect them
without modifying the examples:

```bash
python pack.py status --run examples/demo_checkpoint --allow-code-change
python pack.py verify --run examples/demo_checkpoint --allow-code-change
```

For a new experiment, start a branch from a verified export:

```bash
python refine.py new --from-export examples/demo_result --run runs/example --rounds 5 --proposals-per-round 2000 --refinement none --export results/example
```

Tests copy checkpoints into temporary folders before exercising migration and
continuation. Prefer the same practice rather than changing these fixtures.
