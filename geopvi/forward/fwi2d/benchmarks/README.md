# Solver timing

From `geopvi/forward/fwi2d`, after building the extension:

```bash
OMP_NUM_THREADS=4 python benchmarks/benchmark_constant_vs_variable.py --boundary cpml --repeats 5
OMP_NUM_THREADS=4 python benchmarks/benchmark_constant_vs_variable.py --boundary liao --repeats 5
```

The script compares constant- and variable-density forward calls and
forward-plus-gradient calls through the Python API. All four are warmed up.
The default workload is (nz,nx)=(110,250), 2,000 steps, five shots,
250 receivers, ten absorbing cells and radius 3.

Options: `--boundary cpml|liao` (default CPML), `--repeats N` (default 3),
`--quick` for a small smoke test, and `--output-dir PATH`
(default `benchmarks/output/`).

Outputs are `parameters_<boundary>.txt` and `benchmark_results_<boundary>.json`,
including timing samples, medians, ratios and configuration. Reusing an output
directory overwrites that boundary's results.

Use the same compute node, CPU allocation and affinity for comparisons.
Recorded OpenMP settings are requested values, not measured team sizes.
The script checks finite outputs; accuracy checks belong to the
[test suite](../tests/README.md).
