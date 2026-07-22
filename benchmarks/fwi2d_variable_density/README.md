# FWI2D variable-density benchmark

This benchmark compares the complete Python calls for the legacy
constant-density and variable-density implementations.  The gradient calls
include forward modelling, saved-wavefield allocation, adjoint propagation,
and gradient accumulation.  The constant-density call returns one model
gradient; the variable-density call returns velocity and density gradients.

```bash
cd geopvi/forward/fwi2d
python setup.py build_ext --inplace
cd ../../..
OMP_NUM_THREADS=2 python benchmarks/fwi2d_variable_density/benchmark.py
```

The workload is `110 x 250`, `nt=2000`, five shots, 250 receivers per shot,
`pml0=10`, and `Lc=3`.  Results are written to the ignored `output/`
directory.  `reference_results.json` records the validated two-thread result
from 22 July 2026; runtime values are hardware and system-load dependent.
