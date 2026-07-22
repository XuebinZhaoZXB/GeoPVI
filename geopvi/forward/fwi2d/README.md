# FWI2D acoustic kernels

This directory contains the C/Cython implementations of the two-dimensional
acoustic pressure kernels:

- legacy constant-density modelling and velocity gradient;
- variable-density modelling and velocity/density gradients.

## Build

```bash
python setup.py build_ext --inplace
```

Build and call the extension with the same Python environment.  Only finite
differences are supported; parameter files must use `laplace_solver=0`.

## Documentation

- [Constant-density formulation](docs/CONSTANT_DENSITY.md)
- [Variable-density formulation](docs/VARIABLE_DENSITY.md)

The documents describe equations, units, parameter-file fields, array
layout, gradient conventions, boundary conditions, memory, and limitations.

## Small kernel demos

```bash
OMP_NUM_THREADS=2 python demos/constant_density_demo.py
OMP_NUM_THREADS=2 python demos/variable_density_demo.py
```

Both demos use the same `51 x 81` acquisition geometry.  They generate their
models in memory and write optional arrays to the ignored `demos/output/`
directory.  See [demos/README.md](demos/README.md).

The repository-level `examples/fwi2d/` directory remains the complete
Bayesian inversion workflow; these module-local demos exercise only the
acoustic kernels.

## Tests and benchmark

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests
OMP_NUM_THREADS=2 python benchmarks/benchmark_constant_vs_variable.py
```

The benchmark compares the complete legacy and variable-density Python calls.
See [benchmarks/README.md](benchmarks/README.md) for its workload and
interpretation.
