# FWI2D acoustic solver

Two-dimensional constant- and variable-density pressure solvers for forward
modelling and full-waveform inversion. The kernels use C/OpenMP, with a
Cython Python interface. Time stepping is second order; spatial derivatives
use high-order finite differences. CPML is the default absorbing boundary;
Liao/mixed absorption is also available.

## Build and run

Requires Python, NumPy, Cython, setuptools and a C compiler with OpenMP.
Posterior classes require PyTorch/SciPy; regression tests also require pytest.

From the repository root:

```bash
python -m pip install -e .
cd geopvi/forward/fwi2d
python setup.py build_ext --inplace
OMP_NUM_THREADS=2 python demos/constant_density_demo.py --no-save
OMP_NUM_THREADS=2 python demos/variable_density_demo.py --no-save
```

The native extension must be built in this module directory using the Python
environment that will run it. Restart Python/Jupyter after rebuilding.
On HPC systems, use a compute node for builds and solver runs.

## Python interface

```python
from geopvi.forward.fwi2d import aco2d

data = aco2d.forward(v, dim=ns*nt*nr, paramfile="parameters.txt")
data, gv = aco2d.fwi(v, observed, paramfile="parameters.txt")
data = aco2d.forward_variable_density(
    v, rho, dim=ns*nt*nr, paramfile="parameters.txt")
data, gv, grho = aco2d.fwi_variable_density(
    v, rho, observed, paramfile="parameters.txt")
```

Models are contiguous float64 arrays flattened from `(nz,nx)`; observations
are contiguous float32 arrays flattened from `(ns,nt,nr)`. Returned data
and gradients are flat float32 arrays. Coordinates refer to the unpadded
physical grid. See the solver pages below for mask conventions.

Parameter files contain 19 comment/value pairs, plus an optional pair 20:
`0=Liao`, `1=CPML`; omission selects CPML.

## Documentation and layout

- [Constant density and shared parameter table](docs/CONSTANT_DENSITY.md)
- [Variable density and snapshots](docs/VARIABLE_DENSITY.md)
- [Boundary settings and gradient limitations](docs/CPML.md)
- [Threading and memory](docs/PERFORMANCE.md)
- [Examples](demos/README.md), [tests](tests/README.md), [benchmarks](benchmarks/README.md)

`fdtd2D_modelling.*` and `vdacoustic2D_modelling.*` provide the two solvers;
`cpml2D.*`, `vdcpml2D.*` and their `*_kernels.h` templates implement CPML.
`pyacofwi2D.pyx/.h` expose the Python API; `setup.py/setup.cfg` build it.
`posterior.py` provides the parameter writer, autograd bridge, `Posterior`
and `PosteriorVariableDensity`; `__init__.py` marks the Python package.

For posterior inputs, active parameters are velocity values or
`[velocity[velocity_mask], density[density_mask]]`. Fixed background values
fill inactive cells. Use a separate, unchanged parameter file per configuration.
The INI writer uses section `[FWI]`, the key `laplace_slover`, and optional
`boundary_type` with the same 0/1 selection.

Generated C, compiled extensions, build directories, caches and example outputs
are not source files and are ignored by Git.
