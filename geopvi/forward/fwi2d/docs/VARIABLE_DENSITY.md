# Variable-density solver

## Equation and API

$$
p_{tt}=\rho v^2\,\nabla\cdot(\rho^{-1}\nabla p)+v^2s.
$$

The spatial operator uses a staggered gradient/divergence pair and interpolated
face buoyancy (1/rho). Radius `Lc=1..6` gives order `2*Lc` in smooth
media; time stepping is second order. Units and source scaling match the
[constant-density solver](CONSTANT_DENSITY.md).

```python
from geopvi.forward.fwi2d import aco2d

data = aco2d.forward_variable_density(
    v, rho, dim=ns*nt*nr, paramfile="parameters.txt")
data, gv, grho = aco2d.fwi_variable_density(
    v, rho, observed, paramfile="parameters.txt", data_mask=mask)
```

Velocity (m/s) and density (kg/m³) must be positive, contiguous float64 arrays
flattened from (nz,nx), with nx,nz>=3. Observations are contiguous float32
flattened from (ns,nt,nr). Returned data, dJ/dv and dJ/drho are flat float32.

Parameters use the [shared table](CONSTANT_DENSITY.md#shared-parameter-file-format).
Models and acquisition coordinates refer to the physical, unpadded grid.

## Masks and gradients

The objective is `J = 0.5 * sum((mask*(synthetic-observed))**2)`,
without a time-quadrature factor. Finite real mask weights are supported;
omission means all ones. Pass observations unmasked.

**Return convention:** forward modelling returns masked synthetics, but FWI
returns raw synthetics. For FWI, compute J using `(data-observed)*mask`.
This differs from the constant-density FWI return convention.

CPML transposes the pressure and memory updates, buoyancy interpolation and
model extension. See [CPML](CPML.md) for the fixed-damping qualification.
High-order face interpolation can become nonpositive at extreme density
contrasts; CPML rejects this configuration.

Uniform density recovers the constant-density equation, but the two spatial
discretizations differ: expect convergence under refinement, not identical
samples. Local density gradients can be nonzero even at uniform density.

## Wavefield snapshots

For one shot, with zero-based steps in [0,nt):

```python
import numpy as np

steps = np.array([0, 50, 100], dtype=np.int32)
data, forward, adjoint = aco2d.variable_density_wavefield_snapshots(
    v, rho, observed, steps, paramfile="parameters.txt")
# Snapshot shape: (len(steps), nz*nx)
```

Forward snapshots contain pressure after injection. CPML adjoint snapshots
contain unscaled pressure cotangents; Liao uses scaled adjoint pressure.
Their amplitudes are not directly comparable. This API has no mask argument.

`variable_density_spatial_operator(rho, pressure, nx, nz, lc, dx, dz)`
is an interior spatial-operator diagnostic, not a time-propagation API.
Liao stores every pressure step; its `nt_interval` is not a memory control.
