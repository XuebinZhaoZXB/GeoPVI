# Variable-density acoustic 2D module

## Scope

This module implements two-dimensional, constant-grid-spacing acoustic
pressure modelling for spatially varying velocity and density:

\[
p_{tt}=\rho v^2\nabla\cdot\left(\rho^{-1}\nabla p\right)+v^2s.
\]

It provides:

- pressure-data forward modelling;
- adjoint propagation of pressure-data residuals;
- gradients with respect to velocity and density;
- conservative staggered finite differences with `1 <= Lc <= 6`;
- the legacy Liao/mixed absorbing boundary condition.

Pseudospectral modelling is intentionally not implemented.  Both acoustic
modules require `laplace_solver=0` and no longer depend on FFTW.

## Python API

Build the extension in the same Python environment that will call it:

```bash
cd geopvi/forward/fwi2d
python setup.py build_ext --inplace
```

The public entry points are:

```python
record_syn = aco2d.forward_variable_density(
    velocity, density, dim=ns * nt * nr, paramfile="input_params.txt"
)

record_syn, gradient_velocity, gradient_density = \
    aco2d.fwi_variable_density(
        velocity, density, record_obs, paramfile="input_params.txt"
    )
```

Input and output conventions:

- `velocity` and `density` are C-contiguous `float64` arrays flattened from
  `(nz, nx)` in row-major order.  The propagation kernel converts them to
  `float32` internally.
- `record_obs` is a C-contiguous `float32` array flattened from
  `(ns, nt, nr)` in row-major order.
- `record_syn` has shape `(ns*nt*nr,)` and dtype `float32`.
- both gradients have shape `(nz*nx,)`, dtype `float32`, and the same
  row-major ordering as the models.
- reshape data with `record.reshape(ns, nt, nr)` and gradients with
  `gradient.reshape(nz, nx)`.

The model parameters must be finite and strictly positive.  Sources and
receivers use zero-based indices in the physical, unpadded model.

## Units and objective

The expected units are:

| Quantity | Unit |
|---|---|
| velocity `v` | m/s |
| density `rho` | kg/m^3 |
| `dx`, `dz` | m |
| `dt` | s |
| `f0` | Hz |

Pressure and source amplitude may use any mutually consistent scaling.  The
discrete least-squares objective is

\[
J=\frac12\left\|M(d_{syn}-d_{obs})\right\|_2^2,
\]

where `M` is the optional `data_mask`.  No extra `dt` factor is included in
this sample-sum objective.  Consequently the returned quantities are
`dJ/dv` and `dJ/drho` for the exact source, sampling, and mask conventions in
this implementation.

When a mask is needed, pass the unmasked observed data to
`fwi_variable_density` and pass the mask separately.  Internally the adjoint
source contains `M^T M (d_syn-d_obs)`.  Omitting `data_mask` is identical to
using an all-ones mask.  A binary mask can mute selected samples; a non-binary
mask acts as a data weight.

## Parameter file

The legacy alternating comment/value layout is retained.  The 19 values,
in order, are:

| Index | Name | Meaning |
|---:|---|---|
| 0 | `nx` | physical horizontal samples |
| 1 | `nz` | physical vertical samples |
| 2 | `pml0` | Liao/mixed absorbing-layer thickness |
| 3 | `Lc` | staggered finite-difference radius, 1 through 6 |
| 4 | `laplace_solver` | must be 0 |
| 5 | `ns` | number of shots |
| 6 | `nt` | time samples per shot |
| 7 | `ds` | source spacing in grid samples |
| 8 | `ns0` | first source horizontal index |
| 9 | `depths` | source depth index |
| 10 | `depthr` | receiver depth index |
| 11 | `nr` | receivers per shot |
| 12 | `dr` | receiver spacing in grid samples |
| 13 | `nr0` | first receiver horizontal index |
| 14 | `nt_interval` | compatibility field; currently ignored |
| 15 | `dx` | horizontal spacing |
| 16 | `dz` | vertical spacing |
| 17 | `dt` | time step |
| 18 | `f0` | Ricker peak frequency |

Shot `is` is placed at `ns0 + is*ds`.  Every shot uses the same receiver
spread `nr0 + ir*dr`.

`Lc` is the radius of the matching high-order staggered gradient and
interpolation stencil.  `Lc=1` is second order and `Lc=3` is sixth order in
smooth media.  The physical absorbing thickness remains `pml0`; the
conservative operator additionally requires a `2*Lc` internal halo, so the
physical-model offset in the extended arrays is `pml0 + 2*Lc`.

## Discretisation and gradients

With buoyancy `b=1/rho`, the spatial operator is

\[
L_hp=-G_x^TB_xG_xp-G_z^TB_zG_zp,
\]

where `G` is the staggered pressure-gradient operator and `B` contains the
interpolated face buoyancy.  The matching gradient and divergence preserve
the symmetric, negative-semidefinite structure of
`div((1/rho) grad(p))` for positive density.

The interior time update is

\[
p^{n+1}=2p^n-p^{n-1}+\Delta t^2\rho v^2 L_hp^n,
\]

followed by the legacy source injection `dt^2*v^2*s` and the Liao/mixed
boundary update.  The adjoint uses the same propagation operator and injects
the weighted pressure residual in reverse time.

Using the code's adjoint and source convention, the accumulated gradient
terms are equivalent to

\[
g_v=\sum_n \frac{2\lambda^n p_{tt}^n}{\rho v^3},
\]

and

\[
g_\rho=\sum_n\left[
\frac{\lambda^n p_{tt}^n}{\rho^2v^2}
-\frac{\lambda^n s^n}{\rho^2}
+\frac{\nabla\lambda^n\cdot\nabla p^n}{\rho^2}
\right],
\]

with the density face term evaluated by the exact transpose of the
high-order buoyancy interpolation.  In the optimized implementation,
`p_tt` is recovered from the saved three-level pressure sequence, forward
face derivatives are calculated once, and adjoint face fluxes are reused by
the following reverse-time step.  These are algebraic reuse optimizations;
they do not change the discrete forward equation or gradient definition.

## Boundary condition and stability

The legacy `abc_for_p()` Liao/mixed boundary routine is applied after every
interior update in both forward and adjoint propagation.  It is not replaced
by a sponge or a conventional PML.

The code rejects a model when

\[
v_{max}\,dt\sqrt{dx^{-2}+dz^{-2}}\ge 1.
\]

This is a necessary runtime safety check, not a guarantee that every chosen
high-order configuration is optimally sampled.  The user should also choose
`dx`, `dz`, `dt`, and `f0` with adequate points per minimum wavelength.

## Memory

The gradient routine currently stores every padded forward pressure time
step.  Its dominant allocation is approximately

```text
4 * nt * (nx + 2*pml) * (nz + 2*pml) bytes
```

with `pml = pml0 + 2*Lc`, plus several two-dimensional work arrays.
`nt_interval` remains in the parameter file for compatibility but is not yet
used for checkpointing.  Checkpointing can be added later if memory becomes a
limitation.

## Validation

Run the complete automated suite after building:

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests
```

The tests cover:

- finite, stable propagation for every `Lc=1,...,6`;
- constant-density reduction and comparison with the legacy FDM;
- convergence under grid refinement;
- a manufactured solution for the variable-density spatial operator;
- density-interface reflection polarity and impedance trend;
- zero-residual/zero-gradient behaviour;
- finite-difference and Taylor tests for both model gradients;
- joint-gradient tests for every supported stencil radius.

The final large validation used a `110 x 250` model, `nt=2000`, five shots,
250 receivers per shot, `pml0=10`, and `Lc=3`.  Relative changes from the
pre-optimization reference were `4.44e-7` for the velocity gradient and
`1.28e-6` for the density gradient; synthetic pressure was bitwise identical.

On the same two-thread workload, the median timings over three repeats were:

| Calculation | Legacy constant density | Variable density |
|---|---:|---:|
| forward | 3.010 s | 2.436 s |
| forward + adjoint + gradients | 7.038 s | 7.626 s |

The variable-density call returns both velocity and density gradients and was
8.35% slower than the legacy single-gradient call in this benchmark.

## Reproducible example

See `../demos/variable_density_demo.py`.  It generates
a small layered velocity/density model in memory, models observed pressure,
evaluates a uniform starting model, and writes synthetic data and both
gradients to an ignored `output/` directory.
