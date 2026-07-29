# Constant-density acoustic 2D module

## Scope

The legacy module solves the two-dimensional constant-density pressure
equation

\[
p_{tt}=v^2\nabla^2p+v^2s.
\]

It provides pressure-data forward modelling and a velocity-gradient routine
for the historical GeoPVI FWI workflow.  It uses centred high-order finite
differences and the original Liao/mixed absorbing boundary condition.
Pseudospectral propagation has been retired, so `laplace_solver` must be `0`.

The implementation is intentionally retained as the legacy reference.  Its
parameter, masking, wavefield-storage, and gradient conventions are described
explicitly below rather than silently changing them to the newer
variable-density conventions.

## Python API

Build the extension from `geopvi/forward/fwi2d`:

```bash
python setup.py build_ext --inplace
```

The direct entry points are:

```python
record_syn = aco2d.forward(
    velocity, dim=ns * nt * nr, paramfile="input_params.txt"
)

record_syn, gradient_velocity = aco2d.fwi(
    velocity, record_obs, paramfile="input_params.txt"
)
```

Array conventions:

- `velocity` is a C-contiguous `float64` array flattened from `(nz, nx)` in
  row-major order and is converted to `float32` internally.
- `record_obs` must be a C-contiguous `float32` array flattened from
  `(ns, nt, nr)`.
- `record_syn` is a flat `float32` array of length `ns*nt*nr`.
- `gradient_velocity` is a flat `float32` array of length `nz*nx`.
- reshape with `record.reshape(ns, nt, nr)` and
  `gradient_velocity.reshape(nz, nx)`.

The legacy wrappers perform fewer size and positivity checks than the
variable-density wrappers.  Callers must ensure that model/data lengths match
the parameter file and that velocity is finite and strictly positive.

## Units, residual, and gradient

Use velocity in m/s, `dx` and `dz` in m, `dt` in s, and `f0` in Hz.  Pressure
and source amplitude may use any mutually consistent scaling.

For normal use, keep `is_fwi=1`.  The C routine forms

\[
r=d_{syn}-d_{obs}
\]

and reverse-time propagates this pressure residual.  During forward
propagation it saves

\[
p_{tt}^n=\frac{p^{n+1}-2p^n+p^{n-1}}{\Delta t^2}
\]

at every `nt_interval`-th sample.  The returned velocity gradient uses

\[
g_v\mathrel{+}=\frac{2}{v^3}p_{tt}\lambda\,nt_{interval}.
\]

This is the gradient of the discrete sample-sum objective

\[
J=\frac12\sum_n(d_{syn}^n-d_{obs}^n)^2,
\]

so it contains no `dt` quadrature factor and has the same sign convention as
`fwi_variable_density`.  `nt_interval` compensates the sampled legacy imaging
condition; `nt_interval=1` evaluates it at every time step.  The higher-level
`geopvi.forward.fwi2d.posterior.Posterior` passes this loss gradient to
PyTorch, which supplies the negative sign when differentiating the log
likelihood `-J/sigma**2`.

`is_fwi=0` is a historical alternate mode that back-propagates the supplied
data directly; it is not used by the standard least-squares demo.

### `data_mask`

The constant-density wrapper treats `data_mask` as a legacy sample/trace mute:

- the residual is multiplied by the mask once before reverse propagation;
- the returned synthetic data are also multiplied by the mask;
- omitting it uses an all-ones mask.

Use a binary mask (`0` or `1`) with this API.  A general non-binary weighting
does not have the same squared-weight semantics as the variable-density
objective, which injects `M^T M r`.  Pass unmasked observations and provide
the binary mask separately.

## Parameter file

The parser expects 19 alternating comment/value pairs:

| Index | Name | Meaning |
|---:|---|---|
| 0 | `nx` | physical horizontal samples |
| 1 | `nz` | physical vertical samples |
| 2 | `pml0` | Liao/mixed absorbing-layer thickness |
| 3 | `Lc` | centred finite-difference radius |
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
| 14 | `nt_interval` | saved-acceleration time interval |
| 15 | `dx` | horizontal grid spacing |
| 16 | `dz` | vertical grid spacing |
| 17 | `dt` | time step |
| 18 | `f0` | Ricker peak frequency |

Shot `is` is at `ns0 + is*ds`; all shots share the receiver spread
`nr0 + ir*dr`.  Indices refer to the physical, unpadded model and are
zero-based.

The supported legacy `Lc` values are `3` through `6`, corresponding to
sixth-, eighth-, tenth-, and twelfth-order centred Laplacians.  Internally the
code increments `Lc` by one because its coefficient array also contains the
central coefficient.  Therefore:

```text
physical stencil radius = input Lc
internal halo argument  = input Lc + 1
physical model offset   = pml0 + input Lc + 1
```

This differs from the variable-density conservative operator, whose physical
offset is `pml0 + 2*Lc`.

## Discretisation and boundary

For input radius `Lc`, the centred approximation is

\[
\nabla_h^2p=\frac{1}{dx^2}\sum_{k=-Lc}^{Lc}a_kp_{i+k,j}
            +\frac{1}{dz^2}\sum_{k=-Lc}^{Lc}a_kp_{i,j+k}.
\]

The time update is

\[
p^{n+1}=2p^n-p^{n-1}+dt^2v^2\nabla_h^2p^n,
\]

followed by source injection `dt^2*v^2*s` and the Liao/mixed boundary update.
The same update is used to propagate the residual backwards.

The Liao coefficient setup uses `dx` for its Courant number on every side.
The established and tested use therefore has `dx=dz`; using unequal spacing
would require a separate review of the boundary coefficients.

## Memory and limitations

For gradient calculation, the dominant saved-wavefield allocation is roughly

```text
4 * ceil(nt / nt_interval) *
    (nx + 2*pml) * (nz + 2*pml) bytes
```

where `pml = pml0 + Lc + 1`.  Unlike the current variable-density routine,
the legacy routine already honours `nt_interval`; increasing it lowers memory
and gradient-correlation cost but also samples the imaging condition less
frequently.

Important limitations:

- only `laplace_solver=0` is supported;
- the wrapper does not perform the variable-density module's comprehensive
  geometry, CFL, length, and positivity validation;
- the Liao setup is intended for `dx=dz`;
- `nt_interval>1` samples rather than exactly recomputes the imaging
  condition, so it is an approximation to the full sample-sum gradient.

## Validation and example

The common regression suite checks the constant-density reduction of the new
operator against this legacy forward solver before boundary arrivals and
under grid refinement.  It also checks the constant-density velocity gradient
against a finite-difference directional derivative of the stated sample-sum
objective.  The performance benchmark runs the complete forward and gradient
calls.

Run the small standalone demo from this directory:

```bash
OMP_NUM_THREADS=2 python demos/constant_density_demo.py
```

It constructs a three-layer true velocity and a uniform starting model,
generates pressure observations, and writes the synthetic data and velocity
gradient to `demos/output/`.
