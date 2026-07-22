# Variable-density acoustic 2D module

This module solves the pressure equation

\[
p_{tt}=\rho v^2\nabla\cdot(\rho^{-1}\nabla p)+v^2s.
\]

The Python entry points are `forward_variable_density(...)` and
`fwi_variable_density(...)`.  Both take flattened, row-major `(nz, nx)`
velocity and density arrays.  The FWI entry point returns synthetic pressure
data, the velocity gradient, and the density gradient for

\[
J=\tfrac12\|M(d_{syn}-d_{obs})\|_2^2.
\]

## Parameter-file compatibility

The legacy alternating-line parameter-file layout is retained.

- `laplace_solver` must be `0`.  Pseudospectral modelling is not implemented
  for either acoustic module; the FFTW-dependent legacy PSM implementation is
  not compiled.
- `Lc` may be from `1` through `6`; it is the radius of the high-order
  staggered gradient/interpolation stencil.  `Lc=1` is second order and
  `Lc=3` is sixth order in smooth media.
- The physical Liao/mixed absorbing-layer thickness remains `pml0` samples.
  The high-order conservative operator requires an additional internal halo of
  `2*Lc`, hence the internal physical-model offset is `pml=pml0+2*Lc`.
- `nt_interval` is retained so existing parameter files parse unchanged, but
  is not currently used by the variable-density FWI routine: it stores every
  forward pressure time step.

The legacy Liao/mixed boundary routine `abc_for_p()` is used after each
interior update in both forward and adjoint propagation.

## Discretisation

With `b=1/rho`, the spatial operator is constructed in conservative form:

\[
L_hp=-G_x^TB_xG_xp-G_z^TB_zG_zp,
\]

where `G` is a matching high-order staggered pressure gradient and `B` is the
high-order interpolated face buoyancy.  This preserves the negative-semidefinite
structure of `div(b grad(p))` for positive density.  The time update is

\[
p^{n+1}=2p^n-p^{n-1}+\Delta t^2\rho v^2L_hp^n,
\]

followed by the legacy source convention `dt^2*v^2*s`.

## Validation

Run the automated suite after building the extension in the Python environment
that will run the inversion:

```bash
python setup.py build_ext --inplace
OMP_NUM_THREADS=2 python -m pytest -q tests
```

The tests cover:

- finite, stable propagation for `Lc=1,...,6`;
- constant-density reduction and comparison against the legacy FDM for
  `Lc=3,...,6`;
- old/new agreement improving under grid refinement;
- a smooth variable-density manufactured solution for the actual spatial
  operator;
- density-interface reflection polarity and impedance-amplitude trend;
- velocity and density finite-difference/Taylor gradient checks.

`variable_density_spatial_operator(...)` is a diagnostic Python function used
by the manufactured-solution test.  It evaluates `div((1/rho)*grad(p))` on a
given grid; normal forward modelling does not need to call it.
