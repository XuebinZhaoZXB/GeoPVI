# Absorbing boundaries

Both solvers default to CPML. With 19 comment/value pairs no additional
setting is needed. To select Liao/mixed absorption, append pair 20:

```text
--boundary_type: 0=Liao, 1=CPML
0
```

Use `1` for explicit CPML. Boundary selection also applies to the INI
`boundary_type` key.

## CPML settings

CPML requires `pml0>=1`, `nt_interval=1` and `laplace_solver=0`.
Constant density also requires `is_fwi=1`. Unequal positive dx/dz is supported.
Invalid settings raise errors; there is no automatic fallback to Liao.

CPML applies coordinate stretching through four recursive memory variables,
reset for each shot. Both directions act at corners. The physical interior
uses the ordinary solver operator, including stencil coupling across the
absorbing-layer interface. Models are extended by nearest-edge replication.

Damping is computed internally from the model's maximum velocity, layer
thickness, spacing and source frequency. The coefficient design uses
`kappa=1`, quadratic damping and target reflection `R=1e-4`; there are
no extra user parameters. R is a design value, not a measured error bound.
Choose layer thickness and check absorption for the intended acquisition.

## Gradient and accuracy limits

CPML uses the discrete transpose of pressure and memory updates. Gradients
from the replicated extension are summed back into physical model cells.
This gradient is exact for **fixed damping coefficients**; derivatives through
`max(v)` and the resulting profiles are intentionally omitted. For velocity
finite differences, keep the maximum fixed or freeze profiles in a diagnostic
build. Density has no such profile dependence.

Liao uses an approximate reverse absorber and physical-only accumulation;
it does not provide the same boundary-gradient consistency.

CPML checks an interior CFL bound, not a universal stability guarantee.
Validate time step, spatial resolution and absorption for each problem:
a good absorber does not remove propagation dispersion.
