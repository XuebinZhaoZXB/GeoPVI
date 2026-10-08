# Constant-density solver

## Equation and API

$$
p_{tt}=v^2\nabla^2p+v^2s.
$$

The solver uses a direct centred Laplacian of radius `Lc=3..6`
(spatial orders 6–12 in smooth media), second-order time stepping and a Ricker
source centred at `1/f0`. Source injection is `dt²*v²*s`; sample n records
pressure after step n and source injection.

```python
from geopvi.forward.fwi2d import aco2d

data = aco2d.forward(v, dim=ns*nt*nr, paramfile="parameters.txt")
data, gv = aco2d.fwi(v, observed, paramfile="parameters.txt", data_mask=mask)
```

Velocity is positive float64, flattened from (nz,nx); observations are float32,
flattened from (ns,nt,nr). Arrays must be contiguous. Returned data and
`dJ/dv` are flat float32. Units are m/s, m, s and Hz.

The objective is `J = 0.5 * sum((mask*(synthetic-observed))**2)`,
without a time-quadrature factor. An omitted mask means all ones.
Only binary masks are supported. Pass unmasked observations; both calls
return **masked synthetics**. To evaluate J from returned data, use
`residual = (data-observed)*mask`.

The constant-density posterior subtracts observations from masked synthetics
without remasking. For fixed mask and sigma, this adds only a model-independent
constant from muted observations; the model gradient is unchanged.

## Shared parameter-file format

Each field is one comment line followed by one value line. The table applies
to both solvers; positions are one-based, while acquisition indices are zero-based.

| Position | Name | Meaning |
|---:|---|---|
| 1 | nx | Physical x samples |
| 2 | nz | Physical z samples |
| 3 | pml0 | Absorbing cells per side, excluding stencil halo |
| 4 | Lc | Stencil radius: constant density 3..6; variable density 1..6 |
| 5 | laplace_solver | 0 (finite differences) |
| 6 | ns | Shots |
| 7 | nt | Time samples per shot |
| 8 | ds | Shot spacing in grid points |
| 9 | ns0 | First shot x index |
| 10 | depths | Shot z index |
| 11 | depthr | Receiver z index |
| 12 | nr | Receivers per shot |
| 13 | dr | Receiver spacing in grid points |
| 14 | nr0 | First receiver x index |
| 15 | nt_interval | 1 for CPML |
| 16 | dx | x spacing, m |
| 17 | dz | z spacing, m |
| 18 | dt | Time step, s |
| 19 | f0 | Ricker peak frequency, Hz |
| 20 (optional) | boundary_type | 0=Liao, 1=CPML; default CPML |

Shot s is at x=`ns0+s*ds`; receiver r at x=`nr0+r*dr`.
All shots share the receiver spread. Supply physical models only; padding is
internal. All acquisition points must lie within the physical grid.

CPML requires `pml0>=1`, `nt_interval=1` and `is_fwi=1`
(the API default). See [boundary settings](CPML.md) for gradient qualifications.
Liao uses dx in its boundary coefficients: use dx=dz. Its `nt_interval>1`
subsamples the gradient imaging condition and is approximate.
