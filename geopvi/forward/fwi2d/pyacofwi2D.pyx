import numpy as np
cimport numpy as np
from sys import exit


cdef extern from "pyacofwi2D.h":
    int fwi_2D(char input_file[200], float *vel_inner, float *record_syn, float *record_obs,
                             float *grad, float *data_mask, int run_fwi, int verbose)
    int forward_2D(char input_file[200], float *vel_inner, float *record_syn, int run_fwi, int verbose)
    int forward_vd_2D(char input_file[200], const float *vel_inner, const float *rho_inner,
                       float *record_syn, int verbose)
    int fwi_vd_2D(char input_file[200], const float *vel_inner, const float *rho_inner,
                   float *record_syn, const float *record_obs, float *grad_velocity,
                   float *grad_density, const float *data_mask, int verbose)
    void vd_spatial_operator_2D(int nx, int nz, int lc, float dx, float dz,
                                const float *rho_inner, const float *pressure_inner,
                                float *operator_inner)
    int vd_wavefield_snapshots_2D(char input_file[200], const float *vel_inner,
                                   const float *rho_inner, float *record_syn,
                                   const float *record_obs, int snapshot_count,
                                   const int *snapshot_steps, float *forward_snapshots,
                                   float *adjoint_snapshots, int verbose)


def _require_finite_difference(paramfile):
    """Reject the retired pseudo-spectral option before entering C code."""
    with open(paramfile, "r") as stream:
        lines = [line.strip() for line in stream if line.strip()]
    if len(lines) < 10:
        raise ValueError("parameter file is incomplete")
    try:
        laplace_solver = int(lines[9])
    except ValueError:
        raise ValueError("cannot parse laplace_solver from parameter file")
    if laplace_solver != 0:
        raise ValueError("only laplace_solver=0 (finite difference) is supported")


def _boundary_parameters(lines):
    if len(lines) not in (38, 40):
        raise ValueError('expected 19 parameter pairs plus optional boundary_type; remove old cpml_vref/reflection pairs')
    boundary = int(lines[39]) if len(lines) >= 40 else 1
    if boundary not in (0, 1):
        raise ValueError('boundary_type must be 0 (Liao) or 1 (CPML)')
    return boundary


def _constant_density_sizes(paramfile, is_fwi=1):
    _require_finite_difference(paramfile)
    with open(paramfile, 'r') as stream:
        lines = [line.strip() for line in stream if line.strip()]
    boundary = _boundary_parameters(lines)
    values = [lines[2*i+1] for i in range(19)]
    nx, nz, width, radius, solver, ns, nt, ds, sx, sz, rz, nr, dr, rx, interval = [
        int(value) for value in values[:15]]
    dx, dz, dt, f0 = [float(value) for value in values[15:]]
    if min(nx, nz, ns, nt, nr, interval) <= 0 or width < 0:
        raise ValueError('dimensions and nt_interval must be positive; pml0 must be nonnegative')
    if radius not in (3, 4, 5, 6):
        raise ValueError('constant density supports stencil radius Lc=3,4,5,6 only')
    if not np.all(np.isfinite([dx, dz, dt, f0])) or min(dx, dz, dt, f0) <= 0:
        raise ValueError('dx, dz, dt and f0 must be finite and positive')
    if ds < 0 or dr < 1 or sx < 0 or sx+(ns-1)*ds >= nx or rx < 0 or rx+(nr-1)*dr >= nx:
        raise ValueError('source/receiver horizontal coordinates must lie in the physical model')
    if not 0 <= sz < nz or not 0 <= rz < nz:
        raise ValueError('source/receiver depths must lie in the physical model')
    if boundary == 1 and (width < 1 or interval != 1 or is_fwi != 1):
        raise ValueError('CPML requires pml0>=1, nt_interval=1 and is_fwi=1')
    return nx*nz, ns*nt*nr


def _constant_density_mask(data_mask, size):
    result = np.ones(size, dtype=np.float32) if data_mask is None else np.asarray(data_mask, dtype=np.float32)
    if result.ndim != 1 or result.size != size:
        raise ValueError('data_mask must be one-dimensional with length ns*nt*nr')
    if not np.all((result == 0) | (result == 1)):
        raise ValueError('constant-density data_mask must be binary (0 or 1)')
    return np.ascontiguousarray(result)


def _variable_density_sizes(paramfile):
    """Read model and data sizes from the legacy alternating-line parameter file."""
    with open(paramfile, "r") as stream:
        lines = [line.strip() for line in stream if line.strip()]
    boundary = _boundary_parameters(lines)
    if len(lines) < 38:
        raise ValueError("parameter file is incomplete: expected 19 parameter pairs")
    try:
        values = [lines[2 * index + 1] for index in range(19)]
        nx = int(values[0])
        nz = int(values[1])
        lc = int(values[3])
        laplace_solver = int(values[4])
        ns = int(values[5])
        nt = int(values[6])
        nr = int(values[11])
    except (ValueError, IndexError):
        raise ValueError("cannot parse the variable-density parameter file")
    if min(nx, nz, ns, nt, nr) <= 0:
        raise ValueError("model and acquisition dimensions must be positive")
    if laplace_solver != 0:
        raise ValueError(
            "variable-density modelling supports only laplace_solver=0 "
            "(finite difference)"
        )
    if lc < 1 or lc > 6:
        raise ValueError("variable-density finite difference supports 1 <= Lc <= 6")
    width, ds, sx, sz, rz, dr, rx, interval = [int(values[k]) for k in (2, 7, 8, 9, 10, 12, 13, 14)]
    dx, dz, dt, f0 = [float(v) for v in values[15:]]
    if min(nx, nz) < 3 or width < 0 or interval < 1:
        raise ValueError('nx,nz must be >=3, pml0>=0 and nt_interval>=1')
    if not np.all(np.isfinite([dx, dz, dt, f0])) or min(dx, dz, dt, f0) <= 0:
        raise ValueError('dx, dz, dt and f0 must be finite and positive')
    if ds < 0 or dr < 1 or sx < 0 or sx+(ns-1)*ds >= nx or rx < 0 or rx+(nr-1)*dr >= nx:
        raise ValueError('source/receiver horizontal coordinates must lie in the physical model')
    if not 0 <= sz < nz or not 0 <= rz < nz:
        raise ValueError('source/receiver depths must lie in the physical model')
    if boundary == 1 and (width < 1 or interval != 1):
        raise ValueError('variable-density CPML requires pml0>=1 and nt_interval=1')
    return nx * nz, ns * nt * nr


def _validate_vd_models(vel, rho):
    for model in (vel, rho):
        if not np.all(np.isfinite(model)) or np.any(model <= 0) or np.any(model > np.finfo(np.float32).max):
            raise ValueError('velocity and density must be finite, positive and representable in float32')


def _vd_mask(mask, size):
    result = np.ones(size, dtype=np.float32) if mask is None else np.asarray(mask, dtype=np.float32)
    if result.ndim != 1 or result.size != size or not np.all(np.isfinite(result)):
        raise ValueError('data_mask must be finite and one-dimensional with length ns*nt*nr')
    return np.ascontiguousarray(result)


def fwi(np.ndarray[double, ndim=1, mode="c"] vel not None,
            np.ndarray[float, ndim=1, mode="c"] record_obs not None,
            paramfile = "./input/input_param.txt",
            is_fwi = 1,
            data_mask = None,
            verbose = 0):
    model_size, data_size = _constant_density_sizes(paramfile, is_fwi)
    if vel.size != model_size or record_obs.size != data_size:
        raise ValueError('velocity/observation length does not match the parameter file')
    if not np.all(np.isfinite(vel)) or np.any(vel <= 0) or np.any(vel > np.finfo(np.float32).max):
        raise ValueError('velocity must be finite, positive and representable in float32')
    if not np.all(np.isfinite(record_obs)):
        raise ValueError('observations must be finite')

    cdef np.ndarray[float, ndim=1,mode="c"] vel_f = np.zeros(vel.shape[0], dtype=np.float32)
    vel_f = np.float32(vel)
    # cdef np.ndarray[float, ndim=1,mode="c"] record_obs_f = np.zeros(record_obs.shape[0], dtype=np.float32)
    # record_obs_f = np.float32(record_obs)

    cdef np.ndarray[float, ndim=1, mode="c"] grad = np.zeros(vel_f.shape[0],dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] record_syn = np.zeros(record_obs.shape[0],dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] data_mask_f = np.ones(record_obs.shape[0],dtype=np.float32)
    data_mask_f = _constant_density_mask(data_mask, data_size)

    if fwi_2D(str.encode(paramfile), &vel_f[0], &record_syn[0], &record_obs[0], &grad[0], &data_mask_f[0], is_fwi, verbose) != 0:
        raise RuntimeError('constant-density FWI failed; check parameters, CFL and memory')
    if not np.all(np.isfinite(record_syn)) or not np.all(np.isfinite(grad)):
        raise FloatingPointError('nonfinite constant-density wavefield or gradient')
    record_syn = record_syn * data_mask_f

    return record_syn, grad


def forward(np.ndarray[double, ndim=1, mode="c"] vel not None,
            dim = 1, 
            paramfile = "./input/input_param.txt",
            data_mask = None,
            verbose = 0):
    model_size, data_size = _constant_density_sizes(paramfile)
    if vel.size != model_size or dim != data_size:
        raise ValueError('velocity length must equal nx*nz; dim must equal ns*nt*nr')
    if not np.all(np.isfinite(vel)) or np.any(vel <= 0) or np.any(vel > np.finfo(np.float32).max):
        raise ValueError('velocity must be finite, positive and representable in float32')

    cdef np.ndarray[float, ndim=1,mode="c"] vel_f = np.zeros(vel.shape[0], dtype=np.float32)
    vel_f = np.float32(vel)

    cdef np.ndarray[float, ndim=1, mode="c"] record_syn = np.zeros(dim, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] data_mask_f = np.ones(dim, dtype=np.float32)
    data_mask_f = _constant_density_mask(data_mask, data_size)

    if forward_2D(str.encode(paramfile), &vel_f[0], &record_syn[0], 1, verbose) != 0:
        raise RuntimeError('constant-density forward failed; check parameters, CFL and memory')
    if not np.all(np.isfinite(record_syn)):
        raise FloatingPointError('nonfinite constant-density wavefield')
    record_syn = record_syn * data_mask_f

    return record_syn


def forward_variable_density(np.ndarray[double, ndim=1, mode="c"] vel not None,
                             np.ndarray[double, ndim=1, mode="c"] rho not None,
                             dim=1,
                             paramfile="./input/input_param.txt",
                             data_mask=None,
                             verbose=0):
    """Model pressure data for spatially varying velocity and density."""
    model_size, data_size = _variable_density_sizes(paramfile)
    if vel.shape[0] != rho.shape[0]:
        raise ValueError("velocity and density must have the same number of samples")
    if vel.shape[0] != model_size:
        raise ValueError("velocity and density length must equal nx*nz")
    if np.isnan(vel).any() or np.isnan(rho).any():
        raise ValueError("velocity and density must not contain NaN")
    if np.any(vel <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("velocity and density must be strictly positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if dim != data_size:
        raise ValueError("dim must equal ns*nt*nr")

    cdef np.ndarray[float, ndim=1, mode="c"] vel_f = np.asarray(vel, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] rho_f = np.asarray(rho, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] record_syn = np.zeros(dim, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] data_mask_f = np.ones(dim, dtype=np.float32)
    if data_mask is not None:
        data_mask_f = np.asarray(data_mask, dtype=np.float32)
        if data_mask_f.shape[0] != dim:
            raise ValueError("data_mask length must equal dim")

    _validate_vd_models(vel_f, rho_f)
    data_mask_f = _vd_mask(data_mask, data_size)
    if forward_vd_2D(str.encode(paramfile), &vel_f[0], &rho_f[0], &record_syn[0], verbose) != 0:
        raise RuntimeError('variable-density forward failed; check parameters, CFL, face buoyancy and memory')
    if not np.all(np.isfinite(record_syn)):
        raise FloatingPointError('nonfinite variable-density wavefield')
    return record_syn * data_mask_f


def fwi_variable_density(np.ndarray[double, ndim=1, mode="c"] vel not None,
                         np.ndarray[double, ndim=1, mode="c"] rho not None,
                         np.ndarray[float, ndim=1, mode="c"] record_obs not None,
                         paramfile="./input/input_param.txt",
                         data_mask=None,
                         verbose=0):
    """Return synthetic pressure and L2 gradients for velocity and density."""
    model_size, data_size = _variable_density_sizes(paramfile)
    if vel.shape[0] != rho.shape[0]:
        raise ValueError("velocity and density must have the same number of samples")
    if vel.shape[0] != model_size:
        raise ValueError("velocity and density length must equal nx*nz")
    if np.isnan(vel).any() or np.isnan(rho).any():
        raise ValueError("velocity and density must not contain NaN")
    if np.any(vel <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("velocity and density must be strictly positive")
    if record_obs.shape[0] == 0:
        raise ValueError("record_obs must not be empty")
    if record_obs.shape[0] != data_size:
        raise ValueError("record_obs length must equal ns*nt*nr")

    cdef np.ndarray[float, ndim=1, mode="c"] vel_f = np.asarray(vel, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] rho_f = np.asarray(rho, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] record_syn = np.zeros(record_obs.shape[0], dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] grad_velocity = np.zeros(vel.shape[0], dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] grad_density = np.zeros(rho.shape[0], dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] data_mask_f = np.ones(record_obs.shape[0], dtype=np.float32)
    if data_mask is not None:
        data_mask_f = np.asarray(data_mask, dtype=np.float32)
        if data_mask_f.shape[0] != record_obs.shape[0]:
            raise ValueError("data_mask length must equal record_obs length")

    _validate_vd_models(vel_f, rho_f)
    data_mask_f = _vd_mask(data_mask, data_size)
    if not np.all(np.isfinite(record_obs)):
        raise ValueError('observations must be finite')
    if fwi_vd_2D(str.encode(paramfile), &vel_f[0], &rho_f[0], &record_syn[0],
              &record_obs[0], &grad_velocity[0], &grad_density[0],
              &data_mask_f[0], verbose) != 0:
        raise RuntimeError('variable-density FWI failed; check parameters, CFL, face buoyancy and memory')
    if not (np.all(np.isfinite(record_syn)) and np.all(np.isfinite(grad_velocity)) and np.all(np.isfinite(grad_density))):
        raise FloatingPointError('nonfinite variable-density wavefield or gradient')
    return record_syn, grad_velocity, grad_density


def variable_density_spatial_operator(np.ndarray[double, ndim=1, mode="c"] rho not None,
                                      np.ndarray[double, ndim=1, mode="c"] pressure not None,
                                      int nx, int nz, int lc, double dx, double dz):
    """Evaluate the high-order discrete div((1/rho) grad(pressure)) operator.

    This diagnostic routine is intended for convergence and manufactured-
    solution tests; normal forward modelling does not need to call it.
    """
    if nx <= 0 or nz <= 0 or lc < 1 or lc > 6 or dx <= 0.0 or dz <= 0.0:
        raise ValueError("nx, nz, dx, dz and Lc must define a valid finite-difference grid")
    if rho.shape[0] != nx * nz or pressure.shape[0] != nx * nz:
        raise ValueError("rho and pressure length must equal nx*nz")
    if np.isnan(rho).any() or np.isnan(pressure).any() or np.any(rho <= 0.0):
        raise ValueError("rho must be positive and rho/pressure must not contain NaN")

    cdef np.ndarray[float, ndim=1, mode="c"] rho_f = np.asarray(rho, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] pressure_f = np.asarray(pressure, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] operator_f = np.zeros(nx * nz, dtype=np.float32)
    vd_spatial_operator_2D(nx, nz, lc, dx, dz, &rho_f[0], &pressure_f[0], &operator_f[0])
    return operator_f


def variable_density_wavefield_snapshots(np.ndarray[double, ndim=1, mode="c"] vel not None,
                                         np.ndarray[double, ndim=1, mode="c"] rho not None,
                                         np.ndarray[float, ndim=1, mode="c"] record_obs not None,
                                         np.ndarray[np.int32_t, ndim=1, mode="c"] snapshot_steps not None,
                                         paramfile="./input/input_param.txt", verbose=0):
    """Return synthetic data plus physical-grid forward/adjoint snapshots.

    This is a diagnostic interface.  Snapshot indices are zero-based time-step
    indices and the current implementation supports a single source.
    CPML adjoint snapshots are unscaled discrete pressure cotangents, unlike
    the historical Liao scaled-adjoint pressure.  Do not compare amplitudes
    between these two adjoint conventions directly.
    """
    model_size, data_size = _variable_density_sizes(paramfile)
    if vel.shape[0] != model_size or rho.shape[0] != model_size:
        raise ValueError("velocity and density length must equal nx*nz")
    if record_obs.shape[0] != data_size:
        raise ValueError("record_obs length must equal ns*nt*nr")
    if snapshot_steps.shape[0] == 0:
        raise ValueError("snapshot_steps must not be empty")
    if np.any(vel <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("velocity and density must be strictly positive")
    cdef np.ndarray[float, ndim=1, mode="c"] vel_f = np.asarray(vel, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] rho_f = np.asarray(rho, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] synthetic = np.zeros(data_size, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] forward = np.zeros(snapshot_steps.shape[0] * model_size, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] adjoint = np.zeros(snapshot_steps.shape[0] * model_size, dtype=np.float32)
    _validate_vd_models(vel_f, rho_f)
    if not np.all(np.isfinite(record_obs)):
        raise ValueError('observations must be finite')
    if vd_wavefield_snapshots_2D(str.encode(paramfile), &vel_f[0], &rho_f[0], &synthetic[0],
                              &record_obs[0], snapshot_steps.shape[0], <const int*>&snapshot_steps[0],
                              &forward[0], &adjoint[0], verbose) != 0:
        raise RuntimeError('variable-density snapshots failed; check geometry, steps, CFL and memory')
    if not (np.all(np.isfinite(synthetic)) and np.all(np.isfinite(forward)) and np.all(np.isfinite(adjoint))):
        raise FloatingPointError('nonfinite variable-density snapshots')
    return synthetic, forward.reshape((snapshot_steps.shape[0], -1)), adjoint.reshape((snapshot_steps.shape[0], -1))
