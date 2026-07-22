import numpy as np
cimport numpy as np
from sys import exit

cdef extern from "pyacofwi2D.h":
    void fwi_2D(char input_file[200], float *vel_inner, float *record_syn, float *record_obs,
                             float *grad, float *data_mask, int run_fwi, int verbose)
    void forward_2D(char input_file[200], float *vel_inner, float *record_syn, int run_fwi, int verbose)
    void forward_vd_2D(char input_file[200], const float *vel_inner, const float *rho_inner,
                       float *record_syn, int verbose)
    void fwi_vd_2D(char input_file[200], const float *vel_inner, const float *rho_inner,
                   float *record_syn, const float *record_obs, float *grad_velocity,
                   float *grad_density, const float *data_mask, int verbose)
    void vd_spatial_operator_2D(int nx, int nz, int lc, float dx, float dz,
                                const float *rho_inner, const float *pressure_inner,
                                float *operator_inner)


def _variable_density_sizes(paramfile):
    """Read model and data sizes from the legacy alternating-line parameter file."""
    with open(paramfile, "r") as stream:
        lines = [line.strip() for line in stream if line.strip()]
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
    return nx * nz, ns * nt * nr


def fwi(np.ndarray[double, ndim=1, mode="c"] vel not None,
            np.ndarray[float, ndim=1, mode="c"] record_obs not None,
            paramfile = "./input/input_param.txt",
            is_fwi = 1,
            data_mask = None,
            verbose = 0):
    if(np.isnan(vel).any()):
        print('NaN occured in python')
        exit()

    cdef np.ndarray[float, ndim=1,mode="c"] vel_f = np.zeros(vel.shape[0], dtype=np.float32)
    vel_f = np.float32(vel)
    # cdef np.ndarray[float, ndim=1,mode="c"] record_obs_f = np.zeros(record_obs.shape[0], dtype=np.float32)
    # record_obs_f = np.float32(record_obs)

    cdef np.ndarray[float, ndim=1, mode="c"] grad = np.zeros(vel_f.shape[0],dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] record_syn = np.zeros(record_obs.shape[0],dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] data_mask_f = np.ones(record_obs.shape[0],dtype=np.float32)
    if data_mask is not None:
        data_mask_f = np.float32(data_mask)

    fwi_2D(str.encode(paramfile), &vel_f[0], &record_syn[0], &record_obs[0], &grad[0], &data_mask_f[0], is_fwi, verbose)
    record_syn = record_syn * data_mask_f

    return record_syn, grad


def forward(np.ndarray[double, ndim=1, mode="c"] vel not None,
            dim = 1, 
            paramfile = "./input/input_param.txt",
            data_mask = None,
            verbose = 0):
    if(np.isnan(vel).any()):
        print('NaN occured in python')
        exit()

    cdef np.ndarray[float, ndim=1,mode="c"] vel_f = np.zeros(vel.shape[0], dtype=np.float32)
    vel_f = np.float32(vel)

    cdef np.ndarray[float, ndim=1, mode="c"] record_syn = np.zeros(dim, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] data_mask_f = np.ones(dim, dtype=np.float32)
    if data_mask is not None:
        data_mask_f = np.float32(data_mask)

    forward_2D(str.encode(paramfile), &vel_f[0], &record_syn[0], 1, verbose)
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

    forward_vd_2D(str.encode(paramfile), &vel_f[0], &rho_f[0], &record_syn[0], verbose)
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

    fwi_vd_2D(str.encode(paramfile), &vel_f[0], &rho_f[0], &record_syn[0],
              &record_obs[0], &grad_velocity[0], &grad_density[0],
              &data_mask_f[0], verbose)
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
