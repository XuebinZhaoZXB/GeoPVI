"""Fixed-coefficient discrete correctness tests, run on a compute node.

The diagnostic double build changes precision only, not the algorithm.  Its
finite differences hold the unique velocity maximum fixed, as required by
the deliberately frozen damping-coefficient derivative convention.
"""
import ctypes
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
from test_cpml import parameters
from test_cpml_api import aco2d

HERE = Path(__file__).resolve().parents[1]


class VdKernel:
    def __init__(self, path, dtype):
        self.dtype = dtype
        self.lib = ctypes.CDLL(str(path))
        ptr = np.ctypeslib.ndpointer(dtype=dtype, flags='C_CONTIGUOUS')
        self.lib.forward_vd_2D.argtypes = [ctypes.c_char_p]+[ptr]*3+[ctypes.c_int]
        self.lib.fwi_vd_2D.argtypes = [ctypes.c_char_p]+[ptr]*7+[ctypes.c_int]
        self.lib.forward_vd_2D.restype = self.lib.fwi_vd_2D.restype = ctypes.c_int
        self.lib.vdcpml_test_threads.argtypes = [ctypes.c_int]
        self.lib.vdcpml_test_threads.restype = ctypes.c_int
        self.lib.vdcpml_test_transpose.argtypes = [ctypes.c_int]
        self.lib.vdcpml_test_transpose.restype = ctypes.c_double
        assert self.lib.vdcpml_test_threads(1) == 1

    def forward(self, v, rho, c):
        out = np.zeros(c['ns']*c['nt']*c['nr'], dtype=self.dtype)
        status = self.lib.forward_vd_2D(str(c['path']).encode(),
                    np.ascontiguousarray(v, dtype=self.dtype),
                    np.ascontiguousarray(rho, dtype=self.dtype), out, 0)
        assert status == 0
        assert np.all(np.isfinite(out))
        return out

    def gradient(self, v, rho, obs, c, mask=None):
        v, rho, obs = [np.ascontiguousarray(a, dtype=self.dtype) for a in (v, rho, obs)]
        mask = np.ones_like(obs) if mask is None else np.ascontiguousarray(mask, dtype=self.dtype)
        out, gv, gr = np.zeros_like(obs), np.zeros_like(v), np.zeros_like(rho)
        assert self.lib.fwi_vd_2D(str(c['path']).encode(), v, rho, out, obs, gv, gr, mask, 0) == 0
        assert np.all(np.isfinite(gv)) and np.all(np.isfinite(gr))
        return out, gv, gr


def build_vd_kernel(dtype, tmp_path_factory):
    folder = tmp_path_factory.mktemp('vdcpml_kernel')
    names = ['fdtd2D_modelling.c', 'fdtd2D_modelling.h', 'cpml2D.c', 'cpml2D.h',
             'vdacoustic2D_modelling.c', 'vdacoustic2D_modelling.h', 'vdcpml2D.c', 'vdcpml2D.h',
             'vdcpml2D_kernels.h', 'cpml2D_kernels.h']
    for name in names+['tests/vdcpml_step_harness.c']:
        text = (HERE/name).read_text()
        if dtype == np.float64:
            text = re.sub(r'\bfloat\b', 'double', text)
            text = text.replace('"%f"', '"%lf"')
            for func in ['pow', 'sqrt', 'exp', 'sin', 'cos', 'fabs']:
                text = text.replace(func+'f(', func+'(')
        (folder/Path(name).name).write_text(text)
    subprocess.run(['gcc', '-std=c99', '-O2', '-fPIC', '-shared', '-fopenmp',
                    str(folder/'fdtd2D_modelling.c'), str(folder/'cpml2D.c'),
                    str(folder/'vdacoustic2D_modelling.c'), str(folder/'vdcpml_step_harness.c'),
                    '-lm', '-o', str(folder/'kernel.so')], check=True)
    return VdKernel(folder/'kernel.so', dtype)


@pytest.fixture(scope='module')
def vd_double(tmp_path_factory):
    return build_vd_kernel(np.float64, tmp_path_factory)


@pytest.fixture(scope='module', params=[np.float64, np.float32], ids=['double', 'float'])
def vd_kernel(request, tmp_path_factory, vd_double):
    return vd_double if request.param == np.float64 else build_vd_kernel(np.float32, tmp_path_factory)


@pytest.mark.parametrize('radius', range(1, 7))
def test_full_state_transpose(vd_kernel, radius, record_property):
    error = vd_kernel.lib.vdcpml_test_transpose(radius)
    record_property('relative_dot_error', error)
    assert error < (2.e-13 if vd_kernel.dtype == np.float64 else 3.e-6), error


def models(c, dtype=np.float64):
    z, x = np.mgrid[:c['nz'], :c['nx']]
    v, rho = 1850.+6*x+8*z, 1500.+4*x+6*z
    v[c['nz']//2, c['nx']//2] = 2600.  # unique fixed damping reference
    return v.astype(dtype), rho.astype(dtype)


@pytest.mark.parametrize('radius', range(1, 7))
def test_velocity_density_gradients_corners_edges_source(vd_kernel, vd_double, tmp_path, radius, record_property):
    c = parameters(tmp_path/'model.txt', radius=radius, ns=2, nt=501,
                   sx=0, sz=0, ds=18, rz=16, nr=19, dr=1)
    v, rho = models(c, vd_kernel.dtype)
    observed = vd_kernel.forward(v+15, rho+25, c)
    mask = np.linspace(.2, 1., observed.size).astype(vd_kernel.dtype)
    data, gv, gr = vd_kernel.gradient(v, rho, observed, c, mask)
    np.testing.assert_array_equal(data, vd_kernel.forward(v, rho, c))
    probes = [(0, 0), (0, 18), (16, 0), (16, 18), (16, 1), (15, 0),
              (16, 17), (15, 18), (1, 1), (0, 9), (8, 0), (6, 7)]
    worst = 0.
    for variable, grad in [(0, gv), (1, gr)]:
        for point in probes:
            # A few single-cell derivatives are below float propagation noise.
            # Use the identical double algorithm as the FD oracle, including
            # float-rounded inputs/observations when checking the float gradient.
            eps = .03
            values = []
            for sign in [-1, 1]:
                vp, rp = v.astype(np.float64), rho.astype(np.float64)
                (vp if variable == 0 else rp)[point] += sign*eps
                residual = (vd_double.forward(vp, rp, c)-observed)*mask
                values.append(.5*np.dot(residual, residual))
            fd = (values[1]-values[0])/(2*eps)
            exact = float(grad[point])
            tolerance = 2.e-5 if vd_kernel.dtype == np.float64 else .01
            # Absolute floor only relative to the largest same-parameter gradient.
            scale = max(abs(fd), abs(exact), np.max(np.abs(grad))*1.e-3)
            worst = max(worst, abs(fd-exact)/scale)
            assert abs(fd-exact)/scale < tolerance, (radius, variable, point, fd, exact)
    record_property('max_scaled_derivative_error', worst)


def test_density_scale_invariance_and_multishot_reset(vd_kernel, tmp_path):
    c = parameters(tmp_path/'multi.txt', ns=2, nt=301)
    v, rho = models(c, vd_kernel.dtype)
    data = vd_kernel.forward(v, rho, c)
    scaled = vd_kernel.forward(v, 2*rho, c)
    np.testing.assert_array_equal(data, scaled)
    obs = vd_kernel.forward(v+10, rho+40, c)
    result = vd_kernel.gradient(v, rho, obs, c)
    assert abs(np.sum(result[2].astype(float)*rho)) < (
        2.e-12 if vd_kernel.dtype == np.float64 else 2.e-5)*np.sum(np.abs(result[2]*rho))
    gv, gr = np.zeros_like(v), np.zeros_like(rho)
    for shot in range(c['ns']):
        one = parameters(tmp_path/('shot%d.txt'%shot), ns=1, nt=c['nt'], sx=c['sx']+shot*c['ds'])
        start, end = shot*c['nt']*c['nr'], (shot+1)*c['nt']*c['nr']
        single = vd_kernel.gradient(v, rho, obs[start:end], one)
        np.testing.assert_array_equal(single[0], data[start:end])
        gv += single[1]; gr += single[2]
    np.testing.assert_allclose(gv, result[1], rtol=2.e-5, atol=np.max(abs(gv))*2.e-7)
    np.testing.assert_allclose(gr, result[2], rtol=2.e-5, atol=np.max(abs(gr))*2.e-7)
    try:
        assert vd_kernel.lib.vdcpml_test_threads(4) == 4
        parallel = vd_kernel.gradient(v, rho, obs, c)
        for a, b in zip(result, parallel):
            np.testing.assert_array_equal(a, b)
    finally:
        vd_kernel.lib.vdcpml_test_threads(1)


def test_public_selector_snapshots_and_failures(aco2d, tmp_path):
    c = parameters(tmp_path/'cpml.txt', nt=151)
    v, rho = [a.ravel() for a in models(c)]
    options = dict(paramfile=str(c['path']))
    data = aco2d.forward_variable_density(v, rho, dim=c['nt']*c['nr'], **options)
    obs = data*.95
    result = aco2d.fwi_variable_density(v, rho, obs, **options)
    np.testing.assert_array_equal(data, result[0])
    snaps = aco2d.variable_density_wavefield_snapshots(v, rho, obs,
                np.array([0, 10, 80, 150], dtype=np.int32), **options)
    np.testing.assert_array_equal(data, snaps[0])
    assert all(np.all(np.isfinite(a)) for a in snaps)
    # Receiver samples and pressure snapshot use the same after-source time index.
    for k, t in enumerate([0, 10, 80, 150]):
        field = snaps[1][k].reshape(c['nz'], c['nx'])
        np.testing.assert_array_equal(field[c['rz'], c['rx']::c['dr']], data.reshape(c['nt'], c['nr'])[t])
    default = parameters(tmp_path/'default.txt', boundary=None, nt=151)
    explicit = parameters(tmp_path/'explicit_cpml.txt', boundary=1, nt=151)
    for ca, cb in [(default, explicit)]:
        a = aco2d.fwi_variable_density(v, rho, obs, paramfile=str(ca['path']))
        b = aco2d.fwi_variable_density(v, rho, obs, paramfile=str(cb['path']))
        for aa, bb in zip(a, b):
            np.testing.assert_array_equal(aa, bb)
    for bad in [dict(width=0), dict(interval=2), dict(dt=.01), dict(sx=-1)]:
        cc = parameters(tmp_path/'bad.txt', nt=151, **bad)
        with pytest.raises((ValueError, RuntimeError)):
            aco2d.forward_variable_density(v, rho, dim=data.size, paramfile=str(cc['path']))
    with pytest.raises(ValueError):
        aco2d.fwi_variable_density(v, rho, obs, data_mask=np.full(data.size, np.nan), **options)


def test_float_joint_directional_derivative(aco2d, tmp_path, record_property):
    c = parameters(tmp_path/'direction.txt', ns=2, nt=401)
    v, rho = [a.ravel() for a in models(c)]
    opt = dict(paramfile=str(c['path']))
    obs = aco2d.forward_variable_density(v+20, rho+30, dim=c['ns']*c['nt']*c['nr'], **opt)
    mask = np.linspace(.1, 1., obs.size).astype(np.float32)
    data, gv, gr = aco2d.fwi_variable_density(v, rho, obs, data_mask=mask, **opt)
    rng = np.random.RandomState(817)
    dv, dr = rng.normal(size=v.size)*20, rng.normal(size=v.size)*20
    dv[np.argmax(v)] = 0.
    exact = gv.astype(float).dot(dv)+gr.astype(float).dot(dr)
    errors = []
    for eps in [.2, .1, .05]:
        objectives = []
        for sign in [-1, 1]:
            pred = aco2d.forward_variable_density(v+sign*eps*dv, rho+sign*eps*dr, dim=obs.size, **opt)
            residual = (pred.astype(float)-obs)*mask
            objectives.append(.5*residual.dot(residual))
        fd = (objectives[1]-objectives[0])/(2*eps)
        errors.append(abs(fd-exact)/max(abs(fd), abs(exact)))
    record_property('float_joint_fd_errors', str(errors))
    assert min(errors) < .005, errors


@pytest.mark.parametrize('layered', [False, True])
def test_basic_absorption_against_enlarged_same_operator(aco2d, tmp_path, layered, record_property):
    # A basic smoke test, not a survey of thickness/frequency/incident angle.
    c = parameters(tmp_path/'small.txt', nx=39, nz=31, sx=19, sz=10, rz=12,
                   nr=9, dr=4, rx=3, nt=601, dx=10., dz=10., f0=25.)
    v = np.full((31, 39), 2000.)
    rho = np.full_like(v, 1500.)
    if layered:
        v[18:] = 2400.
        rho[18:] = 2000.
    small = aco2d.forward_variable_density(v.ravel(), rho.ravel(), dim=601*9, paramfile=str(c['path']))
    pad = 100  # earliest geometric external reflection > record length
    big = parameters(tmp_path/'big.txt', nx=39+2*pad, nz=31+2*pad,
                     sx=19+pad, sz=10+pad, rz=12+pad, nr=9, dr=4, rx=3+pad,
                     nt=601, dx=10., dz=10., f0=25.)
    reference = aco2d.forward_variable_density(np.pad(v, pad, mode='edge').ravel(),
                   np.pad(rho, pad, mode='edge').ravel(), dim=601*9, paramfile=str(big['path']))
    error = np.linalg.norm(small.astype(float)-reference)/np.linalg.norm(reference)
    record_property('relative_trace_l2_error', error)
    assert error < .03, error
