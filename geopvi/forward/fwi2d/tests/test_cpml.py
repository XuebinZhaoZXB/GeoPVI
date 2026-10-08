"""CPML tests compile isolated kernels; no installed extension is overwritten."""
import ctypes
import math
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]


class Kernel:
    def __init__(self, path, dtype):
        self.dtype = dtype
        self.lib = ctypes.CDLL(str(path))
        pointer = np.ctypeslib.ndpointer(dtype=dtype, flags='C_CONTIGUOUS')
        self.lib.forward_2D.argtypes = [ctypes.c_char_p, pointer, pointer, ctypes.c_int, ctypes.c_int]
        self.lib.fwi_2D.argtypes = [ctypes.c_char_p] + [pointer]*5 + [ctypes.c_int]*2
        self.lib.forward_2D.restype = ctypes.c_int
        self.lib.fwi_2D.restype = ctypes.c_int

    def forward(self, model, config):
        data = np.zeros(config['ns']*config['nt']*config['nr'], dtype=self.dtype)
        status = self.lib.forward_2D(str(config['path']).encode(),
                                    np.ascontiguousarray(model, dtype=self.dtype), data, 1, 0)
        assert status == 0
        assert np.all(np.isfinite(data))
        return data

    def gradient(self, model, observed, config, mask=None):
        model = np.ascontiguousarray(model, dtype=self.dtype)
        observed = np.ascontiguousarray(observed, dtype=self.dtype)
        mask = np.ones_like(observed) if mask is None else np.ascontiguousarray(mask, dtype=self.dtype)
        data = np.zeros_like(observed)
        grad = np.zeros_like(model)
        status = self.lib.fwi_2D(str(config['path']).encode(), model, data, observed, grad, mask, 1, 0)
        assert status == 0
        assert np.all(np.isfinite(grad))
        return data, grad


def build_kernel(dtype, tmp_path_factory, fixed_reference=None):
    directory = tmp_path_factory.mktemp('cpml_kernel')
    for name in ('fdtd2D_modelling.c', 'fdtd2D_modelling.h', 'cpml2D.c', 'cpml2D.h',
                 'cpml2D_kernels.h'):
        source = (HERE/name).read_text()
        if fixed_reference is not None and name == 'cpml2D.c':
            # Test-only frozen-coefficient oracle, not a public parameter.
            assert 'reference_velocity = (float)vmax;' in source
            source = source.replace('reference_velocity = (float)vmax;',
                                    'reference_velocity = %s;' % fixed_reference)
        if dtype == np.float64:
            # Diagnostic only: identical discrete algorithm, without float FD noise.
            source = re.sub(r'\bfloat\b', 'double', source)
            source = source.replace('powf(', 'pow(').replace('"%f"', '"%lf"')
        (directory/name).write_text(source)
    harness = (HERE/'tests/cpml_step_harness.c').read_text()
    if dtype == np.float64:
        harness = re.sub(r'\bfloat\b', 'double', harness)
    (directory/'cpml_step_harness.c').write_text(harness)
    library = directory/'kernel.so'
    subprocess.run(['gcc', '-std=c99', '-O2', '-fPIC', '-shared', '-fopenmp',
                    str(directory/'fdtd2D_modelling.c'), str(directory/'cpml_step_harness.c'),
                    '-lm', '-o', str(library)], check=True)
    return Kernel(library, dtype)


@pytest.fixture(scope='module')
def double_kernel(tmp_path_factory):
    return build_kernel(np.float64, tmp_path_factory)


@pytest.fixture(scope='module', params=[np.float64, np.float32], ids=['double', 'float'])
def kernel(request, tmp_path_factory, double_kernel):
    if request.param == np.float64:
        return double_kernel
    return build_kernel(np.float32, tmp_path_factory)


def parameters(path, **overrides):
    c = dict(nx=19, nz=17, width=8, radius=3, ns=1, nt=241, ds=5,
             sx=3, sz=3, rz=2, nr=5, dr=4, rx=0, interval=1,
             dx=10., dz=12., dt=.0008, f0=25., boundary=1)
    c.update(overrides)
    values = [c[k] for k in ('nx', 'nz', 'width', 'radius')]+[0]+[
        c[k] for k in ('ns', 'nt', 'ds', 'sx', 'sz', 'rz', 'nr', 'dr', 'rx',
                       'interval', 'dx', 'dz', 'dt', 'f0')]
    if c['boundary'] is not None:
        values.append(c['boundary'])
    path.write_text(''.join('-- parameter %d\n%s\n' % (i, v) for i, v in enumerate(values)))
    c['path'] = path
    return c


def objective(kernel, velocity, observed, c):
    residual = kernel.forward(velocity, c).astype(np.float64)-observed
    return .5*np.dot(residual, residual)


def test_multithreaded_shots_and_history_match_serial(kernel, tmp_path):
    c = parameters(tmp_path/'threads.txt', nx=23, nz=19, ns=2,
                   sx=0, sz=18, ds=22, rz=0, nr=23, rx=0, dr=1, nt=107)
    z, x = np.mgrid[:c['nz'], :c['nx']]
    model = (1900.+3*x+5*z).astype(kernel.dtype)
    kernel.lib.cpml_test_threads.argtypes = [ctypes.c_int]
    kernel.lib.cpml_test_threads.restype = ctypes.c_int
    assert kernel.lib.cpml_test_threads(1) == 1
    observed = kernel.forward(model+20, c)
    expected = kernel.gradient(model, observed, c)
    try:
        assert kernel.lib.cpml_test_threads(4) == 4
        for repeat in range(2):
            actual = kernel.gradient(model, observed, c)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(kernel.forward(model, c), expected[0])
    finally:
        kernel.lib.cpml_test_threads(1)


@pytest.mark.parametrize('radius', [3, 4, 5, 6])
def test_gradient_at_fixed_maximum_including_four_corners(kernel, double_kernel, tmp_path, radius):
    # Small domain, long record and two shots illuminate all four edges.
    c = parameters(tmp_path/'params.txt', radius=radius, ns=2, nt=501)
    z, x = np.mgrid[:c['nz'], :c['nx']]
    model = (1900.+7*x+11*z).astype(kernel.dtype)
    # The requested gradient freezes damping derivatives. Hold the unique
    # maximum fixed here; the adaptive maximum is tested separately below.
    model[12, 14] = 2600.
    truth = model+20*np.cos(x*.18)*np.sin((z+1)*.17)
    observed = kernel.forward(truth, c)
    data, grad = kernel.gradient(model, observed, c)
    np.testing.assert_array_equal(data, kernel.forward(model, c))
    if kernel.dtype == np.float32:
        # Test the production float adjoint against double with EXACTLY the
        # same rounded model and observations. Tiny pointwise float finite
        # differences can be below forward roundoff; they are not an oracle.
        double_data, reference_grad = double_kernel.gradient(model, observed, c)
        assert np.linalg.norm(grad-reference_grad)/np.linalg.norm(reference_grad) < 5.e-4
        # For pointwise validation use identical residuals at the base model:
        # otherwise cancellation in F_float-d changes tiny point derivatives.
        observed = double_data-(data.astype(np.float64)-observed.astype(np.float64))
        _, reference_grad = double_kernel.gradient(model, observed, c)
        np.testing.assert_allclose(grad, reference_grad, rtol=2.e-3, atol=1.e-7)
    rng = np.random.RandomState(19)
    directions = [rng.normal(size=model.shape)]
    directions[0][12, 14] = 0
    for zz, xx in [(0, 0), (0, c['nx']-1), (c['nz']-1, 0), (c['nz']-1, c['nx']-1),
                   (0, 8), (8, 0), (3, 3), (2, 4), (8, 9)]:
        direction = np.zeros_like(model)
        direction[zz, xx] = 1
        directions.append(direction)
    for direction in directions:
        # Round the actual +/- models to kernel precision BEFORE finding the
        # perturbation. Accounts for quantized float model inputs.
        h = .02
        fd_kernel = double_kernel
        plus = np.asarray(model+h*direction, dtype=kernel.dtype)
        minus = np.asarray(model-h*direction, dtype=kernel.dtype)
        delta = (plus.astype(np.float64)-minus.astype(np.float64))/(2*h)
        expected = np.sum(grad.astype(np.float64)*delta)
        finite_difference = (objective(fd_kernel, plus, observed, c)-
                             objective(fd_kernel, minus, observed, c))/(2*h)
        np.testing.assert_allclose(expected, finite_difference,
                                   rtol=2.e-6 if kernel.dtype == np.float64 else 2.e-3,
                                   atol=1.e-7)
    # A well-resolved float directional finite difference is checked separately.
    if kernel.dtype == np.float32:
        direction = directions[0]
        # Return to the actual observations for the production float FD.
        observed = kernel.forward(truth, c)
        plus = np.asarray(model+direction, dtype=np.float32)
        minus = np.asarray(model-direction, dtype=np.float32)
        delta = (plus.astype(np.float64)-minus.astype(np.float64))/2
        finite_difference = (objective(kernel, plus, observed, c)-objective(kernel, minus, observed, c))/2
        np.testing.assert_allclose(np.sum(grad*delta), finite_difference, rtol=3.e-3, atol=1.e-6)


def test_cpml_default_and_explicit_identical(kernel, tmp_path):
    default = parameters(tmp_path/'default.txt', boundary=None, dx=10, dz=10)
    explicit = parameters(tmp_path/'explicit.txt', boundary=1, dx=10, dz=10)
    model = np.full((17, 19), 2000., dtype=kernel.dtype)
    obs = kernel.forward(model+10, default)
    d0, g0 = kernel.gradient(model, obs, default)
    d1, g1 = kernel.gradient(model, obs, explicit)
    np.testing.assert_array_equal(d0, d1)
    np.testing.assert_array_equal(g0, g1)


def test_shot_reset_mask_and_repeat(kernel, tmp_path):
    c = parameters(tmp_path/'multi.txt', ns=2)
    model = np.full((17, 19), 2100., dtype=kernel.dtype)
    observed = kernel.forward(model+25, c)
    mask = (np.arange(observed.size) % 3 != 0).astype(kernel.dtype)
    data, grad = kernel.gradient(model, observed, c, mask)
    repeated_data, repeated_grad = kernel.gradient(model, observed, c, mask)
    np.testing.assert_array_equal(data, repeated_data)
    np.testing.assert_array_equal(grad, repeated_grad)
    pieces, grads = [], []
    for shot in range(2):
        s = parameters(tmp_path/('shot%d.txt' % shot), sx=3+shot*5)
        d, g = kernel.gradient(model, observed.reshape(2, -1)[shot], s, mask.reshape(2, -1)[shot])
        pieces.append(d); grads.append(g)
    np.testing.assert_array_equal(data, np.concatenate(pieces))
    np.testing.assert_allclose(grad, sum(grads), rtol=2.e-6, atol=1.e-10)
    _, zero = kernel.gradient(model, data, c)
    np.testing.assert_array_equal(zero, np.zeros_like(zero))


def test_absorption_against_large_domain(kernel, tmp_path):
    # Same geometry in a larger homogeneous medium: no outer reflection can
    # return during this record. Compare ALL samples, not just an end snapshot.
    common = dict(nx=51, nz=51, sx=25, sz=25, rz=25, nr=3, dr=10, rx=15,
                  dx=10., dz=10., dt=.001, f0=25., nt=650)
    c = parameters(tmp_path/'small.txt', width=16, **common)
    r = parameters(tmp_path/'large.txt', nx=171, nz=171, sx=85, sz=85,
                   rz=85, nr=3, dr=10, rx=75, dx=10., dz=10., dt=.001,
                   f0=25., nt=650, width=16)
    data = kernel.forward(np.full((51, 51), 2000.), c)
    reference = kernel.forward(np.full((171, 171), 2000.), r)
    relative = np.linalg.norm(data-reference)/np.linalg.norm(reference)
    assert relative < .02, relative


def test_long_record_stays_bounded(kernel, tmp_path):
    c = parameters(tmp_path/'long.txt', nt=4001, width=12, sx=1, sz=1,
                   dx=10., dz=10., dt=.001, f0=20.)
    data = kernel.forward(np.full((17, 19), 2000.), c).reshape(c['nt'], c['nr'])
    assert np.max(np.abs(data[-500:])) < .02*np.max(np.abs(data))


@pytest.mark.parametrize('radius', [3, 4, 5, 6])
@pytest.mark.parametrize('geometry', [(19, 17, 8), (1, 3, 1), (3, 1, 8)])
def test_state_transpose_interface_coupling_and_deep_laplacian(kernel, radius, geometry):
    # Tiny physical domains exercise overlapping left/right correction fringes.
    nx, nz, width = geometry
    halo = radius+1
    offset = width+halo
    shape = (nz+2*offset, nx+2*offset)
    pointer = np.ctypeslib.ndpointer(dtype=kernel.dtype, flags='C_CONTIGUOUS')
    function = kernel.lib.cpml_test_step
    function.argtypes = [ctypes.c_int]*4+[pointer]*3+[ctypes.c_int]
    rng = np.random.RandomState(2029)
    velocity = np.asarray(1800+300*rng.rand(nz, nx), dtype=kernel.dtype)
    inputs = []
    for unused in range(2):
        state = np.asarray(rng.normal(size=(6,)+shape), dtype=kernel.dtype)
        state[:, :halo] = 0; state[:, -halo:] = 0
        state[:, :, :halo] = 0; state[:, :, -halo:] = 0
        # The state space excludes identically-zero interior auxiliary states.
        state[[2, 4], :, offset:-offset] = 0
        state[[3, 5], offset:-offset, :] = 0
        inputs.append(np.ascontiguousarray(state))
    state, dual = inputs
    propagated = np.zeros_like(state); transposed = np.zeros_like(state)
    assert function(nx, nz, width, radius, velocity, state, propagated, 0) == 0
    assert function(nx, nz, width, radius, velocity, dual, transposed, 1) == 0
    left = np.sum(propagated.astype(np.float64)*dual)
    right = np.sum(state.astype(np.float64)*transposed)
    np.testing.assert_allclose(left, right, rtol=2.e-13 if kernel.dtype == np.float64 else 2.e-6)
    # Independent rational coefficients, not D(D(p)). Exterior memories MUST
    # couple into the physical stencil fringe, but not into the deep interior.
    coefficients = [0.]+[2.*(-1)**(k+1)*math.factorial(radius)**2/
                           (k*k*math.factorial(radius-k)*math.factorial(radius+k))
                           for k in range(1, radius+1)]
    coefficients[0] = -2*sum(coefficients[1:])
    p = state[0].astype(np.float64)
    current = p[offset:-offset, offset:-offset]
    lap = coefficients[0]*current*(1/10.**2+1/13.**2)
    for k in range(1, radius+1):
        lap += coefficients[k]*(p[offset:-offset, offset+k:-offset+k]+
                                 p[offset:-offset, offset-k:-offset-k])/10.**2
        lap += coefficients[k]*(p[offset+k:-offset+k, offset:-offset]+
                                 p[offset-k:-offset-k, offset:-offset])/13.**2
    ordinary = 2*current-state[1, offset:-offset, offset:-offset]+(.0008*velocity.astype(np.float64))**2*lap
    correction = np.zeros_like(current)
    psi_x, psi_z = propagated[2:4].astype(np.float64)
    for k in range(1, radius+1):
        first = (-1)**(k+1)*math.factorial(radius)**2 / (
            k*math.factorial(radius-k)*math.factorial(radius+k))
        correction += first*(psi_x[offset:-offset, offset+k:-offset+k]-
                             psi_x[offset:-offset, offset-k:-offset-k])/10.
        correction += first*(psi_z[offset+k:-offset+k, offset:-offset]-
                             psi_z[offset-k:-offset-k, offset:-offset])/13.
    assert np.max(np.abs(correction)) > 1.e-3
    np.testing.assert_array_equal(correction[radius:-radius, radius:-radius], 0.)
    expected = ordinary+(.0008*velocity.astype(np.float64))**2*correction
    np.testing.assert_allclose(propagated[0, offset:-offset, offset:-offset], expected,
                               rtol=2.e-11 if kernel.dtype == np.float64 else 2.e-5,
                               atol=2.e-12 if kernel.dtype == np.float64 else 2.e-6)


def test_shallow_layered_absorption_against_extended_domain(kernel, tmp_path):
    # Regression for the notebook geometry: a source one cell from the top
    # and receivers four cells deep. The old masked interface fails this bound.
    model = np.full((55, 90), 2200., dtype=kernel.dtype)
    model[12:] = 2600.; model[47:] = 3400.
    model[22:37, 13:26] = 2000.; model[22:37, 39:51] = 3200.
    common = dict(width=10, ns=1, nt=1001, nr=90, dr=1, radius=3,
                  dx=20., dz=20., dt=.002, f0=10.)
    small = parameters(tmp_path/'shallow.txt', nx=90, nz=55,
                       sx=45, sz=1, rz=4, rx=0, **common)
    # An outer-boundary round trip cannot return within the 2 s record.
    pad = 200
    large = parameters(tmp_path/'extended.txt', nx=90+2*pad, nz=55+2*pad,
                       sx=45+pad, sz=1+pad, rz=4+pad, rx=pad, **common)
    data = kernel.forward(model, small).astype(np.float64)
    reference = kernel.forward(np.pad(model, pad, mode='edge'), large).astype(np.float64)
    relative = np.linalg.norm(data-reference)/np.linalg.norm(reference)
    assert relative < .0015, relative


def test_adaptive_maximum_with_frozen_coefficient_gradient(double_kernel, tmp_path_factory, tmp_path):
    frozen = build_kernel(np.float64, tmp_path_factory, fixed_reference=2600.)
    c = parameters(tmp_path/'adaptive.txt', nt=501)
    model = np.full((17, 19), 2000., dtype=np.float64)
    model[0, 0] = 2600.
    observed = double_kernel.forward(model+10, c)
    data, grad = double_kernel.gradient(model, observed, c)
    frozen_data, frozen_grad = frozen.gradient(model, observed, c)
    np.testing.assert_array_equal(data, frozen_data)
    np.testing.assert_array_equal(grad, frozen_grad)
    direction = np.zeros_like(model); direction[0, 0] = 1
    h = .02
    frozen_fd = (objective(frozen, model+h*direction, observed, c)-
                 objective(frozen, model-h*direction, observed, c))/(2*h)
    np.testing.assert_allclose(grad[0, 0], frozen_fd, rtol=2.e-6)
    # Changing max(v) really changes the absorber. Do not accidentally test
    # only a hard-coded reference speed, or claim this derivative is included.
    adaptive_plus = double_kernel.forward(model+direction, c)
    frozen_plus = frozen.forward(model+direction, c)
    assert np.linalg.norm(adaptive_plus-frozen_plus) > 1.e-9
