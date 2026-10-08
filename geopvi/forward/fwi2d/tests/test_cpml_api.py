"""Public boundary selector and Python/C error handling."""
import importlib.util
import os
from pathlib import Path
import sysconfig

import numpy as np
import pytest

from test_cpml import parameters


@pytest.fixture(scope='module')
def aco2d():
    here = Path(__file__).resolve().parents[1]
    path = os.environ.get('GEOPVI_TEST_ACO2D', str(here/('aco2d'+sysconfig.get_config_var('EXT_SUFFIX'))))
    spec = importlib.util.spec_from_file_location('aco2d', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_cpml_forward_gradient_and_mode_switch(aco2d, tmp_path):
    c = parameters(tmp_path/'cpml.txt')
    l = parameters(tmp_path/'liao.txt', boundary=0)
    velocity = np.full(19*17, 2000., dtype=np.float64)
    options = dict(dim=c['nt']*c['nr'], paramfile=str(c['path']))
    observed = aco2d.forward(velocity+10, **options)
    data, gradient = aco2d.fwi(velocity, observed, paramfile=str(c['path']))
    np.testing.assert_array_equal(data, aco2d.forward(velocity, **options))
    assert np.all(np.isfinite(gradient))
    aco2d.forward(velocity, dim=c['nt']*c['nr'], paramfile=str(l['path']))
    np.testing.assert_array_equal(data, aco2d.forward(velocity, **options))


@pytest.mark.parametrize('variable', [False, True])
def test_nineteen_pairs_default_to_cpml(aco2d, tmp_path, variable):
    default = parameters(tmp_path/'default.txt', boundary=None)
    explicit = parameters(tmp_path/'cpml.txt', boundary=1)
    velocity = np.linspace(1900., 2300., 19*17)
    models = (velocity, np.full_like(velocity, 1800.)) if variable else (velocity,)
    forward = aco2d.forward_variable_density if variable else aco2d.forward
    fwi = aco2d.fwi_variable_density if variable else aco2d.fwi
    size = default['ns']*default['nt']*default['nr']
    data = forward(*models, dim=size, paramfile=str(explicit['path']))
    np.testing.assert_array_equal(data, forward(*models, dim=size, paramfile=str(default['path'])))
    a = fwi(*models, data*.95, paramfile=str(default['path']))
    b = fwi(*models, data*.95, paramfile=str(explicit['path']))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    if variable:
        steps = np.array([0, 80, 240], dtype=np.int32)
        a = aco2d.variable_density_wavefield_snapshots(*models, data*.95, steps,
                                                       paramfile=str(default['path']))
        b = aco2d.variable_density_wavefield_snapshots(*models, data*.95, steps,
                                                       paramfile=str(explicit['path']))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('variable', [False, True])
@pytest.mark.parametrize('change', [dict(width=0), dict(interval=2)])
def test_default_cpml_enforces_cpml_contract(aco2d, tmp_path, variable, change):
    c = parameters(tmp_path/'default_bad.txt', boundary=None, **change)
    velocity = np.full(19*17, 2000.)
    models = (velocity, velocity) if variable else (velocity,)
    forward = aco2d.forward_variable_density if variable else aco2d.forward
    with pytest.raises(ValueError, match='CPML requires'):
        forward(*models, dim=c['nt']*c['nr'], paramfile=str(c['path']))


@pytest.mark.parametrize('change', [dict(boundary=2), dict(interval=2), dict(width=0),
                                    dict(radius=2), dict(sx=100), dict(dt=.01)])
def test_invalid_cpml_parameters_fail(aco2d, tmp_path, change):
    c = parameters(tmp_path/'bad.txt', **change)
    with pytest.raises((ValueError, RuntimeError)):
        aco2d.forward(np.full(19*17, 2000.), dim=c['nt']*c['nr'], paramfile=str(c['path']))


def test_cpml_rejects_unsupported_modes_and_bad_buffers(aco2d, tmp_path):
    c = parameters(tmp_path/'cpml.txt')
    velocity = np.full(19*17, 2000.)
    observed = np.zeros(c['nt']*c['nr'], dtype=np.float32)
    with pytest.raises(ValueError):
        aco2d.fwi(velocity, observed, is_fwi=0, paramfile=str(c['path']))
    with pytest.raises(ValueError):
        aco2d.fwi(velocity, observed, data_mask=np.full_like(observed, .5), paramfile=str(c['path']))
    with pytest.raises(ValueError):
        aco2d.forward(velocity, paramfile=str(c['path']))
    data = aco2d.forward_variable_density(velocity, velocity, dim=observed.size, paramfile=str(c['path']))
    assert np.all(np.isfinite(data))


@pytest.mark.parametrize('vmax', [2000., 4700.])
def test_reference_velocity_tracks_each_model(aco2d, tmp_path, capfd, vmax):
    c = parameters(tmp_path/'auto.txt')
    aco2d.forward(np.full(19*17, vmax), dim=c['nt']*c['nr'],
                  paramfile=str(c['path']), verbose=1)
    # Flush the native stdio buffer before inspecting C's diagnostic output.
    import ctypes
    ctypes.CDLL(None).fflush(None)
    assert 'vmax=%g' % vmax in capfd.readouterr().out


@pytest.mark.parametrize('extra', ['--old vref\n4000\n', '--old vref\n4000\n--old R\n0.0001\n'])
def test_removed_external_cpml_options_are_rejected(aco2d, tmp_path, extra):
    c = parameters(tmp_path/'old_options.txt')
    c['path'].write_text(c['path'].read_text()+extra)
    with pytest.raises(ValueError, match='remove old'):
        aco2d.forward(np.full(19*17, 2000.), dim=c['nt']*c['nr'], paramfile=str(c['path']))
