import importlib.util
from pathlib import Path
import sysconfig

import numpy as np


HERE = Path(__file__).resolve().parents[1]


def _load_extension():
    extension = HERE / ("aco2d" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not extension.exists():
        raise RuntimeError("Build aco2d before running the tests")
    spec = importlib.util.spec_from_file_location("aco2d", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_parameters(path, *, nx=61, nz=61, pml=20, ns=1, nt=240,
                      ds=1, ns0=30, depths=30, depthr=30, nr=1, dr=1,
                      nr0=40, nt_interval=1, dx=10.0, dz=10.0,
                      dt=0.001, f0=15.0):
    values = [
        nx, nz, pml, 3, 0, ns, nt, ds, ns0, depths, depthr, nr, dr,
        nr0, nt_interval, dx, dz, dt, f0,
    ]
    with path.open("w", encoding="ascii") as stream:
        for index, value in enumerate(values):
            stream.write(f"-- parameter {index}\n{value}\n")


def test_uniform_model_produces_finite_pressure(tmp_path):
    aco2d = _load_extension()
    parameter_file = tmp_path / "parameters.txt"
    _write_parameters(parameter_file)
    velocity = np.full(61 * 61, 2000.0, dtype=np.float64)
    density = np.full(61 * 61, 1800.0, dtype=np.float64)

    record = aco2d.forward_variable_density(
        velocity, density, dim=240, paramfile=str(parameter_file)
    )

    assert record.shape == (240,)
    assert np.all(np.isfinite(record))
    assert np.max(np.abs(record)) > 0.0


def test_global_density_scaling_does_not_change_pressure(tmp_path):
    """A spatially constant density cancels from the pressure equation."""
    aco2d = _load_extension()
    parameter_file = tmp_path / "parameters.txt"
    _write_parameters(parameter_file)
    velocity = np.full(61 * 61, 2000.0, dtype=np.float64)
    density1 = np.full(61 * 61, 1000.0, dtype=np.float64)
    density2 = np.full(61 * 61, 2500.0, dtype=np.float64)

    record1 = aco2d.forward_variable_density(
        velocity, density1, dim=240, paramfile=str(parameter_file)
    )
    record2 = aco2d.forward_variable_density(
        velocity, density2, dim=240, paramfile=str(parameter_file)
    )

    np.testing.assert_allclose(record1, record2, rtol=1.0e-4, atol=1.0e-10)


def test_density_contrast_changes_pressure_data(tmp_path):
    aco2d = _load_extension()
    parameter_file = tmp_path / "parameters.txt"
    _write_parameters(parameter_file, depths=15, depthr=15, nr0=30, nt=300)
    velocity = np.full((61, 61), 2000.0, dtype=np.float64)
    uniform_density = np.full((61, 61), 1800.0, dtype=np.float64)
    layered_density = uniform_density.copy()
    layered_density[31:, :] = 2400.0

    uniform_record = aco2d.forward_variable_density(
        velocity.ravel(), uniform_density.ravel(), dim=300,
        paramfile=str(parameter_file)
    )
    layered_record = aco2d.forward_variable_density(
        velocity.ravel(), layered_density.ravel(), dim=300,
        paramfile=str(parameter_file)
    )

    difference = layered_record - uniform_record
    assert np.linalg.norm(difference) > 1.0e-4 * np.linalg.norm(uniform_record)
