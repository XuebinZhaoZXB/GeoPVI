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
                      lc=3, ds=1, ns0=30, depths=30, depthr=30, nr=1, dr=1,
                      nr0=40, nt_interval=1, dx=10.0, dz=10.0,
                      dt=0.001, f0=15.0):
    values = [
        nx, nz, pml, lc, 0, ns, nt, ds, ns0, depths, depthr, nr, dr,
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


def test_variable_density_rejects_pseudospectral_setting(tmp_path):
    aco2d = _load_extension()
    parameter_file = tmp_path / "parameters.txt"
    _write_parameters(parameter_file)
    lines = parameter_file.read_text(encoding="ascii").splitlines()
    # The fifth value is the legacy laplace_solver control.
    lines[9] = "1"
    parameter_file.write_text("\n".join(lines) + "\n", encoding="ascii")
    velocity = np.full(61 * 61, 2000.0, dtype=np.float64)
    density = np.full(61 * 61, 1800.0, dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "laplace_solver=0"):
        aco2d.forward_variable_density(
            velocity, density, dim=240, paramfile=str(parameter_file)
        )


def test_lc_selects_a_different_finite_difference_stencil(tmp_path):
    aco2d = _load_extension()
    parameter_lc1 = tmp_path / "parameters_lc1.txt"
    parameter_lc3 = tmp_path / "parameters_lc3.txt"
    _write_parameters(parameter_lc1, lc=1)
    _write_parameters(parameter_lc3, lc=3)
    velocity = np.full(61 * 61, 2000.0, dtype=np.float64)
    density = np.full(61 * 61, 1800.0, dtype=np.float64)

    record_lc1 = aco2d.forward_variable_density(
        velocity, density, dim=240, paramfile=str(parameter_lc1)
    )
    record_lc3 = aco2d.forward_variable_density(
        velocity, density, dim=240, paramfile=str(parameter_lc3)
    )

    assert np.all(np.isfinite(record_lc1))
    assert np.all(np.isfinite(record_lc3))
    assert np.linalg.norm(record_lc1 - record_lc3) > 1.0e-4 * np.linalg.norm(record_lc3)


def test_variable_density_rejects_unsupported_lc(tmp_path):
    aco2d = _load_extension()
    parameter_file = tmp_path / "parameters.txt"
    _write_parameters(parameter_file, lc=7)
    velocity = np.full(61 * 61, 2000.0, dtype=np.float64)
    density = np.full(61 * 61, 1800.0, dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "1 <= Lc <= 6"):
        aco2d.forward_variable_density(
            velocity, density, dim=240, paramfile=str(parameter_file)
        )


def test_constant_density_matches_legacy_fdm_before_boundary_arrivals(tmp_path):
    """Both formulations solve v**2 Laplace(p) when density is constant."""
    aco2d = _load_extension()
    velocity = np.full(81 * 81, 2000.0, dtype=np.float64)
    density = np.full(81 * 81, 1800.0, dtype=np.float64)

    # The legacy code provides coefficients only for Lc=3,...,6.
    for lc in range(3, 7):
        parameter_file = tmp_path / f"parameters_lc_{lc}.txt"
        _write_parameters(
            parameter_file, nx=81, nz=81, pml=20, lc=lc, nt=180,
            ns0=40, depths=40, depthr=40, nr0=52, f0=15.0,
        )
        legacy_record = aco2d.forward(
            velocity, dim=180, paramfile=str(parameter_file)
        )
        variable_density_record = aco2d.forward_variable_density(
            velocity, density, dim=180, paramfile=str(parameter_file)
        )

        # The two high-order stencils differ, so they are not bitwise identical.
        # Compare only before energy can return from the closest physical boundary.
        comparison_samples = 100
        relative_error = np.linalg.norm(
            variable_density_record[:comparison_samples] - legacy_record[:comparison_samples]
        ) / np.linalg.norm(legacy_record[:comparison_samples])
        assert relative_error < 2.0e-3


def test_constant_density_legacy_difference_decreases_with_grid_refinement(tmp_path):
    """The two consistent high-order discretisations approach one another."""
    aco2d = _load_extension()
    # The physical square is 400 m by 400 m in every case.  dt scales with
    # dx, and the 70 ms record ends before an outer-boundary return arrives.
    for lc in range(3, 7):
        errors = []
        for dx, nx, pml0, nt in ((10.0, 41, 12, 70), (5.0, 81, 24, 140),
                                 (2.5, 161, 48, 280)):
            scale = (nx - 1) // 40
            parameter_file = tmp_path / f"parameters_lc_{lc}_dx_{dx}.txt"
            _write_parameters(
                parameter_file, nx=nx, nz=nx, pml=pml0, lc=lc, nt=nt,
                ns0=20 * scale, depths=20 * scale, depthr=20 * scale,
                nr0=26 * scale, dx=dx, dz=dx, dt=0.001 / scale, f0=50.0,
            )
            velocity = np.full(nx * nx, 2000.0, dtype=np.float64)
            density = np.full(nx * nx, 1800.0, dtype=np.float64)
            legacy_record = aco2d.forward(
                velocity, dim=nt, paramfile=str(parameter_file)
            )
            variable_density_record = aco2d.forward_variable_density(
                velocity, density, dim=nt, paramfile=str(parameter_file)
            )
            errors.append(np.linalg.norm(variable_density_record - legacy_record) /
                          np.linalg.norm(legacy_record))

        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 2.0e-4


def test_density_interface_has_expected_reflection_polarity_and_amplitude_trend(tmp_path):
    """At near-normal incidence, pressure reflection follows impedance contrast."""
    aco2d = _load_extension()
    parameter_file = tmp_path / "parameters.txt"
    _write_parameters(
        parameter_file, nx=101, nz=101, pml=24, lc=3, nt=450,
        ns0=50, depths=20, depthr=20, nr0=50, f0=15.0,
    )
    velocity = np.full((101, 101), 2000.0, dtype=np.float64)
    reference_density = np.full((101, 101), 1800.0, dtype=np.float64)
    high_density = reference_density.copy()
    low_density = reference_density.copy()
    high_density[45:, :] = 2400.0
    low_density[45:, :] = 1400.0

    reference = aco2d.forward_variable_density(
        velocity.ravel(), reference_density.ravel(), dim=450,
        paramfile=str(parameter_file)
    )
    high_contrast = aco2d.forward_variable_density(
        velocity.ravel(), high_density.ravel(), dim=450,
        paramfile=str(parameter_file)
    ) - reference
    low_contrast = aco2d.forward_variable_density(
        velocity.ravel(), low_density.ravel(), dim=450,
        paramfile=str(parameter_file)
    ) - reference

    # The source/receiver are above the horizontal interface.  The expected
    # reflected arrival occupies this window; differencing removes the direct
    # wave present in all three models.
    reflection_window = slice(180, 300)
    high_reflection = high_contrast[reflection_window]
    low_reflection = low_contrast[reflection_window]
    correlation = np.dot(high_reflection, low_reflection) / (
        np.linalg.norm(high_reflection) * np.linalg.norm(low_reflection)
    )
    amplitude_ratio = (np.max(np.abs(high_reflection)) /
                       np.max(np.abs(low_reflection)))
    theoretical_ratio = abs((2400.0 - 1800.0) / (2400.0 + 1800.0)) / abs(
        (1400.0 - 1800.0) / (1400.0 + 1800.0)
    )

    assert correlation < -0.99
    np.testing.assert_allclose(amplitude_ratio, theoretical_ratio, rtol=0.10)
