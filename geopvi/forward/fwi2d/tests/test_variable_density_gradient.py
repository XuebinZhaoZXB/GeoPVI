import importlib.util
from pathlib import Path
import sysconfig

import numpy as np


HERE = Path(__file__).resolve().parents[1]


def _load_extension():
    extension = HERE / ("aco2d" + sysconfig.get_config_var("EXT_SUFFIX"))
    spec = importlib.util.spec_from_file_location("aco2d", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_parameters(path):
    values = [
        31, 31, 8, 3, 0, 1, 150, 1, 10, 10, 10, 3, 5, 8, 1,
        10.0, 10.0, 0.001, 18.0,
    ]
    with path.open("w", encoding="ascii") as stream:
        for index, value in enumerate(values):
            stream.write(f"-- parameter {index}\n{value}\n")


def _objective(aco2d, velocity, density, observed, parameter_file):
    synthetic = aco2d.forward_variable_density(
        velocity, density, dim=observed.size, paramfile=str(parameter_file)
    )
    residual = synthetic.astype(np.float64) - observed.astype(np.float64)
    return 0.5 * np.dot(residual, residual)


def test_velocity_and_density_gradient_directional_derivative(tmp_path):
    aco2d = _load_extension()
    parameter_file = tmp_path / "parameters.txt"
    _write_parameters(parameter_file)
    nz = nx = 31
    z, x = np.mgrid[:nz, :nx]
    velocity = (1950.0 + 120.0 * z / (nz - 1)).astype(np.float64)
    density = (1750.0 + 180.0 * z / (nz - 1)).astype(np.float64)
    true_velocity = velocity + 35.0 * np.exp(-((x - 18) ** 2 + (z - 18) ** 2) / 24.0)
    true_density = density + 45.0 * np.exp(-((x - 14) ** 2 + (z - 20) ** 2) / 20.0)
    observed = aco2d.forward_variable_density(
        true_velocity.ravel(), true_density.ravel(), dim=450,
        paramfile=str(parameter_file)
    )

    synthetic, grad_v, grad_rho = aco2d.fwi_variable_density(
        velocity.ravel(), density.ravel(), observed.astype(np.float32),
        paramfile=str(parameter_file)
    )
    rng = np.random.default_rng(7)
    direction_v = rng.normal(size=(nz, nx))
    direction_rho = rng.normal(size=(nz, nx))
    # Keep the perturbation away from the copied absorbing-boundary model edge.
    taper = np.zeros((nz, nx))
    taper[3:-3, 3:-3] = 1.0
    direction_v *= taper * 40.0 / np.sqrt(np.mean(direction_v[taper > 0] ** 2))
    direction_rho *= taper * 40.0 / np.sqrt(np.mean(direction_rho[taper > 0] ** 2))
    adjoint_velocity = np.dot(grad_v.astype(np.float64), direction_v.ravel())
    adjoint_density = np.dot(grad_rho.astype(np.float64), direction_rho.ravel())
    adjoint_derivative = adjoint_velocity + adjoint_density

    # float32 propagation needs a perturbation large enough to rise above roundoff.
    epsilon = 0.05
    plus = _objective(
        aco2d, (velocity + epsilon * direction_v).ravel(),
        (density + epsilon * direction_rho).ravel(), observed, parameter_file
    )
    minus = _objective(
        aco2d, (velocity - epsilon * direction_v).ravel(),
        (density - epsilon * direction_rho).ravel(), observed, parameter_file
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)

    velocity_plus = _objective(
        aco2d, (velocity + epsilon * direction_v).ravel(), density.ravel(),
        observed, parameter_file
    )
    velocity_minus = _objective(
        aco2d, (velocity - epsilon * direction_v).ravel(), density.ravel(),
        observed, parameter_file
    )
    density_plus = _objective(
        aco2d, velocity.ravel(), (density + epsilon * direction_rho).ravel(),
        observed, parameter_file
    )
    density_minus = _objective(
        aco2d, velocity.ravel(), (density - epsilon * direction_rho).ravel(),
        observed, parameter_file
    )
    finite_difference_velocity = (velocity_plus - velocity_minus) / (2.0 * epsilon)
    finite_difference_density = (density_plus - density_minus) / (2.0 * epsilon)
    np.testing.assert_allclose(
        adjoint_derivative, finite_difference, rtol=3.0e-2,
        atol=1.0e-12 * max(1.0, abs(finite_difference))
    )
    np.testing.assert_allclose(
        adjoint_velocity, finite_difference_velocity, rtol=2.0e-2,
        atol=1.0e-12 * max(1.0, abs(finite_difference_velocity))
    )
    np.testing.assert_allclose(
        adjoint_density, finite_difference_density, rtol=3.0e-2,
        atol=1.0e-12 * max(1.0, abs(finite_difference_density))
    )

    phi0 = _objective(
        aco2d, velocity.ravel(), density.ravel(), observed, parameter_file
    )
    remainders = []
    for step in (0.4, 0.2, 0.1):
        perturbed = _objective(
            aco2d, (velocity + step * direction_v).ravel(),
            (density + step * direction_rho).ravel(), observed, parameter_file
        )
        remainders.append(abs(perturbed - phi0 - step * adjoint_derivative))
    # Halving the perturbation should reduce the first-order-corrected error by ~4.
    assert remainders[1] < 0.35 * remainders[0]
    assert remainders[2] < 0.35 * remainders[1]


def test_zero_residual_has_zero_gradients(tmp_path):
    aco2d = _load_extension()
    parameter_file = tmp_path / "parameters.txt"
    _write_parameters(parameter_file)
    velocity = np.full(31 * 31, 2000.0, dtype=np.float64)
    density = np.full(31 * 31, 1800.0, dtype=np.float64)
    observed = aco2d.forward_variable_density(
        velocity, density, dim=450, paramfile=str(parameter_file)
    )

    synthetic, grad_v, grad_rho = aco2d.fwi_variable_density(
        velocity, density, observed.astype(np.float32),
        paramfile=str(parameter_file)
    )

    np.testing.assert_array_equal(synthetic, observed)
    np.testing.assert_array_equal(grad_v, 0.0)
    np.testing.assert_array_equal(grad_rho, 0.0)
