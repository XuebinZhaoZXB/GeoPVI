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


def _objective(aco2d, velocity, observed, parameter_file):
    synthetic = aco2d.forward(
        velocity, dim=observed.size, paramfile=str(parameter_file)
    )
    residual = synthetic.astype(np.float64) - observed.astype(np.float64)
    return 0.5 * np.dot(residual, residual)


def test_constant_density_velocity_gradient_directional_derivative(tmp_path):
    """The direct C gradient must differentiate the Python sample-sum loss."""
    aco2d = _load_extension()
    parameter_file = tmp_path / "parameters.txt"
    _write_parameters(parameter_file)

    nz = nx = 31
    z, x = np.mgrid[:nz, :nx]
    velocity = (1950.0 + 120.0 * z / (nz - 1)).astype(np.float64)
    true_velocity = velocity + 35.0 * np.exp(
        -((x - 18) ** 2 + (z - 18) ** 2) / 24.0
    )
    observed = aco2d.forward(
        true_velocity.ravel(), dim=450, paramfile=str(parameter_file)
    ).astype(np.float32)
    _, gradient = aco2d.fwi(
        velocity.ravel(), observed, paramfile=str(parameter_file)
    )

    rng = np.random.default_rng(7)
    direction = rng.normal(size=(nz, nx))
    taper = np.zeros((nz, nx))
    taper[7:-7, 7:-7] = 1.0
    direction *= taper * 40.0 / np.sqrt(
        np.mean(direction[taper > 0] ** 2)
    )
    adjoint_derivative = np.dot(
        gradient.astype(np.float64), direction.ravel()
    )

    epsilon = 0.05
    plus = _objective(
        aco2d, (velocity + epsilon * direction).ravel(), observed,
        parameter_file,
    )
    minus = _objective(
        aco2d, (velocity - epsilon * direction).ravel(), observed,
        parameter_file,
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(
        adjoint_derivative, finite_difference, rtol=2.0e-2,
        atol=1.0e-12 * max(1.0, abs(finite_difference)),
    )
