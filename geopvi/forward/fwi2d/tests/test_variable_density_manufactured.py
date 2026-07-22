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


def _manufactured_fields(n):
    """Return p, rho, and exact div((1/rho) grad(p)) on [0, 1]^2."""
    coordinate = np.linspace(0.0, 1.0, n)
    z, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    wave_number = 8.0 * np.pi
    pressure = np.sin(wave_number * x) * np.sin(wave_number * z)
    density = 2.0 + 0.2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * z)

    pressure_x = wave_number * np.cos(wave_number * x) * np.sin(wave_number * z)
    pressure_z = wave_number * np.sin(wave_number * x) * np.cos(wave_number * z)
    pressure_laplacian = -2.0 * wave_number ** 2 * pressure
    density_x = 0.4 * np.pi * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * z)
    density_z = -0.4 * np.pi * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * z)
    exact = (pressure_laplacian / density -
             density_x * pressure_x / density ** 2 -
             density_z * pressure_z / density ** 2)
    return pressure, density, exact


def _relative_operator_error(aco2d, n, lc):
    pressure, density, exact = _manufactured_fields(n)
    numerical = aco2d.variable_density_spatial_operator(
        density.ravel(), pressure.ravel(), n, n, lc,
        1.0 / (n - 1), 1.0 / (n - 1)
    ).reshape(n, n)
    # Exclude the copied outer extension: this test measures the interior
    # high-order operator, not its intentionally simple diagnostic extension.
    margin = 2 * lc
    interior = np.s_[margin:-margin, margin:-margin]
    return (np.linalg.norm(numerical[interior] - exact[interior]) /
            np.linalg.norm(exact[interior]))


def test_smooth_variable_density_manufactured_solution_converges():
    aco2d = _load_extension()
    second_order_errors = [_relative_operator_error(aco2d, n, lc=1) for n in (33, 65)]
    sixth_order_errors = [_relative_operator_error(aco2d, n, lc=3) for n in (33, 65)]

    # Halving h reduces the Lc=1 error by about four.  The Lc=3 operator is
    # substantially more accurate and drops much faster before float32
    # roundoff becomes dominant.
    assert second_order_errors[1] < 0.30 * second_order_errors[0]
    assert sixth_order_errors[1] < 0.05 * sixth_order_errors[0]
    assert sixth_order_errors[0] < 0.01 * second_order_errors[0]
