#!/usr/bin/env python3
"""Small reproducible variable-density forward/gradient demonstration."""

import argparse
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARAMETERS = HERE / "input" / "input_params.txt"
NZ, NX, NT, NS, NR = 51, 81, 400, 1, 81


def load_extension():
    """Import the in-place extension and give a useful build error."""
    sys.path.insert(0, str(ROOT))
    try:
        from geopvi.forward.fwi2d import aco2d
    except ImportError as error:
        build_dir = ROOT / "geopvi" / "forward" / "fwi2d"
        raise RuntimeError(
            f"Build aco2d first: cd {build_dir} && "
            "python setup.py build_ext --inplace"
        ) from error
    return aco2d


def make_models():
    """Return layered true models and uniform starting models."""
    velocity_true = np.full((NZ, NX), 1800.0, dtype=np.float64)
    density_true = np.full((NZ, NX), 1800.0, dtype=np.float64)
    velocity_true[18:, :] = 2200.0
    velocity_true[34:, :] = 2600.0
    density_true[18:, :] = 2100.0
    density_true[34:, :] = 2400.0

    velocity_initial = np.full((NZ, NX), 2050.0, dtype=np.float64)
    density_initial = np.full((NZ, NX), 1950.0, dtype=np.float64)
    return velocity_true, density_true, velocity_initial, density_initial


def main():
    parser = argparse.ArgumentParser(
        description="Run a small variable-density pressure/gradient example."
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="print the summary without writing output/variable_density_demo.npz",
    )
    parser.add_argument("--verbose", action="store_true", help="print C timing details")
    args = parser.parse_args()

    aco2d = load_extension()
    v_true, rho_true, v_initial, rho_initial = make_models()
    data_size = NS * NT * NR

    observed = aco2d.forward_variable_density(
        v_true.ravel(), rho_true.ravel(), dim=data_size,
        paramfile=str(PARAMETERS),
    )
    synthetic, gradient_velocity, gradient_density = \
        aco2d.fwi_variable_density(
            v_initial.ravel(), rho_initial.ravel(),
            np.ascontiguousarray(observed, dtype=np.float32),
            paramfile=str(PARAMETERS), verbose=int(args.verbose),
        )

    observed = observed.reshape(NS, NT, NR)
    synthetic = synthetic.reshape(NS, NT, NR)
    gradient_velocity = gradient_velocity.reshape(NZ, NX)
    gradient_density = gradient_density.reshape(NZ, NX)
    residual = synthetic.astype(np.float64) - observed.astype(np.float64)

    print(f"data shape: {synthetic.shape}")
    print(f"gradient shapes: {gradient_velocity.shape}, {gradient_density.shape}")
    print(f"objective: {0.5 * np.vdot(residual, residual):.6e}")
    print(f"||dJ/dv||_2: {np.linalg.norm(gradient_velocity):.6e}")
    print(f"||dJ/drho||_2: {np.linalg.norm(gradient_density):.6e}")
    print(f"all finite: {bool(np.isfinite(synthetic).all() and np.isfinite(gradient_velocity).all() and np.isfinite(gradient_density).all())}")

    if not args.no_save:
        output = HERE / "output"
        output.mkdir(exist_ok=True)
        filename = output / "variable_density_demo.npz"
        np.savez_compressed(
            filename,
            velocity_true=v_true,
            density_true=rho_true,
            velocity_initial=v_initial,
            density_initial=rho_initial,
            observed=observed,
            synthetic=synthetic,
            gradient_velocity=gradient_velocity,
            gradient_density=gradient_density,
        )
        print(f"wrote {filename}")


if __name__ == "__main__":
    main()
