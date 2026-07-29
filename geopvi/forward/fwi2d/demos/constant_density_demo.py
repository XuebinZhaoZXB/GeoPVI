#!/usr/bin/env python3
"""Small reproducible constant-density forward/gradient demonstration."""

import argparse
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "input" / "constant_density_params.txt"
NZ, NX, NT, NS, NR = 51, 81, 400, 1, 81


def load_extension():
    """Import the in-place extension and give a useful build error."""
    module_dir = HERE.parent
    sys.path.insert(0, str(module_dir))
    try:
        import aco2d
    except ImportError as error:
        raise RuntimeError(
            f"Build aco2d first: cd {module_dir} && "
            "python setup.py build_ext --inplace"
        ) from error
    return aco2d


def make_models():
    """Return a layered true velocity and a uniform starting velocity."""
    velocity_true = np.full((NZ, NX), 1800.0, dtype=np.float64)
    velocity_true[18:, :] = 2200.0
    velocity_true[34:, :] = 2600.0
    velocity_initial = np.full((NZ, NX), 2050.0, dtype=np.float64)
    return velocity_true, velocity_initial


def main():
    parser = argparse.ArgumentParser(
        description="Run a small constant-density pressure/gradient example."
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="print the summary without writing output/constant_density_demo.npz",
    )
    parser.add_argument("--verbose", action="store_true", help="print C messages")
    args = parser.parse_args()

    aco2d = load_extension()
    velocity_true, velocity_initial = make_models()
    data_size = NS * NT * NR
    observed = aco2d.forward(
        velocity_true.ravel(), dim=data_size, paramfile=str(PARAMETERS),
    ).astype(np.float32)
    synthetic, gradient_velocity = aco2d.fwi(
        velocity_initial.ravel(), np.ascontiguousarray(observed),
        paramfile=str(PARAMETERS), verbose=int(args.verbose),
    )

    observed = observed.reshape(NS, NT, NR)
    synthetic = synthetic.reshape(NS, NT, NR)
    gradient_velocity = gradient_velocity.reshape(NZ, NX)
    residual = synthetic.astype(np.float64) - observed.astype(np.float64)

    print(f"data shape: {synthetic.shape}")
    print(f"gradient shape: {gradient_velocity.shape}")
    print(f"objective: {0.5 * np.vdot(residual, residual):.6e}")
    print(f"||dJ/dv||_2: {np.linalg.norm(gradient_velocity):.6e}")
    print(
        "all finite: "
        f"{bool(np.isfinite(synthetic).all() and np.isfinite(gradient_velocity).all())}"
    )

    if not args.no_save:
        output = HERE / "output"
        output.mkdir(exist_ok=True)
        filename = output / "constant_density_demo.npz"
        np.savez_compressed(
            filename,
            velocity_true=velocity_true,
            velocity_initial=velocity_initial,
            observed=observed,
            synthetic=synthetic,
            gradient_velocity=gradient_velocity,
        )
        print(f"wrote {filename}")


if __name__ == "__main__":
    main()
