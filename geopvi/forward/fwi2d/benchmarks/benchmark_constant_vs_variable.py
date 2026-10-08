#!/usr/bin/env python3
"""Compare constant/variable-density runtimes with CPML (default) or Liao."""

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sysconfig
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FWI2D = HERE.parent
NZ, NX, NT, NS, NR = 110, 250, 2000, 5, 250


def load_extension():
    extension = FWI2D / ("aco2d" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not extension.exists():
        raise RuntimeError(f"Build the extension first in {FWI2D}")
    spec = importlib.util.spec_from_file_location("aco2d", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_parameters(path, boundary):
    values = [
        NX, NZ, 10, 3, 0, NS, NT, max(1, (NX-50)//max(1, NS-1)), 25, 9, 9, NR, 1, 0, 1,
        20.0, 20.0, 0.002, 10.0,
    ]
    if boundary == 'liao':
        values.append(0)  # CPML deliberately exercises the 19-pair default.
    with path.open("w", encoding="ascii") as stream:
        for index, value in enumerate(values):
            stream.write(f"-- parameter {index}\n{value}\n")


def timed(callable_object, repeats, label):
    samples = []
    for repeat in range(repeats):
        start = time.perf_counter()
        result = callable_object()
        elapsed = time.perf_counter() - start
        samples.append(elapsed)
        arrays = result if isinstance(result, tuple) else (result,)
        assert all(np.isfinite(array).all() for array in arrays)
        print(f"{label} {repeat + 1}/{repeats}: {elapsed:.3f} s", flush=True)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


def comparison(variable, constant):
    ratio = variable["median_seconds"] / constant["median_seconds"]
    return {
        "variable_to_constant_time_ratio": ratio,
        "variable_slowdown_percent": (ratio - 1.0) * 100.0,
        "variable_relative_throughput": 1.0 / ratio,
    }


def main():
    global NZ, NX, NT, NS, NR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--boundary", choices=['cpml', 'liao'], default='cpml')
    parser.add_argument("--quick", action='store_true', help='small smoke workload, not a timing reference')
    parser.add_argument("--output-dir", type=Path, default=HERE/'output')
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.quick:
        NZ, NX, NT, NS, NR = 51, 81, 120, 1, 81

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter = output_dir / ("parameters_%s.txt" % args.boundary)
    write_parameters(parameter, args.boundary)
    aco2d = load_extension()

    velocity = np.full((NZ, NX), 2000.0, dtype=np.float64)
    density = np.full((NZ, NX), 1800.0, dtype=np.float64)
    z, x = np.mgrid[:NZ, :NX]
    velocity_true = velocity + 100.0 * np.exp(
        -((x - .58*NX) ** 2 / (.0112*NX**2) + (z - .59*NZ) ** 2 / (.0215*NZ**2))
    )
    data_size = NS * NT * NR
    observed_constant = aco2d.forward(
        velocity_true.ravel(), dim=data_size, paramfile=str(parameter)
    ).astype(np.float32)
    observed_variable = aco2d.forward_variable_density(
        velocity_true.ravel(), density.ravel(), dim=data_size,
        paramfile=str(parameter),
    ).astype(np.float32)

    forward_constant_call = lambda: aco2d.forward(
        velocity.ravel(), dim=data_size, paramfile=str(parameter)
    )
    forward_variable_call = lambda: aco2d.forward_variable_density(
        velocity.ravel(), density.ravel(), dim=data_size,
        paramfile=str(parameter),
    )
    gradient_constant_call = lambda: aco2d.fwi(
        velocity.ravel(), observed_constant, paramfile=str(parameter)
    )
    gradient_variable_call = lambda: aco2d.fwi_variable_density(
        velocity.ravel(), density.ravel(), observed_variable,
        paramfile=str(parameter),
    )

    forward_constant_call()
    forward_variable_call()
    gradient_constant_call()
    gradient_variable_call()
    print("completed untimed forward and gradient warm-up", flush=True)
    forward_constant = timed(forward_constant_call, args.repeats, "constant forward")
    forward_variable = timed(forward_variable_call, args.repeats, "variable forward")
    gradient_constant = timed(
        gradient_constant_call, args.repeats, "constant forward+gradient"
    )
    gradient_variable = timed(
        gradient_variable_call, args.repeats, "variable forward+gradients"
    )

    result = {
        "configuration": {
            "model_shape_nz_nx": [NZ, NX],
            "nt": NT,
            "shots": NS,
            "receivers_per_shot": NR,
            "pml0": 10,
            "Lc": 3,
            "boundary": args.boundary,
            "quick": args.quick,
            "requested_openmp_threads": os.environ.get("OMP_NUM_THREADS", "not explicitly set"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "repeats": args.repeats,
        },
        "forward": {
            "constant_density": forward_constant,
            "variable_density": forward_variable,
            "comparison": comparison(forward_variable, forward_constant),
        },
        "forward_adjoint_gradient": {
            "constant_density": gradient_constant,
            "variable_density": gradient_variable,
            "comparison": comparison(gradient_variable, gradient_constant),
        },
    }
    output = output_dir / ("benchmark_results_%s.json" % args.boundary)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
