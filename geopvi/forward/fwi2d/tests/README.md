# Regression tests

From `geopvi/forward/fwi2d`, after building the extension:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests
```

Requires pytest, NumPy, package dependencies and GCC/OpenMP for the test
harnesses. On HPC systems, run on a compute node.

| Tests | Coverage |
|---|---|
| `test_constant_density_gradient.py` | Liao interior velocity gradient |
| `test_cpml.py`, `test_vdcpml.py` | CPML state transposes, finite differences, corners/interfaces, absorption and state handling |
| `test_cpml_api.py`, `test_parameter_writer.py` | Python validation and boundary selection |
| `test_variable_density_forward.py`, `test_variable_density_gradient.py` | Liao physics, density reduction, masks and gradients |
| `test_variable_density_manufactured.py` | Spatial-operator convergence |

`cpml_step_harness.c` and `vdcpml_step_harness.c` expose private states for
tests; they are not production extension sources. Keep the test sources,
not generated libraries or caches.

CPML gradient tests hold damping fixed; see [gradient limits](../docs/CPML.md).
Constant-density reduction means convergence, not bitwise equality between
the two spatial discretizations.
