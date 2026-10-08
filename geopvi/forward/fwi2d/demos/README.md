# Examples

From `geopvi/forward/fwi2d`, after building the extension:

```bash
OMP_NUM_THREADS=2 python demos/constant_density_demo.py --no-save
OMP_NUM_THREADS=2 python demos/variable_density_demo.py --no-save
```

Each script creates synthetic data and computes model gradients. Both use
a 51×81 grid, one shot, 81 receivers, 400 steps, eight absorbing cells,
radius 3, dx=dz=10 m, dt=1 ms and f0=18 Hz.

`input/constant_density_params.txt` and `input/variable_density_params.txt`
contain the corresponding 19-pair CPML configuration. See
[boundary selection](../docs/CPML.md) to use Liao.

Omit `--no-save` to write results to `demos/output/`; `--verbose` prints
native messages. The `run_constant_density.sh` and `run_variable_density.sh`
launchers use python3 and default to two OpenMP threads unless already set.
These are usage examples, not full accuracy tests.
