# Acoustic kernel demos

From the repository root, build the extension in its module directory:

```bash
cd geopvi/forward/fwi2d
python setup.py build_ext --inplace
```

Run either kernel demo:

```bash
OMP_NUM_THREADS=2 python demos/constant_density_demo.py
OMP_NUM_THREADS=2 python demos/variable_density_demo.py
```

Both use a `51 x 81` grid, one shot, 81 receivers, `nt=400`, `pml0=8`, and
`Lc=3`.  Their layered true velocities and uniform starting velocities are
identical; the variable-density demo additionally has layered true density
and uniform starting density.

By default each script writes an NPZ file under the ignored `demos/output/`
directory.  Use `--no-save` for a calculation-only smoke test.  The
variable-density script also accepts `--verbose` to print phase timings.
