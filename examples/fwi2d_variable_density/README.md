# Variable-density FWI2D example

Build the extension first:

```bash
cd geopvi/forward/fwi2d
python setup.py build_ext --inplace
```

Then run from the repository root:

```bash
OMP_NUM_THREADS=2 python examples/fwi2d_variable_density/variable_density_demo.py
```

The script constructs a small three-layer true velocity/density model and a
uniform starting model.  It generates observed pressure, evaluates the
starting model, computes `dJ/dv` and `dJ/drho`, and saves arrays under
`output/`.  Use `--no-save` for a calculation-only smoke test and `--verbose`
for C-side phase timings.
