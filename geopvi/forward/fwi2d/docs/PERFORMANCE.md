# Threading and memory

The solvers run on CPUs using OpenMP and SIMD. Set thread counts before
starting Python, for example:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python your_script.py
```

When using posterior multiprocessing, worker count × OpenMP threads should
fit the allocated CPUs. Each worker needs its own solver memory. More threads
are not necessarily faster for small grids; benchmark on the target machine
with consistent CPU affinity. See the [timing script](../benchmarks/README.md).

## CPML history memory

Let `Ax=nx+2*w`, `Az=nz+2*w`, `w=pml0`, `r=Lc`. Approximate saved-history
sizes in bytes are:

| Solver | History bytes |
|---|---|
| Constant density | `4*nt*Ax*Az` |
| Variable density | `4*nt*(Ax*Az+(4*w+2*r)*(Ax+Az))` |

Work arrays add grid-sized storage. History is reused across shots;
forward-only calls allocate no history or reverse workspace.
Wavefields/history are float32; CPML gradient accumulation is double precision.
