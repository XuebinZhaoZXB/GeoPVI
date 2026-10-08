#ifndef GEOPVI_CPML2D_H
#define GEOPVI_CPML2D_H

/* Per-call ownership; never shared between shots, forward/adjoint, or callers. */
typedef struct Cpml2D Cpml2D;

int read_boundary_options(const char *path, int *boundary);
int cpml_solve(char *path, float *velocity, float *synthetic,
               const float *observed, float *gradient, const float *mask,
               int run_fwi, int verbose);
void cpml_pressure_step(Cpml2D *state, const float *previous,
                        const float *pressure, float *next);

#endif
