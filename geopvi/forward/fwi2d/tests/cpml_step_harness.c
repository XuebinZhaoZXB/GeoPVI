/* Test-only access to internal forward/transpose states. Never built into aco2d. */
#include "cpml2D.c"
#include <omp.h>

/* Set the runtime used by the compiled kernel (not an unrelated OpenMP
 * runtime that NumPy/PyTorch may also have loaded into the Python process). */
int cpml_test_threads(int count)
{
    int actual = 0;
    omp_set_dynamic(0);
    omp_set_num_threads(count);
    #pragma omp parallel
    {
        #pragma omp single
        actual = omp_get_num_threads();
    }
    return actual;
}

int cpml_test_step(int nx, int nz, int width, int radius, float *velocity,
                   float *input, float *output, int transpose)
{
    Cpml2D c;
    int d, n;
    float *p0, *p1, *p2;
    if (cpml_init(&c, nx, nz, radius, width, 10.0, 13.0, 0.0008,
                  20.0, 3000.0, 0.0001, velocity)) { cpml_free(&c); return 1; }
    n = c.count;
    p0 = array(n); p1 = array(n); p2 = array(n);
    if (!p0 || !p1 || !p2) { free(p0); free(p1); free(p2); cpml_free(&c); return 1; }
    for (d = 0; d < 2; ++d) {
        memcpy(c.psi[d], input+(2+d)*n, sizeof(float)*n);
        memcpy(c.eta[d], input+(4+d)*n, sizeof(float)*n);
    }
    if (!transpose) {
        /* State (current pressure, previous pressure, psi_x,z, eta_x,z). */
        memcpy(p1, input, sizeof(float)*n);
        memcpy(p0, input+n, sizeof(float)*n);
        cpml_pressure_step(&c, p0, p1, p2);
        memcpy(output, p2, sizeof(float)*n);
        memcpy(output+n, p1, sizeof(float)*n);
    } else {
        if (cpml_prepare_reverse(&c)) {
            free(p0); free(p1); free(p2); cpml_free(&c); return 1;
        }
        memcpy(p2, input, sizeof(float)*n);
        memcpy(p1, input+n, sizeof(float)*n);
        #pragma omp parallel
        cpml_reverse_workshare(&c, p2, p1, p0, NULL, NULL);
        memcpy(output, p1, sizeof(float)*n);
        memcpy(output+n, p0, sizeof(float)*n);
    }
    for (d = 0; d < 2; ++d) {
        memcpy(output+(2+d)*n, c.psi[d], sizeof(float)*n);
        memcpy(output+(4+d)*n, c.eta[d], sizeof(float)*n);
    }
    free(p0); free(p1); free(p2); cpml_free(&c);
    return 0;
}
