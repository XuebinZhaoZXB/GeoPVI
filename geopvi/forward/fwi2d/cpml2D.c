/* Second-order pressure CPML, kappa=1 (complex-frequency-shifted PML).
 *
 * For each direction d (a=b=0 outside the absorbing strip):
 *   psi' = b psi + a D_d p
 *   H    = L_d p + D_d psi'
 *   eta' = b eta + a H
 *   S_d  = H + eta'
 * p' = 2p - p_old + dt^2 v^2 (S_x + S_z + source).
 * L_d is the ORIGINAL direct high-order second derivative, not D_d D_d.
 * Memory states live only in the exterior strip, but D_d psi' reaches into
 * physical edge cells through its stencil. Do not mask this interface
 * coupling. Deep interior cells still use the original Laplacian.
 * The reverse below transposes this full discrete recurrence.
 * See docs/CPML.md for the continuum construction and input conventions.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fdtd2D_modelling.h"
#include "cpml2D.h"

/* Internal coefficient-design target, not a measured reflection guarantee. */
#define CPML_TARGET_REFLECTION 1.0e-4

struct Cpml2D {
    int nx, nz, halo, offset, count, radius;
    float dx, dz, dt;
    float fd[7], first[7];
    float *velocity, *pressure_scale, *a[2], *b[2], *psi[2], *eta[2];
    float *rhs, *hbar[2], *temporary[2];
    double *gradient_scale;
    /* Axis runs of identical correction/damping flags, computed once. */
    int spans, *xcut, *xflag, *zflag;
};

/* Optional boundary selector pair, after the unchanged 19 old pairs.
 * Missing options select CPML for both constant and variable density.
 * Explicit values retain their meaning: 0=Liao, 1=CPML. */
int read_boundary_options(const char *path, int *boundary)
{
    FILE *stream = fopen(path, "r");
    char line[512], extra;
    double value;
    int index = 0;
    *boundary = 1;
    if (!stream) return 1;
    /* The original format consists of alternating nonempty comment/value lines. */
    while (fgets(line, sizeof(line), stream)) {
        if (sscanf(line, " %c", &extra) != 1) continue;
        if (index >= 38 && index % 2 == 1) {
            if (sscanf(line, " %lf %c", &value, &extra) != 1 || !isfinite(value)) {
                fclose(stream); return 1;
            }
            if (index == 39) {
                if (value != 0.0 && value != 1.0) { fclose(stream); return 1; }
                *boundary = (int)value;
            } else { fclose(stream); return 1; }
        }
        ++index;
    }
    fclose(stream);
    return index != 38 && index != 40;
}

static float *array(size_t count)
{
    if (count > SIZE_MAX / sizeof(float)) return NULL;
    return (float *)calloc(count, sizeof(float));
}

static void cpml_free(Cpml2D *c)
{
    int d;
    for (d = 0; d < 2; ++d) {
        free(c->a[d]); free(c->b[d]); free(c->psi[d]); free(c->eta[d]);
        free(c->hbar[d]); free(c->temporary[d]);
    }
    free(c->velocity); free(c->pressure_scale); free(c->rhs);
    free(c->gradient_scale);
    free(c->xcut); free(c->xflag); free(c->zflag);
}

static void reset_memory(Cpml2D *c)
{
    int d;
    for (d = 0; d < 2; ++d) {
        memset(c->psi[d], 0, sizeof(float) * c->count);
        memset(c->eta[d], 0, sizeof(float) * c->count);
    }
}

/* Reverse-only scratch: a data-only forward call does not allocate it. */
static int cpml_prepare_reverse(Cpml2D *c)
{
    int d;
    for (d = 0; d < 2; ++d) {
        c->hbar[d] = array(c->count);
        c->temporary[d] = array(c->count);
        if (!c->hbar[d] || !c->temporary[d]) return 1;
    }
    return 0;
}

static int cpml_init(Cpml2D *c, int nx, int nz, int radius, int width,
                     float dx, float dz, float dt, float f0,
                     float reference_velocity, float reflection, float *velocity)
{
    int d, x, z, k, j;
    memset(c, 0, sizeof(*c));
    c->halo = radius + 1;
    c->offset = width + c->halo;
    c->nx = nx + 2*c->offset; c->nz = nz + 2*c->offset;
    c->count = c->nx*c->nz; c->radius = radius;
    c->dx = dx; c->dz = dz; c->dt = dt;
    fd_coefficient(radius + 1, c->fd);
    /* Standard 2r-order centred first derivative; used only in corrections. */
    for (k = 1; k <= radius; ++k) {
        double factor = 1.0;
        for (j = 0; j < k; ++j) factor *= (double)(radius-j)/(radius+j+1);
        c->first[k] = (float)((k % 2 ? 1.0 : -1.0)*factor/k);
    }
    c->velocity = array(c->count); c->rhs = array(c->count);
    c->pressure_scale = array(c->count);
    for (d = 0; d < 2; ++d) {
        c->a[d] = array(c->count); c->b[d] = array(c->count);
        c->psi[d] = array(c->count); c->eta[d] = array(c->count);
        if (!c->a[d] || !c->b[d] || !c->psi[d] || !c->eta[d]) return 1;
    }
    if (!c->velocity || !c->pressure_scale || !c->rhs) return 1;
    get_velp(c->offset, c->nx, c->nz, velocity, c->velocity);
    for (j = 0; j < c->count; ++j)
        c->pressure_scale[j] = powf(c->velocity[j]*dt, 2);
    for (d = 0; d < 2; ++d) {
        double h = d == 0 ? dx : dz;
        double sigma_max = -3.0*reference_velocity*log(reflection)/(2.0*width*h);
        for (z = c->halo; z < c->nz-c->halo; ++z) {
            for (x = c->halo; x < c->nx-c->halo; ++x) {
                int position = d == 0 ? x : z;
                int length = d == 0 ? c->nx : c->nz;
                int distance = 0, i = z*c->nx+x;
                if (position < c->offset) distance = c->offset-position;
                if (position >= length-c->offset)
                    distance = position-(length-c->offset-1);
                if (distance > 0) {
                    double fraction = (double)distance/width;
                    double sigma = sigma_max*fraction*fraction;
                    double alpha = 3.14159265358979323846*f0*(1.0-fraction);
                    double exponent = -(sigma+alpha)*dt;
                    c->b[d][i] = (float)exp(exponent);
                    c->a[d][i] = (float)(sigma/(sigma+alpha)*expm1(exponent));
                }
                /* Else a=b=0: interior auxiliary states are identically zero. */
            }
        }
    }
    c->xcut = calloc((size_t)c->nx+1, sizeof(int));
    c->xflag = calloc((size_t)c->nx, sizeof(int));
    c->zflag = calloc((size_t)c->nz, sizeof(int));
    if (!c->xcut || !c->xflag || !c->zflag) return 1;
    for (x = c->halo; x < c->nx-c->halo; ++x) {
        int flag = (x < c->offset+radius || x >= c->nx-c->offset-radius) ? 1 : 0;
        if (c->a[0][c->halo*c->nx+x] != 0.0f) flag |= 2;
        if (!c->spans || flag != c->xflag[c->spans-1]) {
            c->xcut[c->spans] = x;
            c->xflag[c->spans++] = flag;
        }
    }
    c->xcut[c->spans] = c->nx-c->halo;
    for (z = c->halo; z < c->nz-c->halo; ++z) {
        c->zflag[z] = (z < c->offset+radius || z >= c->nz-c->offset-radius) ? 1 : 0;
        if (c->a[1][z*c->nx+c->halo] != 0.0f) c->zflag[z] |= 2;
    }
    return 0;
}

#define CDCP_RADIUS 3
#define CDCP_NAME(name) name##_r3
#include "cpml2D_kernels.h"
#undef CDCP_NAME
#undef CDCP_RADIUS

#define CDCP_RADIUS 4
#define CDCP_NAME(name) name##_r4
#include "cpml2D_kernels.h"
#undef CDCP_NAME
#undef CDCP_RADIUS

#define CDCP_RADIUS 5
#define CDCP_NAME(name) name##_r5
#include "cpml2D_kernels.h"
#undef CDCP_NAME
#undef CDCP_RADIUS

#define CDCP_RADIUS 6
#define CDCP_NAME(name) name##_r6
#include "cpml2D_kernels.h"
#undef CDCP_NAME
#undef CDCP_RADIUS

static void cpml_pressure_workshare(Cpml2D *c, const float *previous, const float *pressure,
                                     float *next, float *history)
{
    switch (c->radius) {
    case 3:
        if (history) cpml_pressure_workshare_r3(c, previous, pressure, next, history);
        else cpml_pressure_workshare_r3(c, previous, pressure, next, NULL);
        break;
    case 4:
        if (history) cpml_pressure_workshare_r4(c, previous, pressure, next, history);
        else cpml_pressure_workshare_r4(c, previous, pressure, next, NULL);
        break;
    case 5:
        if (history) cpml_pressure_workshare_r5(c, previous, pressure, next, history);
        else cpml_pressure_workshare_r5(c, previous, pressure, next, NULL);
        break;
    case 6:
        if (history) cpml_pressure_workshare_r6(c, previous, pressure, next, history);
        else cpml_pressure_workshare_r6(c, previous, pressure, next, NULL);
        break;
    }
}

static void cpml_reverse_workshare(Cpml2D *c, const float *next, float *now, float *old,
                                    const float *history, double *gradient)
{
    switch (c->radius) {
    case 3:
        if (gradient) cpml_reverse_workshare_r3(c, next, now, old, history, gradient);
        else cpml_reverse_workshare_r3(c, next, now, old, NULL, NULL);
        break;
    case 4:
        if (gradient) cpml_reverse_workshare_r4(c, next, now, old, history, gradient);
        else cpml_reverse_workshare_r4(c, next, now, old, NULL, NULL);
        break;
    case 5:
        if (gradient) cpml_reverse_workshare_r5(c, next, now, old, history, gradient);
        else cpml_reverse_workshare_r5(c, next, now, old, NULL, NULL);
        break;
    case 6:
        if (gradient) cpml_reverse_workshare_r6(c, next, now, old, history, gradient);
        else cpml_reverse_workshare_r6(c, next, now, old, NULL, NULL);
        break;
    }
}

void cpml_pressure_step(Cpml2D *c, const float *previous,
                        const float *pressure, float *next)
{
    #pragma omp parallel
    cpml_pressure_workshare(c, previous, pressure, next, NULL);
}

int cpml_solve(char *path, float *velocity, float *synthetic,
               const float *observed, float *gradient, const float *mask,
               int run_fwi, int verbose)
{
    int nx, nz, width, radius, solver, ns, nt, ds, sx, sz, rz, nr, dr, rx, interval;
    int shot, n, x, z, ir, i, status = 1, do_gradient = observed != NULL;
    float dx, dz, dt, f0, *p0 = NULL, *p1 = NULL, *p2 = NULL;
    float *history = NULL, *wavelet = NULL, *swap;
    double *full_gradient = NULL;
    float reference_velocity = 0.0f;
    size_t history_count = 0;
    int history_width = 0;
    const float reflection = CPML_TARGET_REFLECTION;
    Cpml2D c;
    memset(&c, 0, sizeof(c));
    read_parameters(path, &nx, &nz, &width, &radius, &solver, &ns, &nt,
                    &ds, &sx, &sz, &rz, &nr, &dr, &rx, &interval,
                    &dx, &dz, &dt, &f0);
    /* C-call contracts, also validated with descriptive errors in Cython. */
    if (nx <= 0 || nz <= 0 || width < 1 || radius < 3 || radius > 6 || solver != 0 ||
        ns <= 0 || nt <= 0 || nr <= 0 || ds < 0 || dr < 1 || interval != 1 ||
        !run_fwi || !isfinite(dx) || !isfinite(dz) || !isfinite(dt) || !isfinite(f0) ||
        dx <= 0 || dz <= 0 || dt <= 0 || f0 <= 0 ||
        sx < 0 || (long long)sx+(long long)(ns-1)*ds >= nx ||
        rx < 0 || (long long)rx+(long long)(nr-1)*dr >= nx ||
        sz < 0 || sz >= nz || rz < 0 || rz >= nz ||
        (long long)nx+2LL*(width+radius+1LL) > INT_MAX ||
        (long long)nz+2LL*(width+radius+1LL) > INT_MAX ||
        ((long long)nx+2LL*(width+radius+1LL))*
        ((long long)nz+2LL*(width+radius+1LL)) > INT_MAX) goto done;
    {
        float fd[7];
        double vmax = 0, symbol;
        fd_coefficient(radius+1, fd);
        symbol = -fd[0];
        for (i = 1; i <= radius; ++i) symbol -= 2*fd[i]*(i % 2 ? -1 : 1);
        for (i = 0; i < nx*nz; ++i) {
            if (!isfinite(velocity[i]) || velocity[i] <= 0) goto done;
            if (velocity[i] > vmax) vmax = velocity[i];
        }
        /* Recomputed per model/call, shared by all shots and the reverse pass.
         * Deliberately stop the derivative through max(v) and the resulting
         * damping coefficients: requested approximate model gradient. */
        reference_velocity = (float)vmax;
        if (dt*vmax*sqrt(symbol*(1.0/dx/dx+1.0/dz/dz)) >= 2.0) {
            fprintf(stderr, "CPML: interior finite-difference CFL limit exceeded.\n");
            goto done;
        }
    }
    if (cpml_init(&c, nx, nz, radius, width, dx, dz, dt, f0,
                  reference_velocity, reflection, velocity)) goto done;
    p0 = array(c.count); p1 = array(c.count); p2 = array(c.count);
    wavelet = array(nt);
    if (!p0 || !p1 || !p2 || !wavelet) goto done;
    if (do_gradient) {
        if (cpml_prepare_reverse(&c)) goto done;
        /* Do not save the never-updated zero ghost halo. All active CPML
         * cells ARE retained, so the extension gradient is unchanged. */
        history_width = c.nx-2*c.halo;
        history_count = (size_t)history_width*(c.nz-2*c.halo);
        if ((size_t)nt > SIZE_MAX/sizeof(float)/history_count) goto done;
        history = array((size_t)nt*history_count);
        full_gradient = (double *)calloc(c.count, sizeof(double));
        c.gradient_scale = (double *)calloc(c.count, sizeof(double));
        if (!history || !full_gradient || !c.gradient_scale) goto done;
        for (i = 0; i < c.count; ++i)
            c.gradient_scale[i] = 2.0*(double)dt*dt*c.velocity[i];
    }
    getrik(nt, dt, f0, wavelet);
    if (verbose) fprintf(stdout, "Constant-density CPML: width=%d, vmax=%g, R=%g; gradient holds damping coefficients fixed.\n",
                          width, reference_velocity, reflection);
    for (shot = 0; shot < ns; ++shot) {
        int source = (sz+c.offset)*c.nx+sx+shot*ds+c.offset;
        memset(p0, 0, sizeof(float)*c.count);
        memset(p1, 0, sizeof(float)*c.count);
        memset(p2, 0, sizeof(float)*c.count);
        reset_memory(&c);
        /* Keep the team alive across ALL time steps of this shot. Source,
         * receivers, history and pointer rotation are single-writer operations;
         * explicit barriers publish them before the next stencil. */
        #pragma omp parallel private(n)
        {
        for (n = 0; n < nt; ++n) {
            cpml_pressure_workshare(&c, p0, p1, p2,
                                    do_gradient ? history+(size_t)n*history_count : NULL);
            #pragma omp master
            {
            p2[source] += wavelet[n]*dt*dt*c.velocity[source]*c.velocity[source];
            c.rhs[source] += wavelet[n];
            for (ir = 0; ir < nr; ++ir)
                synthetic[((size_t)shot*nt+n)*nr+ir] = p2[(rz+c.offset)*c.nx+rx+ir*dr+c.offset];
            if (do_gradient)
                history[(size_t)n*history_count+
                        (size_t)(sz+width)*history_width+sx+shot*ds+width] = c.rhs[source];
            swap = p0; p0 = p1; p1 = p2; p2 = swap;
            }
            #pragma omp barrier
        }
        }
        if (!do_gradient) continue;
        memset(p0, 0, sizeof(float)*c.count);
        memset(p1, 0, sizeof(float)*c.count);
        memset(p2, 0, sizeof(float)*c.count);
        reset_memory(&c);
        /* p2=bar(u[n+1]), p1=bar(u[n]), p0=bar(u[n-1]). */
        #pragma omp parallel private(n)
        {
        for (n = nt-1; n >= 0; --n) {
            #pragma omp master
            {
            for (ir = 0; ir < nr; ++ir) {
                size_t sample = ((size_t)shot*nt+n)*nr+ir;
                i = (rz+c.offset)*c.nx+rx+ir*dr+c.offset;
                p2[i] += (synthetic[sample]-observed[sample])*(mask ? mask[sample] : 1.0f);
            }
            }
            #pragma omp barrier
            cpml_reverse_workshare(&c, p2, p1, p0,
                              history+(size_t)n*history_count, full_gradient);
            #pragma omp master
            {
            swap = p2; p2 = p1; p1 = p0; p0 = swap;
            }
            #pragma omp barrier
        }
        }
    }
    if (do_gradient) {
        /* E^T: sum all copied exterior sensitivities into their physical owner.
         * Serial fold avoids collisions at edges/corners. */
        double *physical = (double *)calloc((size_t)nx*nz, sizeof(double));
        if (!physical) goto done;
        for (z = c.halo; z < c.nz-c.halo; ++z)
            for (x = c.halo; x < c.nx-c.halo; ++x) {
                int xx = x-c.offset, zz = z-c.offset;
                if (xx < 0) xx = 0;
                if (xx >= nx) xx = nx-1;
                if (zz < 0) zz = 0;
                if (zz >= nz) zz = nz-1;
                physical[zz*nx+xx] += full_gradient[z*c.nx+x];
            }
        for (i = 0; i < nx*nz; ++i) gradient[i] += (float)physical[i];
        free(physical);
    }
    status = 0;
done:
    if (status) fprintf(stderr, "CPML failed: invalid parameters or insufficient memory.\n");
    free(p0); free(p1); free(p2); free(history); free(wavelet); free(full_gradient);
    cpml_free(&c);
    return status;
}
