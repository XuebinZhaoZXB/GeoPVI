/* Variable-density second-order pressure CPML, kappa=1.
 *
 *   g = G p; psi' = b_f psi + a_f g; q = B (g + psi');
 *   h = D q; eta' = b_c eta + a_c h;
 *   p' = 2p - p_old + dt^2 K (hx + etax' + hz + etaz') + dt^2 v^2 s.
 *
 * D=-G^T, B is interpolated 1/rho and K=rho*v^2.  Reverse propagation
 * transposes these operations, including all four memories.  The damping
 * profiles are held fixed in differentiation, even though vmax selects them.
 * Never mask derivative corrections at the physical/PML interface: the
 * stencil crosses it.  Pressure is zero only in the outer stencil halo.
 */
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "fdtd2D_modelling.h"
#include "vdcpml2D.h"

typedef struct {
    int nx, nz, X, Z, N, r, halo, offset, ax, az;
    float dx, dz, dt, d[6], interp[6];
    float *v, *rho, *K, *scale, *B[2], *flux[2];
    float *a[4], *b[4], *mem[4], *bar[4];
    float *u[2], *hbar[2], *saved_p;
    double *gK, *gB[2], *source_gv;
    int *indices[4], counts[4];
    size_t offsets[4], stride;
    void *allocations[64];
    int allocation_count;
} VdCpml;

static void *vdcp_alloc(VdCpml *c, size_t count, size_t size)
{
    void *p;
    if (count == 0 || size > (size_t)-1 / count || c->allocation_count >= 64)
        return NULL;
    p = calloc(count, size);
    if (p) c->allocations[c->allocation_count++] = p;
    return p;
}

static inline float vdcp_G(const VdCpml *c, const float *p, ptrdiff_t i,
                            ptrdiff_t step, float inv_h, int radius)
{
    float g = 0.0f;
    int k;
    for (k = 0; k < radius; ++k)
        g += c->d[k] * (p[i + (k+1)*step] - p[i-k*step]);
    return g * inv_h;
}

static inline float vdcp_D(const VdCpml *c, const float *q, ptrdiff_t i,
                            ptrdiff_t step, float inv_h, int radius)
{
    float h = 0.0f;
    int k;
    for (k = 0; k < radius; ++k)
        h += c->d[k] * (q[i+k*step] - q[i-(k+1)*step]);
    return h * inv_h;
}

static void vdcp_profile(int n, int offset, int width, float shift,
                          float spacing, float dt, float f0, float vmax,
                          float *a, float *b)
{
    int j;
    double sigma_max = -3.0 * vmax * log(1.e-4) / (2.0 * width * spacing);
    for (j = 0; j < n; ++j) {
        double pos = j + shift;
        double dist = fmax(offset-pos, pos-(n-offset-1));
        if (dist > 0.0) {
            double t = fmin(dist / width, 1.0);
            double sigma = sigma_max*t*t;
            double alpha = 3.14159265358979323846*f0*(1.0-t);
            b[j] = (float)exp(-(sigma+alpha)*dt);
            a[j] = (float)(sigma/(sigma+alpha)*expm1(-(sigma+alpha)*dt));
        }
    }
}

/* These worksharing routines are called by EVERY thread of a persistent
 * OpenMP team, in the same order.  No scatter/atomic model-gradient updates.
 * Distribute complete rows to threads and SIMD lanes along contiguous x;
 * avoid collapsed-loop index reconstruction and preserve stencil sum order. */
#define VDCP_RADIUS 1
#define VDCP_NAME(name) name##_r1
#include "vdcpml2D_kernels.h"
#undef VDCP_NAME
#undef VDCP_RADIUS

#define VDCP_RADIUS 2
#define VDCP_NAME(name) name##_r2
#include "vdcpml2D_kernels.h"
#undef VDCP_NAME
#undef VDCP_RADIUS

#define VDCP_RADIUS 3
#define VDCP_NAME(name) name##_r3
#include "vdcpml2D_kernels.h"
#undef VDCP_NAME
#undef VDCP_RADIUS

#define VDCP_RADIUS 4
#define VDCP_NAME(name) name##_r4
#include "vdcpml2D_kernels.h"
#undef VDCP_NAME
#undef VDCP_RADIUS

#define VDCP_RADIUS 5
#define VDCP_NAME(name) name##_r5
#include "vdcpml2D_kernels.h"
#undef VDCP_NAME
#undef VDCP_RADIUS

#define VDCP_RADIUS 6
#define VDCP_NAME(name) name##_r6
#include "vdcpml2D_kernels.h"
#undef VDCP_NAME
#undef VDCP_RADIUS

static void vdcp_step(VdCpml *c, const float *old, const float *p, float *next)
{
    switch (c->r) {
    case 1: vdcp_step_r1(c, old, p, next); break;
    case 2: vdcp_step_r2(c, old, p, next); break;
    case 3: vdcp_step_r3(c, old, p, next); break;
    case 4: vdcp_step_r4(c, old, p, next); break;
    case 5: vdcp_step_r5(c, old, p, next); break;
    case 6: vdcp_step_r6(c, old, p, next); break;
    }
}

static void vdcp_reverse(VdCpml *c, float *lambda, float *carry, int gradient)
{
    switch (c->r) {
    case 1:
        if (gradient) vdcp_reverse_r1(c, lambda, carry, 1);
        else vdcp_reverse_r1(c, lambda, carry, 0);
        break;
    case 2:
        if (gradient) vdcp_reverse_r2(c, lambda, carry, 1);
        else vdcp_reverse_r2(c, lambda, carry, 0);
        break;
    case 3:
        if (gradient) vdcp_reverse_r3(c, lambda, carry, 1);
        else vdcp_reverse_r3(c, lambda, carry, 0);
        break;
    case 4:
        if (gradient) vdcp_reverse_r4(c, lambda, carry, 1);
        else vdcp_reverse_r4(c, lambda, carry, 0);
        break;
    case 5:
        if (gradient) vdcp_reverse_r5(c, lambda, carry, 1);
        else vdcp_reverse_r5(c, lambda, carry, 0);
        break;
    case 6:
        if (gradient) vdcp_reverse_r6(c, lambda, carry, 1);
        else vdcp_reverse_r6(c, lambda, carry, 0);
        break;
    }
}

/* History contains active pressure and only nonzero CPML memory strips;
 * reused shot by shot.  Avoids three complete volume histories and avoids
 * unstable backward reconstruction by dividing damped memories by b. */
static void vdcp_history(VdCpml *c, float *p, float *history, int restore)
{
    int x, z, m, j;
#pragma omp for schedule(static) nowait
    for (z = 0; z < c->az; ++z) {
        #pragma omp simd
        for (x = 0; x < c->ax; ++x) {
            int i = (z+c->halo)*c->X+x+c->halo;
            size_t h = (size_t)z*c->ax+x;
            if (restore) p[i] = history[h];
            else history[h] = p[i];
        }
    }
    for (m = 0; m < 4; ++m) {
        #pragma omp for schedule(static) nowait
        for (j = 0; j < c->counts[m]; ++j) {
            int i = c->indices[m][j];
            if (restore) c->mem[m][i] = history[c->offsets[m]+j];
            else history[c->offsets[m]+j] = c->mem[m][i];
        }
    }
    #pragma omp barrier
}


static void vdcp_snapshot(const VdCpml *c, const float *p, float *out)
{
    int z;
    for (z = 0; z < c->nz; ++z)
        memcpy(out+(size_t)z*c->nx, p+(z+c->offset)*c->X+c->offset,
               (size_t)c->nx*sizeof(float));
}

int vdcpml_solve(char *path, const float *velocity, const float *density,
                 float *synthetic, const float *observed,
                 float *grad_velocity, float *grad_density, const float *mask,
                 int snapshot_count, const int *snapshot_steps,
                 float *forward_snapshots, float *adjoint_snapshots, int verbose)
{
    VdCpml c = {0};
    int width, solver, ns, nt, ds, sx, sz, rz, nr, dr, rx, interval;
    int i, m, x, z, shot, status = 1, gradient = grad_velocity != NULL;
    float f0, vmax, *buoyancy, *wavelet, *p0, *p1, *p2, *history = NULL;
    double sumd = 0.0, stability_bound = 0.0;
    double *out_v = NULL, *out_rho = NULL;
    size_t data_count;
    if (!path || !velocity || !density || !synthetic ||
        (gradient && (!grad_density || !observed))) return 1;
    read_parameters(path, &c.nx, &c.nz, &width, &c.r, &solver, &ns, &nt,
                    &ds, &sx, &sz, &rz, &nr, &dr, &rx, &interval,
                    &c.dx, &c.dz, &c.dt, &f0);
    if (c.nx < 3 || c.nz < 3 || width < 1 || c.r < 1 || c.r > 6 || solver != 0 ||
        ns < 1 || nt < 1 || nr < 1 || interval != 1 || ds < 0 || dr < 1 ||
        sx < 0 || (long long)sx+(long long)(ns-1)*ds >= c.nx ||
        rx < 0 || (long long)rx+(long long)(nr-1)*dr >= c.nx ||
        sz < 0 || sz >= c.nz || rz < 0 || rz >= c.nz ||
        !isfinite(c.dx) || !isfinite(c.dz) || !isfinite(c.dt) || !isfinite(f0) ||
        c.dx <= 0 || c.dz <= 0 || c.dt <= 0 || f0 <= 0) {
        fprintf(stderr, "Invalid variable-density CPML parameters (requires interval=1).\n");
        return 1;
    }
    if (snapshot_count < 0 || (snapshot_count &&
        (ns != 1 || !snapshot_steps || !forward_snapshots || !adjoint_snapshots || !observed))) return 1;
    for (i = 0; i < snapshot_count; ++i)
        if (snapshot_steps[i] < 0 || snapshot_steps[i] >= nt) return 1;
    c.halo = 2*c.r;
    if ((long long)c.nx+2LL*((long long)width+c.halo) > INT_MAX ||
        (long long)c.nz+2LL*((long long)width+c.halo) > INT_MAX) return 1;
    c.offset = width+c.halo;
    c.X = c.nx+2*c.offset; c.Z = c.nz+2*c.offset;
    if ((size_t)c.X*c.Z > INT_MAX) return 1;
    c.N = c.X*c.Z; c.ax = c.X-2*c.halo; c.az = c.Z-2*c.halo;
    if ((size_t)ns > (size_t)-1/(size_t)nt/(size_t)nr) return 1;
    data_count = (size_t)ns*nt*nr;
    vd_staggered_coefficients(c.r, c.d, c.interp);
    #define ALLOC(field, count, type) do { \
        field = vdcp_alloc(&c, (size_t)(count), sizeof(type)); \
        if (!(field)) goto cleanup; \
    } while (0)
    ALLOC(c.v, c.N, float); ALLOC(c.rho, c.N, float);
    ALLOC(c.K, c.N, float); ALLOC(c.scale, c.N, float);
    ALLOC(buoyancy, c.N, float);
    ALLOC(p0, c.N, float); ALLOC(p1, c.N, float); ALLOC(p2, c.N, float);
    ALLOC(wavelet, nt, float);
    if (!vd_extend_models(c.nx, c.nz, c.offset, c.X, c.Z, velocity, density,
                          c.v, c.rho, c.K, buoyancy, &vmax)) goto cleanup;
    for (m = 0; m < 2; ++m) {
        ALLOC(c.B[m], c.N, float); ALLOC(c.flux[m], c.N, float);
        if (observed) { ALLOC(c.hbar[m], c.N, float); }
        if (gradient) {
            ALLOC(c.u[m], c.N, float); ALLOC(c.gB[m], c.N, double);
        }
    }
    vd_prepare_face_buoyancies(c.X, c.Z, c.r, c.interp, buoyancy, c.B[0], c.B[1]);
    for (i = 0; i < c.N; ++i) {
        if (!isfinite(c.K[i]) || c.K[i] <= 0) goto cleanup;
        c.scale[i] = c.dt*c.dt*c.K[i];
    }
    for (m = 0; m < 4; ++m) {
        int horizontal = !(m & 1), n = horizontal ? c.X : c.Z;
        int xlo = c.halo, xhi = c.X-c.halo;
        int zlo = c.halo, zhi = c.Z-c.halo;
        ALLOC(c.a[m], n, float); ALLOC(c.b[m], n, float);
        ALLOC(c.mem[m], c.N, float);
        if (observed) { ALLOC(c.bar[m], c.N, float); }
        if (gradient) { ALLOC(c.indices[m], c.N, int); }
        vdcp_profile(n, c.offset, width, m < 2 ? 0.5f : 0.0f,
                      horizontal ? c.dx : c.dz, c.dt, f0, vmax, c.a[m], c.b[m]);
        if (m == 0) { xlo -= c.r; xhi += c.r-1; }
        if (m == 1) { zlo -= c.r; zhi += c.r-1; }
        for (z = zlo; z < zhi; ++z)
            for (x = xlo; x < xhi; ++x) {
                i = z*c.X+x;
                if (m < 2) {
                    float bface = c.B[m][i];
                    if (!isfinite(bface) || bface <= 0) {
                        fprintf(stderr, "CPML face buoyancy must be positive; reduce Lc or density contrast.\n");
                        goto cleanup;
                    }
                }
                if (gradient && c.a[m][horizontal ? x : z] != 0.0f)
                    c.indices[m][c.counts[m]++] = i;
            }
    }
    for (i = 0; i < c.r; ++i) sumd += fabs(c.d[i]);
    /* Gershgorin/triangle bound on dt^2 K G^T B G.  Using local face B
     * avoids the needlessly restrictive global max(K)*max(B) product when
     * the maximum density and minimum density are far apart. */
    for (z = c.halo; z < c.Z-c.halo; ++z)
        for (x = c.halo; x < c.X-c.halo; ++x) {
            int k;
            double bx = 0.0, bz = 0.0;
            i = z*c.X+x;
            for (k = 0; k < c.r; ++k) {
                bx += fabs(c.d[k])*(c.B[0][i+k]+c.B[0][i-1-k]);
                bz += fabs(c.d[k])*(c.B[1][i+k*c.X]+c.B[1][i-(1+k)*c.X]);
            }
            stability_bound = fmax(stability_bound, c.scale[i]*2.0*sumd*
                (bx/((double)c.dx*c.dx)+bz/((double)c.dz*c.dz)));
        }
    if (!isfinite(stability_bound) || stability_bound >= 4.0) {
        fprintf(stderr, "Variable-density CPML violates the conservative interior CFL bound.\n");
        goto cleanup;
    }
    c.stride = (size_t)c.ax*c.az;
    for (m = 0; m < 4; ++m) {
        c.offsets[m] = c.stride; c.stride += c.counts[m];
    }
    if (gradient) {
        if (c.stride > (size_t)-1/(size_t)nt) goto cleanup;
        ALLOC(history, c.stride*nt, float);
        ALLOC(c.saved_p, c.N, float); ALLOC(c.gK, c.N, double);
        ALLOC(c.source_gv, (size_t)c.nx*c.nz, double);
        ALLOC(out_v, (size_t)c.nx*c.nz, double);
        ALLOC(out_rho, (size_t)c.nx*c.nz, double);
    }
    getrik(nt, c.dt, f0, wavelet);
    if (verbose) printf("VD CPML: vmax=%.6g, R=1e-4, per-shot history=%.2f MiB\n",
                         vmax, gradient ? (double)c.stride*nt*sizeof(float)/1048576.0 : 0.0);
    for (shot = 0; shot < ns; ++shot) {
        int source_model = sz*c.nx+sx+shot*ds;
        int source = (sz+c.offset)*c.X+sx+shot*ds+c.offset;
        memset(p0, 0, (size_t)c.N*sizeof(float));
        memset(p1, 0, (size_t)c.N*sizeof(float));
        memset(p2, 0, (size_t)c.N*sizeof(float));
        for (m = 0; m < 4; ++m) memset(c.mem[m], 0, (size_t)c.N*sizeof(float));
        #pragma omp parallel
        {
            int t;
            for (t = 0; t < nt; ++t) {
                vdcp_step(&c, p0, p1, p2);
                if (gradient) vdcp_history(&c, p1, history+(size_t)t*c.stride, 0);
                #pragma omp single
                {
                    int ir, j;
                    float *tmp;
                    p2[source] += c.dt*c.dt*c.v[source]*c.v[source]*wavelet[t];
                    for (ir = 0; ir < nr; ++ir)
                        synthetic[((size_t)shot*nt+t)*nr+ir] =
                            p2[(rz+c.offset)*c.X+rx+ir*dr+c.offset];
                    for (j = 0; j < snapshot_count; ++j)
                        if (snapshot_steps[j] == t)
                            vdcp_snapshot(&c, p2, forward_snapshots+(size_t)j*c.nx*c.nz);
                    tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
                }
            }
        }
        if (!observed) continue;
        memset(p0, 0, (size_t)c.N*sizeof(float));
        memset(p1, 0, (size_t)c.N*sizeof(float));
        for (m = 0; m < 4; ++m) memset(c.bar[m], 0, (size_t)c.N*sizeof(float));
        #pragma omp parallel
        {
            int t;
            for (t = nt-1; t >= 0; --t) {
                #pragma omp single
                {
                    int ir, j;
                    for (ir = 0; ir < nr; ++ir) {
                        size_t di = ((size_t)shot*nt+t)*nr+ir;
                        float w = mask ? mask[di] : 1.0f;
                        p1[(rz+c.offset)*c.X+rx+ir*dr+c.offset] +=
                            (synthetic[di]-observed[di])*w*w;
                    }
                    if (gradient) c.source_gv[source_model] +=
                        2.0*c.v[source]*(c.dt*c.dt)*wavelet[t]*p1[source];
                    for (j = 0; j < snapshot_count; ++j)
                        if (snapshot_steps[j] == t)
                            vdcp_snapshot(&c, p1, adjoint_snapshots+(size_t)j*c.nx*c.nz);
                }
                if (gradient) vdcp_history(&c, c.saved_p, history+(size_t)t*c.stride, 1);
                vdcp_reverse(&c, p1, p0, gradient);
                #pragma omp single
                { float *tmp = p1; p1 = p0; p0 = tmp; }
            }
        }
    }
    if (gradient) {
        /* I^T folds face-buoyancy derivatives onto centres; E^T then folds
         * ALL copied model cells, including interpolation halo, to the model.
         * This one-time serial reduction is deterministic and not a hot loop. */
        for (z = 0; z < c.Z; ++z)
            for (x = 0; x < c.X; ++x) {
                int k, mx = x-c.offset, mz = z-c.offset, j;
                double gb = 0.0, v, rho;
                i = z*c.X+x;
                for (k = 0; k < c.r; ++k) {
                    double term = 0.0;
                    if (x+k < c.X) term += c.gB[0][i+k];
                    if (x-1-k >= 0) term += c.gB[0][i-1-k];
                    if (z+k < c.Z) term += c.gB[1][i+k*c.X];
                    if (z-1-k >= 0) term += c.gB[1][i-(1+k)*c.X];
                    gb += c.interp[k]*term;
                }
                if (mx < 0) mx = 0;
                if (mx >= c.nx) mx = c.nx-1;
                if (mz < 0) mz = 0;
                if (mz >= c.nz) mz = c.nz-1;
                j = mz*c.nx+mx; v = c.v[i]; rho = c.rho[i];
                out_v[j] += 2.0*rho*v*c.gK[i];
                out_rho[j] += v*v*c.gK[i]-gb/(rho*rho);
            }
        for (i = 0; i < c.nx*c.nz; ++i) {
            grad_velocity[i] = (float)(out_v[i]+c.source_gv[i]);
            grad_density[i] = (float)out_rho[i];
            if (!isfinite(grad_velocity[i]) || !isfinite(grad_density[i])) goto cleanup;
        }
    }
    {
        size_t j;
        for (j = 0; j < data_count; ++j)
            if (!isfinite(synthetic[j])) goto cleanup;
    }
    status = 0;
cleanup:
    if (status) fprintf(stderr, "Variable-density CPML failed (parameters, stability or allocation).\n");
    for (i = 0; i < c.allocation_count; ++i) free(c.allocations[i]);
    return status;
    #undef ALLOC
}
