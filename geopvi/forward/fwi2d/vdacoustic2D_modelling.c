#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fdtd2D_modelling.h"
#include "vdacoustic2D_modelling.h"

/*
 * Variable-density acoustic pressure equation, using the source convention of
 * the legacy constant-density code:
 *
 *   p_tt = rho*v^2 div(1/rho grad(p)) + v^2 s.
 *
 * The outer boundary is the legacy Liao/mixed ABC implemented by abc_for_p().
 */

static void *vd_malloc(size_t count, size_t size, const char *name)
{
    void *ptr;

    if (count != 0 && size > ((size_t)-1) / count) {
        fprintf(stderr, "Allocation size overflow for %s.\n", name);
        return NULL;
    }
    ptr = malloc(count * size);
    if (ptr == NULL) fprintf(stderr, "Cannot allocate %s.\n", name);
    return ptr;
}

static int vd_validate_parameters(int nx, int nz, int pml0, int ns, int nt,
                                  int ds, int ns0, int depths, int depthr,
                                  int nr, int dr, int nr0, float dx, float dz,
                                  float dt, float f0)
{
    if (nx < 3 || nz < 3 || pml0 < 0 || ns <= 0 || nt <= 0 || nr <= 0 ||
        ds < 0 || dr <= 0 || dx <= 0.0f || dz <= 0.0f || dt <= 0.0f ||
        f0 <= 0.0f) {
        fprintf(stderr, "Invalid variable-density modelling parameters.\n");
        return 0;
    }
    if (ns0 < 0 || ns0 + (ns - 1) * ds >= nx || depths < 0 || depths >= nz ||
        nr0 < 0 || nr0 + (nr - 1) * dr >= nx || depthr < 0 || depthr >= nz) {
        fprintf(stderr, "A source or receiver lies outside the physical model.\n");
        return 0;
    }
    return 1;
}

static int vd_extend_models(int nx, int nz, int pml, int ntx, int ntz,
                            const float *vel_inner, const float *rho_inner,
                            float *velocity, float *density, float *bulk,
                            float *buoyancy, float *vmax)
{
    int ix, iz;
    int valid = 1;
    float local_vmax = 0.0f;

    #pragma omp parallel for collapse(2) reduction(&:valid) reduction(max:local_vmax)
    for (iz = 0; iz < ntz; ++iz) {
        for (ix = 0; ix < ntx; ++ix) {
            int mx = ix - pml;
            int mz = iz - pml;
            int ip = iz * ntx + ix;
            float v, rho;

            if (mx < 0) mx = 0;
            if (mx >= nx) mx = nx - 1;
            if (mz < 0) mz = 0;
            if (mz >= nz) mz = nz - 1;
            v = vel_inner[mz * nx + mx];
            rho = rho_inner[mz * nx + mx];
            if (!isfinite(v) || !isfinite(rho) || v <= 0.0f || rho <= 0.0f) {
                valid = 0;
                v = 1.0f;
                rho = 1.0f;
            }
            velocity[ip] = v;
            density[ip] = rho;
            bulk[ip] = rho * v * v;
            buoyancy[ip] = 1.0f / rho;
            if (v > local_vmax) local_vmax = v;
        }
    }
    *vmax = local_vmax;
    if (!valid) fprintf(stderr, "Velocity and density must contain finite positive values.\n");
    return valid;
}

static int vd_prepare_liao_abc(int pml0, int ntx, int ntz, float dt, float dx,
                                const float *velocity, float **w_out,
                                float **t11_out, float **t12_out, float **t13_out)
{
    int ix, iz;
    int ntp = ntx * ntz;
    float *w = NULL;
    float *t11 = NULL, *t12 = NULL, *t13 = NULL;

    if (pml0 > 0) w = vd_malloc((size_t)pml0, sizeof(float), "Liao ABC weights");
    t11 = vd_malloc((size_t)ntp, sizeof(float), "Liao ABC t11");
    t12 = vd_malloc((size_t)ntp, sizeof(float), "Liao ABC t12");
    t13 = vd_malloc((size_t)ntp, sizeof(float), "Liao ABC t13");
    if ((pml0 > 0 && !w) || !t11 || !t12 || !t13) {
        free(w); free(t11); free(t12); free(t13);
        return 0;
    }
    for (ix = 0; ix < pml0; ++ix) w[ix] = 1.0f - (float)ix / (float)pml0;

    #pragma omp parallel for collapse(2)
    for (iz = 0; iz < ntz; ++iz) {
        for (ix = 0; ix < ntx; ++ix) {
            int ip = iz * ntx + ix;
            float s = velocity[ip] * dt / dx;
            t11[ip] = (2.0f - s) * (1.0f - s) / 2.0f;
            t12[ip] = s * (2.0f - s);
            t13[ip] = s * (s - 1.0f) / 2.0f;
        }
    }
    *w_out = w;
    *t11_out = t11;
    *t12_out = t12;
    *t13_out = t13;
    return 1;
}

/* Conservative second-order discretisation of div(b grad(p)). */
static float vd_div_b_grad_at(int ip, int ntx, float inv_dx2, float inv_dz2,
                              const float *buoyancy, const float *field)
{
    float bx_plus = 0.5f * (buoyancy[ip] + buoyancy[ip + 1]);
    float bx_minus = 0.5f * (buoyancy[ip] + buoyancy[ip - 1]);
    float bz_plus = 0.5f * (buoyancy[ip] + buoyancy[ip + ntx]);
    float bz_minus = 0.5f * (buoyancy[ip] + buoyancy[ip - ntx]);

    return (bx_plus * (field[ip + 1] - field[ip]) -
            bx_minus * (field[ip] - field[ip - 1])) * inv_dx2 +
           (bz_plus * (field[ip + ntx] - field[ip]) -
            bz_minus * (field[ip] - field[ip - ntx])) * inv_dz2;
}

static void vd_step(int ntx, int ntz, float dx, float dz, float dt,
                    const float *bulk, const float *buoyancy,
                    const float *p0, const float *p1, float *p2)
{
    int ix, iz;
    float inv_dx2 = 1.0f / (dx * dx);
    float inv_dz2 = 1.0f / (dz * dz);
    float dt2 = dt * dt;

    #pragma omp parallel for collapse(2)
    for (iz = 1; iz < ntz - 1; ++iz) {
        for (ix = 1; ix < ntx - 1; ++ix) {
            int ip = iz * ntx + ix;
            p2[ip] = 2.0f * p1[ip] - p0[ip] + dt2 * bulk[ip] *
                vd_div_b_grad_at(ip, ntx, inv_dx2, inv_dz2, buoyancy, p1);
        }
    }
}

static void vd_accumulate_gradients(int ntx, int ntz, int pml,
                                    float dx, float dz, float dt,
                                    int source_index, float source_value,
                                    const float *velocity, const float *density,
                                    const float *bulk, const float *buoyancy,
                                    const float *forward_p, const float *adjoint_p,
                                    float *grad_velocity, float *grad_density)
{
    int ix, iz;
    float inv_dx2 = 1.0f / (dx * dx);
    float inv_dz2 = 1.0f / (dz * dz);

    #pragma omp parallel for collapse(2)
    for (iz = pml; iz < ntz - pml; ++iz) {
        for (ix = pml; ix < ntx - pml; ++ix) {
            int ip = iz * ntx + ix;
            int model_ip = (iz - pml) * (ntx - 2 * pml) + (ix - pml);
            float spatial = vd_div_b_grad_at(ip, ntx, inv_dx2, inv_dz2,
                                             buoyancy, forward_p);
            float source = (ip == source_index) ? source_value : 0.0f;
            float p_tt = bulk[ip] * spatial + velocity[ip] * velocity[ip] * source;
            float rho = density[ip];
            float v = velocity[ip];

            /*
             * g_v = 2/(rho v^3) integral(lambda p_tt) dt in the
             * continuous objective.  The Python objective is an unweighted
             * time-sample sum, so its corresponding discrete factor is
             * absorbed into the adjoint-source convention used here.
             */
            grad_velocity[model_ip] +=
                2.0f * adjoint_p[ip] * p_tt / (rho * v * v * v);

            /*
             * The p_tt and source terms in the continuous rho-gradient
             * combine to adjoint_p * div(b grad(p)) / rho.  The remaining
             * derivative of face buoyancy is accumulated below, face by face,
             * with exactly the same arithmetic face average as vd_step().
             */
            grad_density[model_ip] += adjoint_p[ip] * spatial / rho;
        }
    }

    /* x-face contribution from d[0.5(1/rho_left + 1/rho_right)]/d rho. */
    for (iz = pml; iz < ntz - pml; ++iz) {
        for (ix = pml; ix < ntx - pml - 1; ++ix) {
            int left = iz * ntx + ix;
            int right = left + 1;
            int model_left = (iz - pml) * (ntx - 2 * pml) + (ix - pml);
            int model_right = model_left + 1;
            float coefficient = (adjoint_p[left] - adjoint_p[right]) *
                (forward_p[right] - forward_p[left]) * inv_dx2;
            grad_density[model_left] -= 0.5f * coefficient /
                                        (density[left] * density[left]);
            grad_density[model_right] -= 0.5f * coefficient /
                                         (density[right] * density[right]);
        }
    }

    /* z-face contribution from d[0.5(1/rho_top + 1/rho_bottom)]/d rho. */
    for (iz = pml; iz < ntz - pml - 1; ++iz) {
        for (ix = pml; ix < ntx - pml; ++ix) {
            int top = iz * ntx + ix;
            int bottom = top + ntx;
            int model_top = (iz - pml) * (ntx - 2 * pml) + (ix - pml);
            int model_bottom = model_top + (ntx - 2 * pml);
            float coefficient = (adjoint_p[top] - adjoint_p[bottom]) *
                (forward_p[bottom] - forward_p[top]) * inv_dz2;
            grad_density[model_top] -= 0.5f * coefficient /
                                       (density[top] * density[top]);
            grad_density[model_bottom] -= 0.5f * coefficient /
                                          (density[bottom] * density[bottom]);
        }
    }
}

void forward_vd_2D(char input_file[200], const float *vel_inner,
                   const float *rho_inner, float *record_syn, int verbose)
{
    int nx, nz, pml0, input_lc, laplace_solver, ns, nt, ds, ns0;
    int depths, depthr, nr, dr, nr0, nt_interval;
    int pml, ntx, ntz, ntp, is, it, ir;
    float dx, dz, dt, f0, vmax;
    float *velocity = NULL, *density = NULL, *bulk = NULL, *buoyancy = NULL;
    float *w = NULL, *t11 = NULL, *t12 = NULL, *t13 = NULL;
    float *p0 = NULL, *p1 = NULL, *p2 = NULL, *wavelet = NULL;

    read_parameters(input_file, &nx, &nz, &pml0, &input_lc, &laplace_solver,
                    &ns, &nt, &ds, &ns0, &depths, &depthr, &nr, &dr, &nr0,
                    &nt_interval, &dx, &dz, &dt, &f0);
    if (!vd_validate_parameters(nx, nz, pml0, ns, nt, ds, ns0, depths,
                                depthr, nr, dr, nr0, dx, dz, dt, f0)) return;

    /* Lc=1: one halo for the second-order conservative spatial stencil. */
    pml = pml0 + 1;
    ntx = nx + 2 * pml;
    ntz = nz + 2 * pml;
    if ((size_t)ntx > ((size_t)-1) / (size_t)ntz) return;
    ntp = ntx * ntz;

    velocity = vd_malloc(ntp, sizeof(float), "extended velocity");
    density = vd_malloc(ntp, sizeof(float), "extended density");
    bulk = vd_malloc(ntp, sizeof(float), "bulk modulus");
    buoyancy = vd_malloc(ntp, sizeof(float), "buoyancy");
    p0 = vd_malloc(ntp, sizeof(float), "forward p0");
    p1 = vd_malloc(ntp, sizeof(float), "forward p1");
    p2 = vd_malloc(ntp, sizeof(float), "forward p2");
    wavelet = vd_malloc(nt, sizeof(float), "Ricker wavelet");
    if (!velocity || !density || !bulk || !buoyancy || !p0 || !p1 || !p2 || !wavelet) goto cleanup;
    if (!vd_extend_models(nx, nz, pml, ntx, ntz, vel_inner, rho_inner,
                          velocity, density, bulk, buoyancy, &vmax)) goto cleanup;
    if (vmax * dt * sqrtf(1.0f / (dx * dx) + 1.0f / (dz * dz)) >= 1.0f) {
        fprintf(stderr, "Variable-density model violates the CFL condition.\n");
        goto cleanup;
    }
    if (!vd_prepare_liao_abc(pml0, ntx, ntz, dt, dx, velocity, &w, &t11, &t12, &t13)) goto cleanup;
    getrik(nt, dt, f0, wavelet);

    for (is = 0; is < ns; ++is) {
        int source_index = (pml + depths) * ntx + pml + ns0 + is * ds;
        memset(p0, 0, (size_t)ntp * sizeof(float));
        memset(p1, 0, (size_t)ntp * sizeof(float));
        memset(p2, 0, (size_t)ntp * sizeof(float));
        for (it = 0; it < nt; ++it) {
            float *tmp;
            vd_step(ntx, ntz, dx, dz, dt, bulk, buoyancy, p0, p1, p2);
            abc_for_p(ntx, ntz, 1, pml, p0, p1, p2, w, t11, t12, t13);
            p2[source_index] += dt * dt * velocity[source_index] * velocity[source_index] * wavelet[it];
            #pragma omp parallel for
            for (ir = 0; ir < nr; ++ir) {
                int receiver_index = (pml + depthr) * ntx + pml + nr0 + ir * dr;
                record_syn[((size_t)is * nt + it) * nr + ir] = p2[receiver_index];
            }
            tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
        }
    }

cleanup:
    free(velocity); free(density); free(bulk); free(buoyancy);
    free(w); free(t11); free(t12); free(t13);
    free(p0); free(p1); free(p2); free(wavelet);
}

void fwi_vd_2D(char input_file[200], const float *vel_inner,
               const float *rho_inner, float *record_syn,
               const float *record_obs, float *grad_velocity,
               float *grad_density, const float *data_mask, int verbose)
{
    int nx, nz, pml0, input_lc, laplace_solver, ns, nt, ds, ns0;
    int depths, depthr, nr, dr, nr0, nt_interval;
    int pml, ntx, ntz, ntp, is, it, ir;
    size_t wavefield_count;
    float dx, dz, dt, f0, vmax;
    float *velocity = NULL, *density = NULL, *bulk = NULL, *buoyancy = NULL;
    float *w = NULL, *t11 = NULL, *t12 = NULL, *t13 = NULL;
    float *p0 = NULL, *p1 = NULL, *p2 = NULL, *wavelet = NULL, *forward_p = NULL;
    float *adj0 = NULL, *adj1 = NULL, *adj2 = NULL;

    read_parameters(input_file, &nx, &nz, &pml0, &input_lc, &laplace_solver,
                    &ns, &nt, &ds, &ns0, &depths, &depthr, &nr, &dr, &nr0,
                    &nt_interval, &dx, &dz, &dt, &f0);
    if (!vd_validate_parameters(nx, nz, pml0, ns, nt, ds, ns0, depths,
                                depthr, nr, dr, nr0, dx, dz, dt, f0)) return;

    pml = pml0 + 1;
    ntx = nx + 2 * pml;
    ntz = nz + 2 * pml;
    if ((size_t)ntx > ((size_t)-1) / (size_t)ntz) return;
    ntp = ntx * ntz;
    wavefield_count = (size_t)ntp * (size_t)nt;

    velocity = vd_malloc(ntp, sizeof(float), "extended velocity");
    density = vd_malloc(ntp, sizeof(float), "extended density");
    bulk = vd_malloc(ntp, sizeof(float), "bulk modulus");
    buoyancy = vd_malloc(ntp, sizeof(float), "buoyancy");
    p0 = vd_malloc(ntp, sizeof(float), "forward p0");
    p1 = vd_malloc(ntp, sizeof(float), "forward p1");
    p2 = vd_malloc(ntp, sizeof(float), "forward p2");
    adj0 = vd_malloc(ntp, sizeof(float), "adjoint p0");
    adj1 = vd_malloc(ntp, sizeof(float), "adjoint p1");
    adj2 = vd_malloc(ntp, sizeof(float), "adjoint p2");
    wavelet = vd_malloc(nt, sizeof(float), "Ricker wavelet");
    forward_p = vd_malloc(wavefield_count, sizeof(float), "forward wavefield");
    if (!velocity || !density || !bulk || !buoyancy || !p0 || !p1 || !p2 || !adj0 || !adj1 || !adj2 || !wavelet || !forward_p) goto cleanup;
    if (!vd_extend_models(nx, nz, pml, ntx, ntz, vel_inner, rho_inner,
                          velocity, density, bulk, buoyancy, &vmax)) goto cleanup;
    if (vmax * dt * sqrtf(1.0f / (dx * dx) + 1.0f / (dz * dz)) >= 1.0f) {
        fprintf(stderr, "Variable-density model violates the CFL condition.\n");
        goto cleanup;
    }
    if (!vd_prepare_liao_abc(pml0, ntx, ntz, dt, dx, velocity, &w, &t11, &t12, &t13)) goto cleanup;
    getrik(nt, dt, f0, wavelet);
    memset(grad_velocity, 0, (size_t)nx * nz * sizeof(float));
    memset(grad_density, 0, (size_t)nx * nz * sizeof(float));

    for (is = 0; is < ns; ++is) {
        int source_index = (pml + depths) * ntx + pml + ns0 + is * ds;
        memset(p0, 0, (size_t)ntp * sizeof(float));
        memset(p1, 0, (size_t)ntp * sizeof(float));
        memset(p2, 0, (size_t)ntp * sizeof(float));
        for (it = 0; it < nt; ++it) {
            float *tmp;
            memcpy(forward_p + (size_t)it * ntp, p1, (size_t)ntp * sizeof(float));
            vd_step(ntx, ntz, dx, dz, dt, bulk, buoyancy, p0, p1, p2);
            abc_for_p(ntx, ntz, 1, pml, p0, p1, p2, w, t11, t12, t13);
            p2[source_index] += dt * dt * velocity[source_index] * velocity[source_index] * wavelet[it];
            #pragma omp parallel for
            for (ir = 0; ir < nr; ++ir) {
                int receiver_index = (pml + depthr) * ntx + pml + nr0 + ir * dr;
                record_syn[((size_t)is * nt + it) * nr + ir] = p2[receiver_index];
            }
            tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
        }

        memset(adj0, 0, (size_t)ntp * sizeof(float));
        memset(adj1, 0, (size_t)ntp * sizeof(float));
        memset(adj2, 0, (size_t)ntp * sizeof(float));
        for (it = nt - 1; it >= 0; --it) {
            float *tmp;
            vd_step(ntx, ntz, dx, dz, dt, bulk, buoyancy, adj0, adj1, adj2);
            abc_for_p(ntx, ntz, 1, pml, adj0, adj1, adj2, w, t11, t12, t13);
            for (ir = 0; ir < nr; ++ir) {
                size_t data_index = ((size_t)is * nt + it) * nr + ir;
                int receiver_index = (pml + depthr) * ntx + pml + nr0 + ir * dr;
                float mask = data_mask ? data_mask[data_index] : 1.0f;
                float residual = (record_syn[data_index] - record_obs[data_index]) * mask * mask;
                /* m lambda_tt - div(b grad lambda) = R^T residual. */
                adj2[receiver_index] += dt * dt * bulk[receiver_index] * residual;
            }
            vd_accumulate_gradients(ntx, ntz, pml, dx, dz, dt, source_index,
                                    wavelet[it], velocity, density, bulk, buoyancy,
                                    forward_p + (size_t)it * ntp, adj2,
                                    grad_velocity, grad_density);
            tmp = adj0; adj0 = adj1; adj1 = adj2; adj2 = tmp;
        }
    }
    if (verbose) printf("Variable-density forward and adjoint-gradient calculation finished.\n");

cleanup:
    free(velocity); free(density); free(bulk); free(buoyancy);
    free(w); free(t11); free(t12); free(t13);
    free(p0); free(p1); free(p2); free(adj0); free(adj1); free(adj2);
    free(wavelet); free(forward_p);
}
