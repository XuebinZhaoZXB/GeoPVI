#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fdtd2D_modelling.h"
#include "vdacoustic2D_modelling.h"

#define VD_PI 3.14159265358979323846

static void *vd_malloc(size_t count, size_t size, const char *name)
{
    void *ptr;

    if (count != 0 && size > ((size_t)-1) / count) {
        fprintf(stderr, "Allocation size overflow for %s.\n", name);
        return NULL;
    }
    ptr = malloc(count * size);
    if (ptr == NULL) {
        fprintf(stderr, "Cannot allocate %s (%zu bytes).\n", name, count * size);
    }
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

static int vd_extend_models(int nx, int nz, int pad, int ntx, int ntz,
                            const float *vel_inner, const float *rho_inner,
                            float *bulk, float *buoyancy, float *vmax)
{
    int ix, iz;
    int valid = 1;
    float local_vmax = 0.0f;

    #pragma omp parallel for collapse(2) reduction(&:valid) reduction(max:local_vmax)
    for (iz = 0; iz < ntz; ++iz) {
        for (ix = 0; ix < ntx; ++ix) {
            int mx = ix - pad;
            int mz = iz - pad;
            int model_index;
            int index = iz * ntx + ix;
            float velocity, density;

            if (mx < 0) mx = 0;
            if (mx >= nx) mx = nx - 1;
            if (mz < 0) mz = 0;
            if (mz >= nz) mz = nz - 1;
            model_index = mz * nx + mx;
            velocity = vel_inner[model_index];
            density = rho_inner[model_index];
            if (!isfinite(velocity) || !isfinite(density) ||
                velocity <= 0.0f || density <= 0.0f) {
                valid = 0;
                velocity = 1.0f;
                density = 1.0f;
            }
            bulk[index] = density * velocity * velocity;
            buoyancy[index] = 1.0f / density;
            if (velocity > local_vmax) local_vmax = velocity;
        }
    }
    *vmax = local_vmax;
    if (!valid) {
        fprintf(stderr, "Velocity and density must contain finite positive values.\n");
    }
    return valid;
}

static void vd_build_sponge(int ntx, int ntz, int pad, float *sponge)
{
    int ix, iz;

    #pragma omp parallel for collapse(2)
    for (iz = 0; iz < ntz; ++iz) {
        for (ix = 0; ix < ntx; ++ix) {
            int distance = ix;
            float x;

            if (ntx - 1 - ix < distance) distance = ntx - 1 - ix;
            if (iz < distance) distance = iz;
            if (ntz - 1 - iz < distance) distance = ntz - 1 - iz;
            if (distance >= pad || pad == 0) {
                sponge[iz * ntx + ix] = 1.0f;
            } else {
                x = (float)(pad - distance) / (float)pad;
                sponge[iz * ntx + ix] = expf(-0.015f * x * x);
            }
        }
    }
}

static void vd_step(int ntx, int ntz, float dx, float dz, float dt,
                    const float *bulk, const float *buoyancy,
                    const float *p0, const float *p1, float *p2)
{
    int ix, iz;
    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);
    const float dt2 = dt * dt;

    #pragma omp parallel for collapse(2)
    for (iz = 1; iz < ntz - 1; ++iz) {
        for (ix = 1; ix < ntx - 1; ++ix) {
            int ip = iz * ntx + ix;
            float bx_plus = 0.5f * (buoyancy[ip] + buoyancy[ip + 1]);
            float bx_minus = 0.5f * (buoyancy[ip] + buoyancy[ip - 1]);
            float bz_plus = 0.5f * (buoyancy[ip] + buoyancy[ip + ntx]);
            float bz_minus = 0.5f * (buoyancy[ip] + buoyancy[ip - ntx]);
            float div_b_grad_p;

            div_b_grad_p =
                (bx_plus * (p1[ip + 1] - p1[ip]) -
                 bx_minus * (p1[ip] - p1[ip - 1])) * inv_dx2 +
                (bz_plus * (p1[ip + ntx] - p1[ip]) -
                 bz_minus * (p1[ip] - p1[ip - ntx])) * inv_dz2;

            p2[ip] = 2.0f * p1[ip] - p0[ip] +
                     dt2 * bulk[ip] * div_b_grad_p;
        }
    }
}

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

static void vd_adjoint_step(int ntx, int ntz, float dx, float dz, float dt,
                            const float *bulk, const float *buoyancy,
                            const float *sponge, const float *lambda_a_next,
                            const float *lambda_b_next, float *lambda_a,
                            float *lambda_b, float *qbar, float *weighted_qbar)
{
    int ix, iz;
    int ntp = ntx * ntz;
    float inv_dx2 = 1.0f / (dx * dx);
    float inv_dz2 = 1.0f / (dz * dz);
    float dt2 = dt * dt;

    #pragma omp parallel for
    for (int ip = 0; ip < ntp; ++ip) {
        qbar[ip] = sponge[ip] * lambda_b_next[ip];
        weighted_qbar[ip] = dt2 * bulk[ip] * qbar[ip];
        lambda_a[ip] = 0.0f;
        lambda_b[ip] = 0.0f;
    }

    #pragma omp parallel for collapse(2)
    for (iz = 1; iz < ntz - 1; ++iz) {
        for (ix = 1; ix < ntx - 1; ++ix) {
            int ip = iz * ntx + ix;
            lambda_a[ip] = -qbar[ip];
            lambda_b[ip] = sponge[ip] * lambda_a_next[ip] + 2.0f * qbar[ip] +
                vd_div_b_grad_at(ip, ntx, inv_dx2, inv_dz2, buoyancy,
                                 weighted_qbar);
        }
    }
}

static void vd_accumulate_parameter_gradient(
    int ntx, int ntz, float dx, float dz, float dt, int source_index,
    float source_value, const float *velocity, const float *density,
    const float *buoyancy, const float *forward_p, const float *qbar,
    float *grad_velocity_ext, float *grad_density_ext)
{
    int ix, iz;
    float inv_dx2 = 1.0f / (dx * dx);
    float inv_dz2 = 1.0f / (dz * dz);
    float dt2 = dt * dt;

    /* Derivatives of the cell-centred K=rho*v^2 multiplier. */
    #pragma omp parallel for collapse(2)
    for (iz = 1; iz < ntz - 1; ++iz) {
        for (ix = 1; ix < ntx - 1; ++ix) {
            int ip = iz * ntx + ix;
            float spatial = vd_div_b_grad_at(ip, ntx, inv_dx2, inv_dz2,
                                             buoyancy, forward_p);
            grad_velocity_ext[ip] += dt2 * qbar[ip] *
                                     2.0f * density[ip] * velocity[ip] * spatial;
            grad_density_ext[ip] += dt2 * qbar[ip] *
                                    velocity[ip] * velocity[ip] * spatial;
        }
    }

    /* The legacy pressure source is dt^2*v^2*s(t). */
    grad_velocity_ext[source_index] +=
        dt2 * qbar[source_index] * 2.0f * velocity[source_index] * source_value;

    /* Derivative of face buoyancy averages with respect to cell density. */
    for (iz = 0; iz < ntz; ++iz) {
        for (ix = 0; ix < ntx - 1; ++ix) {
            int left = iz * ntx + ix;
            int right = left + 1;
            float h_left = dt2 * density[left] * velocity[left] * velocity[left] * qbar[left];
            float h_right = dt2 * density[right] * velocity[right] * velocity[right] * qbar[right];
            float coefficient = (h_left - h_right) *
                                (forward_p[right] - forward_p[left]) * inv_dx2;
            grad_density_ext[left] -= 0.5f * coefficient /
                                      (density[left] * density[left]);
            grad_density_ext[right] -= 0.5f * coefficient /
                                       (density[right] * density[right]);
        }
    }
    for (iz = 0; iz < ntz - 1; ++iz) {
        for (ix = 0; ix < ntx; ++ix) {
            int top = iz * ntx + ix;
            int bottom = top + ntx;
            float h_top = dt2 * density[top] * velocity[top] * velocity[top] * qbar[top];
            float h_bottom = dt2 * density[bottom] * velocity[bottom] * velocity[bottom] * qbar[bottom];
            float coefficient = (h_top - h_bottom) *
                                (forward_p[bottom] - forward_p[top]) * inv_dz2;
            grad_density_ext[top] -= 0.5f * coefficient /
                                     (density[top] * density[top]);
            grad_density_ext[bottom] -= 0.5f * coefficient /
                                        (density[bottom] * density[bottom]);
        }
    }
}

static void vd_fold_extended_gradient(int nx, int nz, int pad, int ntx, int ntz,
                                      const float *extended, float *inner)
{
    int ix, iz;

    for (iz = 0; iz < ntz; ++iz) {
        for (ix = 0; ix < ntx; ++ix) {
            int mx = ix - pad;
            int mz = iz - pad;
            if (mx < 0) mx = 0;
            if (mx >= nx) mx = nx - 1;
            if (mz < 0) mz = 0;
            if (mz >= nz) mz = nz - 1;
            inner[mz * nx + mx] += extended[iz * ntx + ix];
        }
    }
}

static void vd_apply_sponge(int ntp, const float *sponge, float *p0,
                            float *p1, float *p2)
{
    int ip;

    #pragma omp parallel for
    for (ip = 0; ip < ntp; ++ip) {
        p0[ip] *= sponge[ip];
        p1[ip] *= sponge[ip];
        p2[ip] *= sponge[ip];
    }
}

void forward_vd_2D(char input_file[200], const float *vel_inner,
                   const float *rho_inner, float *record_syn, int verbose)
{
    int nx, nz, pml0, input_lc, laplace_solver, ns, nt, ds, ns0;
    int depths, depthr, nr, dr, nr0, nt_interval;
    int pad, ntx, ntz, ntp;
    int is, it, ir;
    float dx, dz, dt, f0, vmax;
    float *bulk = NULL, *buoyancy = NULL, *sponge = NULL;
    float *p0 = NULL, *p1 = NULL, *p2 = NULL, *wavelet = NULL;

    read_parameters(input_file, &nx, &nz, &pml0, &input_lc, &laplace_solver,
                    &ns, &nt, &ds, &ns0, &depths, &depthr, &nr, &dr, &nr0,
                    &nt_interval, &dx, &dz, &dt, &f0);
    if (!vd_validate_parameters(nx, nz, pml0, ns, nt, ds, ns0, depths,
                                depthr, nr, dr, nr0, dx, dz, dt, f0)) {
        return;
    }

    /* One halo cell is required by the conservative second-order stencil. */
    pad = pml0 + 1;
    ntx = nx + 2 * pad;
    ntz = nz + 2 * pad;
    if ((size_t)ntx > ((size_t)-1) / (size_t)ntz) {
        fprintf(stderr, "Extended model dimensions overflow.\n");
        return;
    }
    ntp = ntx * ntz;

    bulk = vd_malloc((size_t)ntp, sizeof(float), "bulk modulus");
    buoyancy = vd_malloc((size_t)ntp, sizeof(float), "buoyancy");
    sponge = vd_malloc((size_t)ntp, sizeof(float), "sponge");
    p0 = vd_malloc((size_t)ntp, sizeof(float), "p0");
    p1 = vd_malloc((size_t)ntp, sizeof(float), "p1");
    p2 = vd_malloc((size_t)ntp, sizeof(float), "p2");
    wavelet = vd_malloc((size_t)nt, sizeof(float), "Ricker wavelet");
    if (!bulk || !buoyancy || !sponge || !p0 || !p1 || !p2 || !wavelet) {
        goto cleanup;
    }
    if (!vd_extend_models(nx, nz, pad, ntx, ntz, vel_inner, rho_inner,
                          bulk, buoyancy, &vmax)) {
        goto cleanup;
    }
    /* Conservative second-order 2-D stability limit. */
    if (vmax * dt * sqrtf(1.0f / (dx * dx) + 1.0f / (dz * dz)) >= 1.0f) {
        fprintf(stderr, "Variable-density model violates the CFL condition.\n");
        goto cleanup;
    }

    memset(p0, 0, (size_t)ntp * sizeof(float));
    memset(p1, 0, (size_t)ntp * sizeof(float));
    memset(p2, 0, (size_t)ntp * sizeof(float));
    getrik(nt, dt, f0, wavelet);
    vd_build_sponge(ntx, ntz, pad, sponge);

    if (verbose) {
        printf("Variable-density acoustic forward modelling\n");
        printf("nx=%d, nz=%d, nt=%d, ns=%d, nr=%d\n", nx, nz, nt, ns, nr);
        printf("dx=%g, dz=%g, dt=%g, vmax=%g\n", dx, dz, dt, vmax);
    }

    for (is = 0; is < ns; ++is) {
        int source_x = pad + ns0 + is * ds;
        int source_z = pad + depths;
        int source_index = source_z * ntx + source_x;

        memset(p0, 0, (size_t)ntp * sizeof(float));
        memset(p1, 0, (size_t)ntp * sizeof(float));
        memset(p2, 0, (size_t)ntp * sizeof(float));

        for (it = 0; it < nt; ++it) {
            float *tmp;

            vd_step(ntx, ntz, dx, dz, dt, bulk, buoyancy, p0, p1, p2);
            /* Match the legacy pressure-source convention in the constant-density limit. */
            p2[source_index] += dt * dt * bulk[source_index] *
                                buoyancy[source_index] * wavelet[it];
            vd_apply_sponge(ntp, sponge, p0, p1, p2);

            #pragma omp parallel for
            for (ir = 0; ir < nr; ++ir) {
                int receiver_x = pad + nr0 + ir * dr;
                int receiver_index = (pad + depthr) * ntx + receiver_x;
                record_syn[((size_t)is * nt + it) * nr + ir] = p2[receiver_index];
            }

            tmp = p0;
            p0 = p1;
            p1 = p2;
            p2 = tmp;
        }
    }

cleanup:
    free(bulk);
    free(buoyancy);
    free(sponge);
    free(p0);
    free(p1);
    free(p2);
    free(wavelet);
}

void fwi_vd_2D(char input_file[200], const float *vel_inner,
               const float *rho_inner, float *record_syn,
               const float *record_obs, float *grad_velocity,
               float *grad_density, const float *data_mask, int verbose)
{
    int nx, nz, pml0, input_lc, laplace_solver, ns, nt, ds, ns0;
    int depths, depthr, nr, dr, nr0, nt_interval;
    int pad, ntx, ntz, ntp, is, it, ir;
    size_t wavefield_count;
    float dx, dz, dt, f0, vmax;
    float *bulk = NULL, *buoyancy = NULL, *velocity = NULL, *density = NULL;
    float *sponge = NULL, *p0 = NULL, *p1 = NULL, *p2 = NULL;
    float *wavelet = NULL, *forward_wavefield = NULL;
    float *lambda_a = NULL, *lambda_b = NULL, *previous_a = NULL, *previous_b = NULL;
    float *qbar = NULL, *weighted_qbar = NULL;
    float *grad_velocity_ext = NULL, *grad_density_ext = NULL;

    read_parameters(input_file, &nx, &nz, &pml0, &input_lc, &laplace_solver,
                    &ns, &nt, &ds, &ns0, &depths, &depthr, &nr, &dr, &nr0,
                    &nt_interval, &dx, &dz, &dt, &f0);
    if (!vd_validate_parameters(nx, nz, pml0, ns, nt, ds, ns0, depths,
                                depthr, nr, dr, nr0, dx, dz, dt, f0)) return;

    pad = pml0 + 1;
    ntx = nx + 2 * pad;
    ntz = nz + 2 * pad;
    if ((size_t)ntx > ((size_t)-1) / (size_t)ntz) return;
    ntp = ntx * ntz;
    wavefield_count = (size_t)ntp * (size_t)nt;

    bulk = vd_malloc(ntp, sizeof(float), "bulk modulus");
    buoyancy = vd_malloc(ntp, sizeof(float), "buoyancy");
    velocity = vd_malloc(ntp, sizeof(float), "extended velocity");
    density = vd_malloc(ntp, sizeof(float), "extended density");
    sponge = vd_malloc(ntp, sizeof(float), "sponge");
    p0 = vd_malloc(ntp, sizeof(float), "p0");
    p1 = vd_malloc(ntp, sizeof(float), "p1");
    p2 = vd_malloc(ntp, sizeof(float), "p2");
    wavelet = vd_malloc(nt, sizeof(float), "Ricker wavelet");
    forward_wavefield = vd_malloc(wavefield_count, sizeof(float), "forward wavefield");
    lambda_a = vd_malloc(ntp, sizeof(float), "lambda a");
    lambda_b = vd_malloc(ntp, sizeof(float), "lambda b");
    previous_a = vd_malloc(ntp, sizeof(float), "previous lambda a");
    previous_b = vd_malloc(ntp, sizeof(float), "previous lambda b");
    qbar = vd_malloc(ntp, sizeof(float), "q adjoint");
    weighted_qbar = vd_malloc(ntp, sizeof(float), "weighted q adjoint");
    grad_velocity_ext = vd_malloc(ntp, sizeof(float), "extended velocity gradient");
    grad_density_ext = vd_malloc(ntp, sizeof(float), "extended density gradient");
    if (!bulk || !buoyancy || !velocity || !density || !sponge || !p0 || !p1 ||
        !p2 || !wavelet || !forward_wavefield || !lambda_a || !lambda_b ||
        !previous_a || !previous_b || !qbar || !weighted_qbar ||
        !grad_velocity_ext || !grad_density_ext) goto cleanup;

    if (!vd_extend_models(nx, nz, pad, ntx, ntz, vel_inner, rho_inner,
                          bulk, buoyancy, &vmax)) goto cleanup;
    for (int ip = 0; ip < ntp; ++ip) {
        density[ip] = 1.0f / buoyancy[ip];
        velocity[ip] = sqrtf(bulk[ip] * buoyancy[ip]);
    }
    if (vmax * dt * sqrtf(1.0f / (dx * dx) + 1.0f / (dz * dz)) >= 1.0f) {
        fprintf(stderr, "Variable-density model violates the CFL condition.\n");
        goto cleanup;
    }
    getrik(nt, dt, f0, wavelet);
    vd_build_sponge(ntx, ntz, pad, sponge);
    memset(grad_velocity, 0, (size_t)nx * nz * sizeof(float));
    memset(grad_density, 0, (size_t)nx * nz * sizeof(float));

    for (is = 0; is < ns; ++is) {
        int source_x = pad + ns0 + is * ds;
        int source_z = pad + depths;
        int source_index = source_z * ntx + source_x;

        memset(p0, 0, (size_t)ntp * sizeof(float));
        memset(p1, 0, (size_t)ntp * sizeof(float));
        memset(p2, 0, (size_t)ntp * sizeof(float));
        for (it = 0; it < nt; ++it) {
            float *tmp;
            memcpy(forward_wavefield + (size_t)it * ntp, p1,
                   (size_t)ntp * sizeof(float));
            vd_step(ntx, ntz, dx, dz, dt, bulk, buoyancy, p0, p1, p2);
            p2[source_index] += dt * dt * bulk[source_index] *
                                buoyancy[source_index] * wavelet[it];
            vd_apply_sponge(ntp, sponge, p0, p1, p2);
            #pragma omp parallel for
            for (ir = 0; ir < nr; ++ir) {
                int receiver_x = pad + nr0 + ir * dr;
                int receiver_index = (pad + depthr) * ntx + receiver_x;
                record_syn[((size_t)is * nt + it) * nr + ir] = p2[receiver_index];
            }
            tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
        }

        memset(lambda_a, 0, (size_t)ntp * sizeof(float));
        memset(lambda_b, 0, (size_t)ntp * sizeof(float));
        memset(grad_velocity_ext, 0, (size_t)ntp * sizeof(float));
        memset(grad_density_ext, 0, (size_t)ntp * sizeof(float));

        for (it = nt - 1; it >= 0; --it) {
            float *tmp;
            for (ir = 0; ir < nr; ++ir) {
                size_t data_index = ((size_t)is * nt + it) * nr + ir;
                int receiver_x = pad + nr0 + ir * dr;
                int receiver_index = (pad + depthr) * ntx + receiver_x;
                float mask = data_mask ? data_mask[data_index] : 1.0f;
                lambda_b[receiver_index] +=
                    (record_syn[data_index] - record_obs[data_index]) * mask * mask;
            }

            vd_adjoint_step(ntx, ntz, dx, dz, dt, bulk, buoyancy, sponge,
                            lambda_a, lambda_b, previous_a, previous_b,
                            qbar, weighted_qbar);
            vd_accumulate_parameter_gradient(
                ntx, ntz, dx, dz, dt, source_index, wavelet[it], velocity,
                density, buoyancy, forward_wavefield + (size_t)it * ntp,
                qbar, grad_velocity_ext, grad_density_ext);

            tmp = lambda_a; lambda_a = previous_a; previous_a = tmp;
            tmp = lambda_b; lambda_b = previous_b; previous_b = tmp;
        }
        vd_fold_extended_gradient(nx, nz, pad, ntx, ntz,
                                  grad_velocity_ext, grad_velocity);
        vd_fold_extended_gradient(nx, nz, pad, ntx, ntz,
                                  grad_density_ext, grad_density);
    }
    if (verbose) printf("Variable-density forward and gradient calculation finished.\n");

cleanup:
    free(bulk); free(buoyancy); free(velocity); free(density); free(sponge);
    free(p0); free(p1); free(p2); free(wavelet); free(forward_wavefield);
    free(lambda_a); free(lambda_b); free(previous_a); free(previous_b);
    free(qbar); free(weighted_qbar); free(grad_velocity_ext); free(grad_density_ext);
}
