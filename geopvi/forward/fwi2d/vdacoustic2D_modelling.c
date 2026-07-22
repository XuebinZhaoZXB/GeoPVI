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

static int vd_validate_parameters(int nx, int nz, int pml0, int input_lc,
                                  int laplace_solver,
                                  int ns, int nt,
                                  int ds, int ns0, int depths, int depthr,
                                  int nr, int dr, int nr0, float dx, float dz,
                                  float dt, float f0)
{
    if (laplace_solver != 0) {
        fprintf(stderr, "Variable-density modelling supports only laplace_solver=0 (finite difference).\n");
        return 0;
    }
    if (input_lc < 1 || input_lc > 6) {
        fprintf(stderr, "Variable-density finite difference supports 1 <= Lc <= 6.\n");
        return 0;
    }
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

/*
 * Staggered-grid coefficients at x=i+1/2.  derivative[k] multiplies
 * (p[i+1+k]-p[i-k])/dx; interpolation[k] multiplies the corresponding
 * pair of cell-centred buoyancies.  Their matching adjoint divergence
 * makes -G^T B G a conservative, symmetric discrete operator.
 */
static int vd_staggered_coefficients(int radius, float *derivative,
                                     float *interpolation)
{
    static const float d[6][6] = {
        {1.0f},
        {9.0f / 8.0f, -1.0f / 24.0f},
        {75.0f / 64.0f, -25.0f / 384.0f, 3.0f / 640.0f},
        {1225.0f / 1024.0f, -245.0f / 3072.0f, 49.0f / 5120.0f, -5.0f / 7168.0f},
        {19845.0f / 16384.0f, -735.0f / 8192.0f, 567.0f / 40960.0f,
         -405.0f / 229376.0f, 35.0f / 294912.0f},
        {160083.0f / 131072.0f, -12705.0f / 131072.0f, 22869.0f / 1310720.0f,
         -5445.0f / 1835008.0f, 847.0f / 2359296.0f, -63.0f / 2883584.0f}
    };
    static const float b[6][6] = {
        {1.0f / 2.0f},
        {9.0f / 16.0f, -1.0f / 16.0f},
        {75.0f / 128.0f, -25.0f / 256.0f, 3.0f / 256.0f},
        {1225.0f / 2048.0f, -245.0f / 2048.0f, 49.0f / 2048.0f, -5.0f / 2048.0f},
        {19845.0f / 32768.0f, -2205.0f / 16384.0f, 567.0f / 16384.0f,
         -405.0f / 65536.0f, 35.0f / 65536.0f},
        {160083.0f / 262144.0f, -38115.0f / 262144.0f, 22869.0f / 524288.0f,
         -5445.0f / 524288.0f, 847.0f / 524288.0f, -63.0f / 524288.0f}
    };
    int k;

    if (radius < 1 || radius > 6) return 0;
    for (k = 0; k < radius; ++k) {
        derivative[k] = d[radius - 1][k];
        interpolation[k] = b[radius - 1][k];
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

static float vd_face_gradient_x(int ip, float inv_dx, int radius,
                                const float *derivative, const float *field)
{
    int k;
    float gradient = 0.0f;
    for (k = 0; k < radius; ++k)
        gradient += derivative[k] * (field[ip + 1 + k] - field[ip - k]);
    return gradient * inv_dx;
}

static float vd_face_gradient_z(int ip, int ntx, float inv_dz, int radius,
                                const float *derivative, const float *field)
{
    int k;
    float gradient = 0.0f;
    for (k = 0; k < radius; ++k)
        gradient += derivative[k] * (field[ip + (1 + k) * ntx] - field[ip - k * ntx]);
    return gradient * inv_dz;
}

static float vd_face_buoyancy_x(int ip, int radius, const float *interpolation,
                                const float *buoyancy)
{
    int k;
    float face_buoyancy = 0.0f;
    for (k = 0; k < radius; ++k)
        face_buoyancy += interpolation[k] * (buoyancy[ip - k] + buoyancy[ip + 1 + k]);
    return face_buoyancy;
}

static float vd_face_buoyancy_z(int ip, int ntx, int radius,
                                const float *interpolation, const float *buoyancy)
{
    int k;
    float face_buoyancy = 0.0f;
    for (k = 0; k < radius; ++k)
        face_buoyancy += interpolation[k] *
            (buoyancy[ip - k * ntx] + buoyancy[ip + (1 + k) * ntx]);
    return face_buoyancy;
}

static void vd_prepare_face_buoyancies(int ntx, int ntz, int radius,
                                       const float *interpolation,
                                       const float *buoyancy,
                                       float *face_bx, float *face_bz)
{
    int ix, iz;

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (iz = 0; iz < ntz; ++iz)
            for (ix = radius - 1; ix < ntx - radius; ++ix) {
                int ip = iz * ntx + ix;
                face_bx[ip] = vd_face_buoyancy_x(
                    ip, radius, interpolation, buoyancy
                );
            }
        #pragma omp for collapse(2)
        for (iz = radius - 1; iz < ntz - radius; ++iz)
            for (ix = 0; ix < ntx; ++ix) {
                int ip = iz * ntx + ix;
                face_bz[ip] = vd_face_buoyancy_z(
                    ip, ntx, radius, interpolation, buoyancy
                );
            }
    }
}

static void vd_compute_fluxes(int ntx, int ntz, int radius, int halo,
                              float inv_dx, float inv_dz,
                              const float *derivative,
                              const float *face_bx, const float *face_bz,
                              const float *field, float *flux_x, float *flux_z)
{
    int ix, iz;

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (iz = halo; iz < ntz - halo; ++iz)
            for (ix = halo - radius; ix < ntx - halo + radius - 1; ++ix) {
                int ip = iz * ntx + ix;
                flux_x[ip] = face_bx[ip] * vd_face_gradient_x(
                    ip, inv_dx, radius, derivative, field
                );
            }
        #pragma omp for collapse(2)
        for (iz = halo - radius; iz < ntz - halo + radius - 1; ++iz)
            for (ix = halo; ix < ntx - halo; ++ix) {
                int ip = iz * ntx + ix;
                flux_z[ip] = face_bz[ip] * vd_face_gradient_z(
                    ip, ntx, inv_dz, radius, derivative, field
                );
            }
    }
}

static void vd_compute_physical_face_gradients(
    int ntx, int ntz, int pml, int radius, float inv_dx, float inv_dz,
    const float *derivative, const float *field,
    float *gradient_x, float *gradient_z)
{
    int ix, iz;

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (iz = pml; iz < ntz - pml; ++iz)
            for (ix = pml; ix < ntx - pml - 1; ++ix) {
                int face = iz * ntx + ix;
                gradient_x[face] = vd_face_gradient_x(
                    face, inv_dx, radius, derivative, field
                );
            }
        #pragma omp for collapse(2)
        for (iz = pml; iz < ntz - pml - 1; ++iz)
            for (ix = pml; ix < ntx - pml; ++ix) {
                int face = iz * ntx + ix;
                gradient_z[face] = vd_face_gradient_z(
                    face, ntx, inv_dz, radius, derivative, field
                );
            }
    }
}

static float vd_div_flux_at(int ip, int ntx, int radius,
                            float inv_dx, float inv_dz,
                            const float *derivative,
                            const float *flux_x, const float *flux_z)
{
    int k;
    float div_x = 0.0f, div_z = 0.0f;

    for (k = 0; k < radius; ++k) {
        div_x += derivative[k] * (flux_x[ip + k] - flux_x[ip - 1 - k]);
        div_z += derivative[k] *
            (flux_z[ip + k * ntx] - flux_z[ip - (1 + k) * ntx]);
    }
    return div_x * inv_dx + div_z * inv_dz;
}

/* Conservative high-order discretisation: div(b grad(p)) = -G^T B G p. */
static float vd_div_b_grad_at(int ip, int ntx, int radius,
                              float inv_dx, float inv_dz,
                              const float *derivative, const float *interpolation,
                              const float *buoyancy, const float *field)
{
    int k;
    float div_x = 0.0f, div_z = 0.0f;

    for (k = 0; k < radius; ++k) {
        int right_face = ip + k;
        int left_face = ip - 1 - k;
        int bottom_face = ip + k * ntx;
        int top_face = ip - (1 + k) * ntx;
        float qx_right = vd_face_buoyancy_x(right_face, radius, interpolation, buoyancy) *
            vd_face_gradient_x(right_face, inv_dx, radius, derivative, field);
        float qx_left = vd_face_buoyancy_x(left_face, radius, interpolation, buoyancy) *
            vd_face_gradient_x(left_face, inv_dx, radius, derivative, field);
        float qz_bottom = vd_face_buoyancy_z(bottom_face, ntx, radius, interpolation, buoyancy) *
            vd_face_gradient_z(bottom_face, ntx, inv_dz, radius, derivative, field);
        float qz_top = vd_face_buoyancy_z(top_face, ntx, radius, interpolation, buoyancy) *
            vd_face_gradient_z(top_face, ntx, inv_dz, radius, derivative, field);

        div_x += derivative[k] * (qx_right - qx_left);
        div_z += derivative[k] * (qz_bottom - qz_top);
    }
    return div_x * inv_dx + div_z * inv_dz;
}

static void vd_step_from_flux(int ntx, int ntz, int radius, int halo,
                              float dx, float dz, float dt,
                              const float *derivative, const float *bulk,
                              const float *p0, const float *p1, float *p2,
                              const float *flux_x, const float *flux_z)
{
    int ix, iz;
    float inv_dx = 1.0f / dx;
    float inv_dz = 1.0f / dz;
    float dt2 = dt * dt;

    #pragma omp parallel for collapse(2)
    for (iz = halo; iz < ntz - halo; ++iz)
        for (ix = halo; ix < ntx - halo; ++ix) {
            int ip = iz * ntx + ix;
            p2[ip] = 2.0f * p1[ip] - p0[ip] + dt2 * bulk[ip] *
                vd_div_flux_at(ip, ntx, radius, inv_dx, inv_dz, derivative,
                               flux_x, flux_z);
        }
}

static void vd_step(int ntx, int ntz, int radius, int halo,
                    float dx, float dz, float dt,
                    const float *derivative, const float *face_bx,
                    const float *face_bz, const float *bulk,
                    const float *p0, const float *p1, float *p2,
                    float *flux_x, float *flux_z)
{
    vd_compute_fluxes(ntx, ntz, radius, halo, 1.0f / dx, 1.0f / dz,
                      derivative, face_bx, face_bz, p1, flux_x, flux_z);
    vd_step_from_flux(ntx, ntz, radius, halo, dx, dz, dt, derivative,
                      bulk, p0, p1, p2, flux_x, flux_z);
}

static void vd_accumulate_gradients(int ntx, int ntz, int pml, int radius,
                                    float dx, float dz, float inv_dt2,
                                    int source_index, float source_value,
                                    const float *velocity_acceleration_factor,
                                    const float *density_acceleration_factor,
                                    const float *density_face_factor,
                                    const float *face_bx, const float *face_bz,
                                    const float *derivative, const float *interpolation,
                                    const float *forward_previous,
                                    const float *forward_p,
                                    const float *forward_next,
                                    const float *adjoint_p,
                                    float *grad_velocity, float *grad_density,
                                    float *flux_x, float *flux_z,
                                    float *face_product_x, float *face_product_z,
                                    double *profile)
{
    int ix, iz, k;
    int nx = ntx - 2 * pml;
    float inv_dx = 1.0f / dx;
    float inv_dz = 1.0f / dz;
    double started = profile ? omp_get_wtime() : 0.0;

    vd_compute_physical_face_gradients(
        ntx, ntz, pml, radius, inv_dx, inv_dz, derivative, forward_p,
        face_product_x, face_product_z
    );
    if (profile) {
        profile[0] += omp_get_wtime() - started;
        started = omp_get_wtime();
    }
    #pragma omp parallel for collapse(2)
    for (iz = pml; iz < ntz - pml; ++iz) {
        for (ix = pml; ix < ntx - pml; ++ix) {
            int ip = iz * ntx + ix;
            int model_ip = (iz - pml) * (ntx - 2 * pml) + (ix - pml);
            float previous = forward_previous[ip];
            float acceleration =
                (forward_next[ip] - 2.0f * forward_p[ip] + previous) * inv_dt2;

            /*
             * g_v = 2/(rho v^3) integral(lambda p_tt) dt in the
             * continuous objective.  The Python objective is an unweighted
             * time-sample sum, so its corresponding discrete factor is
             * absorbed into the adjoint-source convention used here.
             */
            grad_velocity[model_ip] +=
                adjoint_p[ip] * acceleration *
                velocity_acceleration_factor[model_ip];

            /*
             * The acceleration contribution is lambda*p_tt/(rho^2*v^2).
             * Its source part is removed once below, leaving the equivalent
             * lambda*div(b grad(p))/rho term.  The derivative of interpolated
             * face buoyancy is accumulated separately after this loop.
             */
            grad_density[model_ip] += adjoint_p[ip] * acceleration *
                density_acceleration_factor[model_ip];
        }
    }
    {
        int source_model = (source_index / ntx - pml) * nx +
            (source_index % ntx - pml);
        grad_density[source_model] -= adjoint_p[source_index] * source_value *
            density_face_factor[source_model];
    }
    if (profile) {
        profile[1] += omp_get_wtime() - started;
        started = omp_get_wtime();
    }

    /* Reuse the saved forward gradients for the density face term.  At the
     * same time prepare b*G(adjoint) for the next reverse-time step. */
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (iz = 2 * radius; iz < ntz - 2 * radius; ++iz)
            for (ix = radius; ix < ntx - radius - 1; ++ix) {
                int face = iz * ntx + ix;
                float gradient = vd_face_gradient_x(
                    face, inv_dx, radius, derivative, adjoint_p
                );
                flux_x[face] = face_bx[face] * gradient;
                if (iz >= pml && iz < ntz - pml &&
                    ix >= pml && ix < ntx - pml - 1)
                    face_product_x[face] *= gradient;
            }
        #pragma omp for collapse(2)
        for (iz = radius; iz < ntz - radius - 1; ++iz)
            for (ix = 2 * radius; ix < ntx - 2 * radius; ++ix) {
                int face = iz * ntx + ix;
                float gradient = vd_face_gradient_z(
                    face, ntx, inv_dz, radius, derivative, adjoint_p
                );
                flux_z[face] = face_bz[face] * gradient;
                if (iz >= pml && iz < ntz - pml - 1 &&
                    ix >= pml && ix < ntx - pml)
                    face_product_z[face] *= gradient;
            }
    }
    if (profile) {
        profile[2] += omp_get_wtime() - started;
        started = omp_get_wtime();
    }

    /* Gather face contributions by model node.  This is algebraically the
     * same interpolation transpose as the former face-to-node scatter, but
     * every node is now independent and can be updated in parallel. */
    #pragma omp parallel for collapse(2)
    for (iz = pml; iz < ntz - pml; ++iz) {
        for (ix = pml; ix < ntx - pml; ++ix) {
            int ip = iz * ntx + ix;
            int model_ip = (iz - pml) * (ntx - 2 * pml) + (ix - pml);
            float face_term = 0.0f;
            for (k = 0; k < radius; ++k) {
                face_term += interpolation[k] * face_product_x[ip + k];
                face_term += interpolation[k] * face_product_x[ip - 1 - k];
                face_term += interpolation[k] * face_product_z[ip + k * ntx];
                face_term += interpolation[k] *
                    face_product_z[ip - (1 + k) * ntx];
            }
            grad_density[model_ip] += face_term * density_face_factor[model_ip];
        }
    }
    if (profile) profile[3] += omp_get_wtime() - started;
}

/* Diagnostic operator used by the manufactured-solution regression test.
 * It evaluates L_h p = div(b grad(p)) on the physical grid; no time stepping,
 * source injection, or absorbing boundary condition is involved. */
void vd_spatial_operator_2D(int nx, int nz, int lc, float dx, float dz,
                            const float *rho_inner, const float *pressure_inner,
                            float *operator_inner)
{
    int halo, ntx, ntz, ntp, ix, iz;
    float derivative[6], interpolation[6];
    float *density = NULL, *buoyancy = NULL, *pressure = NULL;

    if (nx < 1 || nz < 1 || dx <= 0.0f || dz <= 0.0f || lc < 1 || lc > 6 ||
        !rho_inner || !pressure_inner || !operator_inner) return;
    halo = 2 * lc;
    ntx = nx + 2 * halo;
    ntz = nz + 2 * halo;
    if ((size_t)ntx > ((size_t)-1) / (size_t)ntz ||
        !vd_staggered_coefficients(lc, derivative, interpolation)) return;
    ntp = ntx * ntz;
    density = vd_malloc((size_t)ntp, sizeof(float), "diagnostic density");
    buoyancy = vd_malloc((size_t)ntp, sizeof(float), "diagnostic buoyancy");
    pressure = vd_malloc((size_t)ntp, sizeof(float), "diagnostic pressure");
    if (!density || !buoyancy || !pressure) goto cleanup;

    for (iz = 0; iz < ntz; ++iz) {
        for (ix = 0; ix < ntx; ++ix) {
            int mx = ix - halo;
            int mz = iz - halo;
            int ip = iz * ntx + ix;
            float rho;
            if (mx < 0) mx = 0;
            if (mx >= nx) mx = nx - 1;
            if (mz < 0) mz = 0;
            if (mz >= nz) mz = nz - 1;
            rho = rho_inner[mz * nx + mx];
            if (!isfinite(rho) || rho <= 0.0f) {
                fprintf(stderr, "Density must contain finite positive values.\n");
                goto cleanup;
            }
            density[ip] = rho;
            buoyancy[ip] = 1.0f / rho;
            pressure[ip] = pressure_inner[mz * nx + mx];
        }
    }

    #pragma omp parallel for collapse(2)
    for (iz = 0; iz < nz; ++iz) {
        for (ix = 0; ix < nx; ++ix) {
            int ip = (iz + halo) * ntx + ix + halo;
            operator_inner[iz * nx + ix] = vd_div_b_grad_at(
                ip, ntx, lc, 1.0f / dx, 1.0f / dz, derivative,
                interpolation, buoyancy, pressure
            );
        }
    }

cleanup:
    free(density);
    free(buoyancy);
    free(pressure);
}

void forward_vd_2D(char input_file[200], const float *vel_inner,
                   const float *rho_inner, float *record_syn, int verbose)
{
    int nx, nz, pml0, input_lc, laplace_solver, ns, nt, ds, ns0;
    int depths, depthr, nr, dr, nr0, nt_interval;
    int halo, pml, ntx, ntz, ntp, is, it, ir;
    float dx, dz, dt, f0, vmax;
    float derivative[6], interpolation[6];
    float *velocity = NULL, *density = NULL, *bulk = NULL, *buoyancy = NULL;
    float *face_bx = NULL, *face_bz = NULL, *flux_x = NULL, *flux_z = NULL;
    float *w = NULL, *t11 = NULL, *t12 = NULL, *t13 = NULL;
    float *p0 = NULL, *p1 = NULL, *p2 = NULL, *wavelet = NULL;

    read_parameters(input_file, &nx, &nz, &pml0, &input_lc, &laplace_solver,
                    &ns, &nt, &ds, &ns0, &depths, &depthr, &nr, &dr, &nr0,
                    &nt_interval, &dx, &dz, &dt, &f0);
    if (!vd_validate_parameters(nx, nz, pml0, input_lc, laplace_solver, ns, nt, ds, ns0, depths,
                                depthr, nr, dr, nr0, dx, dz, dt, f0)) return;
    if (!vd_staggered_coefficients(input_lc, derivative, interpolation)) return;

    /* G and its adjoint divergence each span input_lc cells, so the composed
     * conservative operator needs a 2*Lc halo.  The remaining pml0 cells are
     * the legacy Liao/mixed absorbing layer. */
    halo = 2 * input_lc;
    pml = pml0 + halo;
    ntx = nx + 2 * pml;
    ntz = nz + 2 * pml;
    if ((size_t)ntx > ((size_t)-1) / (size_t)ntz) return;
    ntp = ntx * ntz;

    velocity = vd_malloc(ntp, sizeof(float), "extended velocity");
    density = vd_malloc(ntp, sizeof(float), "extended density");
    bulk = vd_malloc(ntp, sizeof(float), "bulk modulus");
    buoyancy = vd_malloc(ntp, sizeof(float), "buoyancy");
    face_bx = vd_malloc(ntp, sizeof(float), "x-face buoyancy");
    face_bz = vd_malloc(ntp, sizeof(float), "z-face buoyancy");
    flux_x = vd_malloc(ntp, sizeof(float), "x flux workspace");
    flux_z = vd_malloc(ntp, sizeof(float), "z flux workspace");
    p0 = vd_malloc(ntp, sizeof(float), "forward p0");
    p1 = vd_malloc(ntp, sizeof(float), "forward p1");
    p2 = vd_malloc(ntp, sizeof(float), "forward p2");
    wavelet = vd_malloc(nt, sizeof(float), "Ricker wavelet");
    if (!velocity || !density || !bulk || !buoyancy || !face_bx || !face_bz ||
        !flux_x || !flux_z || !p0 || !p1 || !p2 || !wavelet) goto cleanup;
    if (!vd_extend_models(nx, nz, pml, ntx, ntz, vel_inner, rho_inner,
                          velocity, density, bulk, buoyancy, &vmax)) goto cleanup;
    vd_prepare_face_buoyancies(ntx, ntz, input_lc, interpolation, buoyancy,
                               face_bx, face_bz);
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
            vd_step(ntx, ntz, input_lc, halo, dx, dz, dt, derivative,
                    face_bx, face_bz, bulk, p0, p1, p2, flux_x, flux_z);
            abc_for_p(ntx, ntz, halo, pml, p0, p1, p2, w, t11, t12, t13);
            p2[source_index] += dt * dt * velocity[source_index] * velocity[source_index] * wavelet[it];
            for (ir = 0; ir < nr; ++ir) {
                int receiver_index = (pml + depthr) * ntx + pml + nr0 + ir * dr;
                record_syn[((size_t)is * nt + it) * nr + ir] = p2[receiver_index];
            }
            tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
        }
    }

cleanup:
    free(velocity); free(density); free(bulk); free(buoyancy);
    free(face_bx); free(face_bz); free(flux_x); free(flux_z);
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
    int halo, pml, ntx, ntz, ntp, is, it, ir;
    size_t wavefield_count;
    float dx, dz, dt, f0, vmax, inv_dt2;
    float derivative[6], interpolation[6];
    float *velocity = NULL, *density = NULL, *bulk = NULL, *buoyancy = NULL;
    float *face_bx = NULL, *face_bz = NULL, *flux_x = NULL, *flux_z = NULL;
    float *face_product_x = NULL, *face_product_z = NULL;
    float *velocity_acceleration_factor = NULL;
    float *density_acceleration_factor = NULL;
    float *density_face_factor = NULL;
    float *w = NULL, *t11 = NULL, *t12 = NULL, *t13 = NULL;
    float *p0 = NULL, *p1 = NULL, *p2 = NULL, *wavelet = NULL, *forward_p = NULL;
    float *adj0 = NULL, *adj1 = NULL, *adj2 = NULL;
    double profile[10] = {0.0};

    read_parameters(input_file, &nx, &nz, &pml0, &input_lc, &laplace_solver,
                    &ns, &nt, &ds, &ns0, &depths, &depthr, &nr, &dr, &nr0,
                    &nt_interval, &dx, &dz, &dt, &f0);
    if (!vd_validate_parameters(nx, nz, pml0, input_lc, laplace_solver, ns, nt, ds, ns0, depths,
                                depthr, nr, dr, nr0, dx, dz, dt, f0)) return;
    if (!vd_staggered_coefficients(input_lc, derivative, interpolation)) return;
    inv_dt2 = 1.0f / (dt * dt);

    halo = 2 * input_lc;
    pml = pml0 + halo;
    ntx = nx + 2 * pml;
    ntz = nz + 2 * pml;
    if ((size_t)ntx > ((size_t)-1) / (size_t)ntz) return;
    ntp = ntx * ntz;
    wavefield_count = (size_t)ntp * (size_t)nt;

    velocity = vd_malloc(ntp, sizeof(float), "extended velocity");
    density = vd_malloc(ntp, sizeof(float), "extended density");
    bulk = vd_malloc(ntp, sizeof(float), "bulk modulus");
    buoyancy = vd_malloc(ntp, sizeof(float), "buoyancy");
    face_bx = vd_malloc(ntp, sizeof(float), "x-face buoyancy");
    face_bz = vd_malloc(ntp, sizeof(float), "z-face buoyancy");
    flux_x = vd_malloc(ntp, sizeof(float), "x flux workspace");
    flux_z = vd_malloc(ntp, sizeof(float), "z flux workspace");
    face_product_x = vd_malloc(ntp, sizeof(float), "x face-product workspace");
    face_product_z = vd_malloc(ntp, sizeof(float), "z face-product workspace");
    velocity_acceleration_factor = vd_malloc(
        (size_t)nx * nz, sizeof(float), "velocity acceleration-gradient factor"
    );
    density_acceleration_factor = vd_malloc(
        (size_t)nx * nz, sizeof(float), "density acceleration-gradient factor"
    );
    density_face_factor = vd_malloc((size_t)nx * nz, sizeof(float),
                                    "density face-gradient factor");
    p0 = vd_malloc(ntp, sizeof(float), "forward p0");
    p1 = vd_malloc(ntp, sizeof(float), "forward p1");
    p2 = vd_malloc(ntp, sizeof(float), "forward p2");
    adj0 = vd_malloc(ntp, sizeof(float), "adjoint p0");
    adj1 = vd_malloc(ntp, sizeof(float), "adjoint p1");
    adj2 = vd_malloc(ntp, sizeof(float), "adjoint p2");
    wavelet = vd_malloc(nt, sizeof(float), "Ricker wavelet");
    forward_p = vd_malloc(wavefield_count, sizeof(float), "forward wavefield");
    if (!velocity || !density || !bulk || !buoyancy || !face_bx || !face_bz ||
        !flux_x || !flux_z || !face_product_x || !face_product_z ||
        !velocity_acceleration_factor || !density_acceleration_factor ||
        !density_face_factor || !p0 || !p1 || !p2 || !adj0 || !adj1 ||
        !adj2 || !wavelet || !forward_p) goto cleanup;
    if (!vd_extend_models(nx, nz, pml, ntx, ntz, vel_inner, rho_inner,
                          velocity, density, bulk, buoyancy, &vmax)) goto cleanup;
    vd_prepare_face_buoyancies(ntx, ntz, input_lc, interpolation, buoyancy,
                               face_bx, face_bz);
    #pragma omp parallel for
    for (it = 0; it < nx * nz; ++it) {
        float v = vel_inner[it];
        float rho = rho_inner[it];
        velocity_acceleration_factor[it] = 2.0f / (rho * v * v * v);
        density_acceleration_factor[it] = 1.0f / (rho * rho * v * v);
        density_face_factor[it] = 1.0f / (rho * rho);
    }
    if (vmax * dt * sqrtf(1.0f / (dx * dx) + 1.0f / (dz * dz)) >= 1.0f) {
        fprintf(stderr, "Variable-density model violates the CFL condition.\n");
        goto cleanup;
    }
    if (!vd_prepare_liao_abc(pml0, ntx, ntz, dt, dx, velocity, &w, &t11, &t12, &t13)) goto cleanup;
    getrik(nt, dt, f0, wavelet);
    memset(grad_velocity, 0, (size_t)nx * nz * sizeof(float));
    memset(grad_density, 0, (size_t)nx * nz * sizeof(float));
    memset(face_product_x, 0, (size_t)ntp * sizeof(float));
    memset(face_product_z, 0, (size_t)ntp * sizeof(float));

    for (is = 0; is < ns; ++is) {
        int source_index = (pml + depths) * ntx + pml + ns0 + is * ds;
        memset(p0, 0, (size_t)ntp * sizeof(float));
        memset(p1, 0, (size_t)ntp * sizeof(float));
        memset(p2, 0, (size_t)ntp * sizeof(float));
        for (it = 0; it < nt; ++it) {
            float *tmp;
            double started = verbose ? omp_get_wtime() : 0.0;
            memcpy(forward_p + (size_t)it * ntp, p1, (size_t)ntp * sizeof(float));
            if (verbose) {
                profile[4] += omp_get_wtime() - started;
                started = omp_get_wtime();
            }
            vd_step(ntx, ntz, input_lc, halo, dx, dz, dt, derivative,
                    face_bx, face_bz, bulk, p0, p1, p2, flux_x, flux_z);
            if (verbose) {
                profile[5] += omp_get_wtime() - started;
                started = omp_get_wtime();
            }
            abc_for_p(ntx, ntz, halo, pml, p0, p1, p2, w, t11, t12, t13);
            p2[source_index] += dt * dt * velocity[source_index] * velocity[source_index] * wavelet[it];
            for (ir = 0; ir < nr; ++ir) {
                int receiver_index = (pml + depthr) * ntx + pml + nr0 + ir * dr;
                record_syn[((size_t)is * nt + it) * nr + ir] = p2[receiver_index];
            }
            if (verbose) profile[6] += omp_get_wtime() - started;
            tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
        }

        memset(adj0, 0, (size_t)ntp * sizeof(float));
        memset(adj1, 0, (size_t)ntp * sizeof(float));
        memset(adj2, 0, (size_t)ntp * sizeof(float));
        memset(flux_x, 0, (size_t)ntp * sizeof(float));
        memset(flux_z, 0, (size_t)ntp * sizeof(float));
        for (it = nt - 1; it >= 0; --it) {
            float *tmp;
            double started = verbose ? omp_get_wtime() : 0.0;
            vd_step_from_flux(ntx, ntz, input_lc, halo, dx, dz, dt,
                              derivative, bulk, adj0, adj1, adj2,
                              flux_x, flux_z);
            if (verbose) {
                profile[7] += omp_get_wtime() - started;
                started = omp_get_wtime();
            }
            abc_for_p(ntx, ntz, halo, pml, adj0, adj1, adj2, w, t11, t12, t13);
            for (ir = 0; ir < nr; ++ir) {
                size_t data_index = ((size_t)is * nt + it) * nr + ir;
                int receiver_index = (pml + depthr) * ntx + pml + nr0 + ir * dr;
                float mask = data_mask ? data_mask[data_index] : 1.0f;
                float residual = (record_syn[data_index] - record_obs[data_index]) * mask * mask;
                /* m lambda_tt - div(b grad lambda) = R^T residual. */
                adj2[receiver_index] += dt * dt * bulk[receiver_index] * residual;
            }
            if (verbose) profile[8] += omp_get_wtime() - started;
            vd_accumulate_gradients(ntx, ntz, pml, input_lc, dx, dz,
                                    inv_dt2, source_index,
                                    wavelet[it], velocity_acceleration_factor,
                                    density_acceleration_factor, density_face_factor,
                                    face_bx, face_bz,
                                    derivative, interpolation,
                                    it > 0 ? forward_p + (size_t)(it - 1) * ntp : forward_p,
                                    forward_p + (size_t)it * ntp,
                                    it < nt - 1 ? forward_p + (size_t)(it + 1) * ntp : p1,
                                    adj2,
                                    grad_velocity, grad_density, flux_x, flux_z,
                                    face_product_x, face_product_z,
                                    verbose ? profile : NULL);
            tmp = adj0; adj0 = adj1; adj1 = adj2; adj2 = tmp;
        }
    }
    if (verbose) {
        double gradient_total = profile[0] + profile[1] + profile[2] + profile[3];
        printf("Variable-density timing (seconds):\n");
        printf("  forward wavefield save:       %.6f\n", profile[4]);
        printf("  forward propagation:          %.6f\n", profile[5]);
        printf("  forward ABC/source/receivers: %.6f\n", profile[6]);
        printf("  adjoint propagation:          %.6f\n", profile[7]);
        printf("  adjoint ABC/residual source:  %.6f\n", profile[8]);
        printf("  gradient forward face deriv.: %.6f\n", profile[0]);
        printf("  gradient point terms:         %.6f\n", profile[1]);
        printf("  gradient face products:       %.6f\n", profile[2]);
        printf("  gradient face gather:         %.6f\n", profile[3]);
        printf("  gradient total:               %.6f\n", gradient_total);
        printf("Variable-density forward and adjoint-gradient calculation finished.\n");
    }

cleanup:
    free(velocity); free(density); free(bulk); free(buoyancy);
    free(face_bx); free(face_bz); free(flux_x); free(flux_z);
    free(face_product_x); free(face_product_z);
    free(velocity_acceleration_factor); free(density_acceleration_factor);
    free(density_face_factor);
    free(w); free(t11); free(t12); free(t13);
    free(p0); free(p1); free(p2); free(adj0); free(adj1); free(adj2);
    free(wavelet); free(forward_p);
}

void vd_wavefield_snapshots_2D(char input_file[200], const float *vel_inner,
                               const float *rho_inner, float *record_syn,
                               const float *record_obs, int snapshot_count,
                               const int *snapshot_steps, float *forward_snapshots,
                               float *adjoint_snapshots, int verbose)
{
    int nx, nz, pml0, input_lc, laplace_solver, ns, nt, ds, ns0;
    int depths, depthr, nr, dr, nr0, nt_interval;
    int halo, pml, ntx, ntz, ntp, is, it, ir, snap;
    float dx, dz, dt, f0, vmax;
    float derivative[6], interpolation[6];
    float *velocity = NULL, *density = NULL, *bulk = NULL, *buoyancy = NULL;
    float *face_bx = NULL, *face_bz = NULL, *flux_x = NULL, *flux_z = NULL;
    float *w = NULL, *t11 = NULL, *t12 = NULL, *t13 = NULL;
    float *p0 = NULL, *p1 = NULL, *p2 = NULL, *wavelet = NULL;
    float *adj0 = NULL, *adj1 = NULL, *adj2 = NULL;

    read_parameters(input_file, &nx, &nz, &pml0, &input_lc, &laplace_solver,
                    &ns, &nt, &ds, &ns0, &depths, &depthr, &nr, &dr, &nr0,
                    &nt_interval, &dx, &dz, &dt, &f0);
    if (!vd_validate_parameters(nx, nz, pml0, input_lc, laplace_solver, ns, nt, ds, ns0, depths,
                                depthr, nr, dr, nr0, dx, dz, dt, f0) || ns != 1 ||
        snapshot_count <= 0 || !snapshot_steps || !forward_snapshots || !adjoint_snapshots) return;
    for (snap = 0; snap < snapshot_count; ++snap)
        if (snapshot_steps[snap] < 0 || snapshot_steps[snap] >= nt) return;
    if (!vd_staggered_coefficients(input_lc, derivative, interpolation)) return;
    halo = 2 * input_lc;
    pml = pml0 + halo;
    ntx = nx + 2 * pml;
    ntz = nz + 2 * pml;
    ntp = ntx * ntz;
    velocity = vd_malloc(ntp, sizeof(float), "snapshot extended velocity");
    density = vd_malloc(ntp, sizeof(float), "snapshot extended density");
    bulk = vd_malloc(ntp, sizeof(float), "snapshot bulk modulus");
    buoyancy = vd_malloc(ntp, sizeof(float), "snapshot buoyancy");
    face_bx = vd_malloc(ntp, sizeof(float), "snapshot x-face buoyancy");
    face_bz = vd_malloc(ntp, sizeof(float), "snapshot z-face buoyancy");
    flux_x = vd_malloc(ntp, sizeof(float), "snapshot x flux workspace");
    flux_z = vd_malloc(ntp, sizeof(float), "snapshot z flux workspace");
    p0 = vd_malloc(ntp, sizeof(float), "snapshot forward p0");
    p1 = vd_malloc(ntp, sizeof(float), "snapshot forward p1");
    p2 = vd_malloc(ntp, sizeof(float), "snapshot forward p2");
    adj0 = vd_malloc(ntp, sizeof(float), "snapshot adjoint p0");
    adj1 = vd_malloc(ntp, sizeof(float), "snapshot adjoint p1");
    adj2 = vd_malloc(ntp, sizeof(float), "snapshot adjoint p2");
    wavelet = vd_malloc(nt, sizeof(float), "snapshot Ricker wavelet");
    if (!velocity || !density || !bulk || !buoyancy || !face_bx || !face_bz ||
        !flux_x || !flux_z || !p0 || !p1 || !p2 || !adj0 || !adj1 || !adj2 ||
        !wavelet) goto cleanup;
    if (!vd_extend_models(nx, nz, pml, ntx, ntz, vel_inner, rho_inner,
                          velocity, density, bulk, buoyancy, &vmax)) goto cleanup;
    vd_prepare_face_buoyancies(ntx, ntz, input_lc, interpolation, buoyancy,
                               face_bx, face_bz);
    if (vmax * dt * sqrtf(1.0f / (dx * dx) + 1.0f / (dz * dz)) >= 1.0f) goto cleanup;
    if (!vd_prepare_liao_abc(pml0, ntx, ntz, dt, dx, velocity, &w, &t11, &t12, &t13)) goto cleanup;
    getrik(nt, dt, f0, wavelet);

    for (is = 0; is < ns; ++is) {
        int source_index = (pml + depths) * ntx + pml + ns0 + is * ds;
        memset(p0, 0, (size_t)ntp * sizeof(float));
        memset(p1, 0, (size_t)ntp * sizeof(float));
        memset(p2, 0, (size_t)ntp * sizeof(float));
        for (it = 0; it < nt; ++it) {
            float *tmp;
            vd_step(ntx, ntz, input_lc, halo, dx, dz, dt, derivative,
                    face_bx, face_bz, bulk, p0, p1, p2, flux_x, flux_z);
            abc_for_p(ntx, ntz, halo, pml, p0, p1, p2, w, t11, t12, t13);
            p2[source_index] += dt * dt * velocity[source_index] * velocity[source_index] * wavelet[it];
            for (ir = 0; ir < nr; ++ir) {
                int receiver_index = (pml + depthr) * ntx + pml + nr0 + ir * dr;
                record_syn[(size_t)it * nr + ir] = p2[receiver_index];
            }
            for (snap = 0; snap < snapshot_count; ++snap) if (snapshot_steps[snap] == it) {
                int iz, ix;
                for (iz = 0; iz < nz; ++iz)
                    for (ix = 0; ix < nx; ++ix)
                        forward_snapshots[(size_t)snap * nx * nz + iz * nx + ix] =
                            p2[(iz + pml) * ntx + ix + pml];
            }
            tmp = p0; p0 = p1; p1 = p2; p2 = tmp;
        }
        memset(adj0, 0, (size_t)ntp * sizeof(float));
        memset(adj1, 0, (size_t)ntp * sizeof(float));
        memset(adj2, 0, (size_t)ntp * sizeof(float));
        for (it = nt - 1; it >= 0; --it) {
            float *tmp;
            vd_step(ntx, ntz, input_lc, halo, dx, dz, dt, derivative,
                    face_bx, face_bz, bulk, adj0, adj1, adj2, flux_x, flux_z);
            abc_for_p(ntx, ntz, halo, pml, adj0, adj1, adj2, w, t11, t12, t13);
            for (ir = 0; ir < nr; ++ir) {
                int receiver_index = (pml + depthr) * ntx + pml + nr0 + ir * dr;
                float residual = record_syn[(size_t)it * nr + ir] - record_obs[(size_t)it * nr + ir];
                adj2[receiver_index] += dt * dt * bulk[receiver_index] * residual;
            }
            for (snap = 0; snap < snapshot_count; ++snap) if (snapshot_steps[snap] == it) {
                int iz, ix;
                for (iz = 0; iz < nz; ++iz)
                    for (ix = 0; ix < nx; ++ix)
                        adjoint_snapshots[(size_t)snap * nx * nz + iz * nx + ix] =
                            adj2[(iz + pml) * ntx + ix + pml];
            }
            tmp = adj0; adj0 = adj1; adj1 = adj2; adj2 = tmp;
        }
    }
    if (verbose) printf("Variable-density wavefield snapshots finished.\n");
cleanup:
    free(velocity); free(density); free(bulk); free(buoyancy);
    free(face_bx); free(face_bz); free(flux_x); free(flux_z);
    free(w); free(t11); free(t12); free(t13);
    free(p0); free(p1); free(p2); free(adj0); free(adj1); free(adj2); free(wavelet);
}
