/* Intentionally included once per radius. Fixed stencil lengths let the
 * compiler vectorize across x without changing a single stencil coefficient,
 * accumulation order, CPML memory operation or cross-interface coupling. */
#define VDCP_G(c,p,i,s,h) vdcp_G(c,p,i,s,h,VDCP_RADIUS)
#define VDCP_D(c,p,i,s,h) vdcp_D(c,p,i,s,h,VDCP_RADIUS)

static inline __attribute__((always_inline)) void VDCP_NAME(vdcp_faces)(VdCpml *c, const float *p, int replay)
{
    int x, z;
    const int H = c->halo, X = c->X, Z = c->Z, r = VDCP_RADIUS;
    const float idx = 1.0f/c->dx, idz = 1.0f/c->dz;
#pragma omp for schedule(static) nowait
    for (z = H; z < Z-H; ++z) {
        #pragma omp simd
        for (x = H-r; x < X-H+r-1; ++x) {
            ptrdiff_t i = (ptrdiff_t)z*X+x;
            float g = VDCP_G(c, p, i, 1, idx), u;
            if (!replay) c->mem[0][i] = c->b[0][x]*c->mem[0][i]+c->a[0][x]*g;
            u = g+c->mem[0][i];
            if (replay) c->u[0][i] = u;
            c->flux[0][i] = c->B[0][i]*u;
        }
    }
#pragma omp for schedule(static)
    for (z = H-r; z < Z-H+r-1; ++z) {
        #pragma omp simd
        for (x = H; x < X-H; ++x) {
            ptrdiff_t i = (ptrdiff_t)z*X+x;
            float g = VDCP_G(c, p, i, X, idz), u;
            if (!replay) c->mem[1][i] = c->b[1][z]*c->mem[1][i]+c->a[1][z]*g;
            u = g+c->mem[1][i];
            if (replay) c->u[1][i] = u;
            c->flux[1][i] = c->B[1][i]*u;
        }
    }
}

static void VDCP_NAME(vdcp_step)(VdCpml *c, const float *old, const float *p, float *next)
{
    int x, z;
    const int H = c->halo, X = c->X, Z = c->Z;
    const float idx = 1.0f/c->dx, idz = 1.0f/c->dz;
    VDCP_NAME(vdcp_faces)(c, p, 0);
#pragma omp for schedule(static)
    for (z = H; z < Z-H; ++z) {
        #pragma omp simd
        for (x = H; x < X-H; ++x) {
            ptrdiff_t i = (ptrdiff_t)z*X+x;
            float hx = VDCP_D(c, c->flux[0], i, 1, idx);
            float hz = VDCP_D(c, c->flux[1], i, X, idz);
            c->mem[2][i] = c->b[2][x]*c->mem[2][i]+c->a[2][x]*hx;
            c->mem[3][i] = c->b[3][z]*c->mem[3][i]+c->a[3][z]*hz;
            next[i] = 2.0f*p[i]-old[i]+c->scale[i]*
                (hx+c->mem[2][i]+hz+c->mem[3][i]);
        }
    }
}

static inline __attribute__((always_inline)) void VDCP_NAME(vdcp_reverse)(VdCpml *c, float *lambda, float *carry, int gradient)
{
    int x, z;
    const int H = c->halo, X = c->X, Z = c->Z, r = VDCP_RADIUS;
    const float idx = 1.0f/c->dx, idz = 1.0f/c->dz, dt2 = c->dt*c->dt;
    if (gradient) VDCP_NAME(vdcp_faces)(c, c->saved_p, 1);
#pragma omp for schedule(static)
    for (z = H; z < Z-H; ++z) {
        #pragma omp simd
        for (x = H; x < X-H; ++x) {
            ptrdiff_t i = (ptrdiff_t)z*X+x;
            float q = c->scale[i]*lambda[i];
            float tx = c->bar[2][i]+q, tz = c->bar[3][i]+q;
            c->hbar[0][i] = q+c->a[2][x]*tx;
            c->hbar[1][i] = q+c->a[3][z]*tz;
            c->bar[2][i] = c->b[2][x]*tx;
            c->bar[3][i] = c->b[3][z]*tz;
            if (gradient) {
                float hx = VDCP_D(c, c->flux[0], i, 1, idx);
                float hz = VDCP_D(c, c->flux[1], i, X, idz);
                c->gK[i] += (double)dt2*lambda[i]*
                    (hx+c->mem[2][i]+hz+c->mem[3][i]);
            }
        }
    }
#pragma omp for schedule(static) nowait
    for (z = H; z < Z-H; ++z) {
        #pragma omp simd
        for (x = H-r; x < X-H+r-1; ++x) {
            ptrdiff_t i = (ptrdiff_t)z*X+x;
            float qb = -VDCP_G(c, c->hbar[0], i, 1, idx);
            float ub = c->B[0][i]*qb, t = c->bar[0][i]+ub;
            if (gradient) c->gB[0][i] += (double)qb*c->u[0][i];
            c->flux[0][i] = ub+c->a[0][x]*t;
            c->bar[0][i] = c->b[0][x]*t;
        }
    }
#pragma omp for schedule(static)
    for (z = H-r; z < Z-H+r-1; ++z) {
        #pragma omp simd
        for (x = H; x < X-H; ++x) {
            ptrdiff_t i = (ptrdiff_t)z*X+x;
            float qb = -VDCP_G(c, c->hbar[1], i, X, idz);
            float ub = c->B[1][i]*qb, t = c->bar[1][i]+ub;
            if (gradient) c->gB[1][i] += (double)qb*c->u[1][i];
            c->flux[1][i] = ub+c->a[1][z]*t;
            c->bar[1][i] = c->b[1][z]*t;
        }
    }
#pragma omp for schedule(static)
    for (z = H; z < Z-H; ++z) {
        #pragma omp simd
        for (x = H; x < X-H; ++x) {
            ptrdiff_t i = (ptrdiff_t)z*X+x;
            carry[i] += 2.0f*lambda[i]-
                (VDCP_D(c, c->flux[0], i, 1, idx)+
                 VDCP_D(c, c->flux[1], i, X, idz));
            lambda[i] = -lambda[i];
        }
    }
}

#undef VDCP_G
#undef VDCP_D
