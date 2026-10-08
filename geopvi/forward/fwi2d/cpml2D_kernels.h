/* Private fixed-radius template; intentionally included for radii 3..6. */
static inline __attribute__((always_inline)) float CDCP_NAME(first_derivative)(const Cpml2D *c, const float *u, ptrdiff_t i, int d)
{
    ptrdiff_t stride = d == 0 ? 1 : c->nx;
    float result = 0.0f, h = d == 0 ? c->dx : c->dz;
    /* Supported radii are 3..6. Unroll in the ORIGINAL summation order;
     * no reassociation, reciprocal approximation or coefficient change. */
    result += c->first[1]*(u[i+stride]-u[i-stride]);
    result += c->first[2]*(u[i+2*stride]-u[i-2*stride]);
    result += c->first[3]*(u[i+3*stride]-u[i-3*stride]);
    if (CDCP_RADIUS >= 4) result += c->first[4]*(u[i+4*stride]-u[i-4*stride]);
    if (CDCP_RADIUS >= 5) result += c->first[5]*(u[i+5*stride]-u[i-5*stride]);
    if (CDCP_RADIUS >= 6) result += c->first[6]*(u[i+6*stride]-u[i-6*stride]);
    return result/h;
}

static inline __attribute__((always_inline)) float CDCP_NAME(second_derivative)(const Cpml2D *c, const float *u, ptrdiff_t i, int d)
{
    ptrdiff_t stride = d == 0 ? 1 : c->nx;
    float result = c->fd[0]*u[i], h = d == 0 ? c->dx : c->dz;
    result += c->fd[1]*(u[i+stride]+u[i-stride]);
    result += c->fd[2]*(u[i+2*stride]+u[i-2*stride]);
    result += c->fd[3]*(u[i+3*stride]+u[i-3*stride]);
    if (CDCP_RADIUS >= 4) result += c->fd[4]*(u[i+4*stride]+u[i-4*stride]);
    if (CDCP_RADIUS >= 5) result += c->fd[5]*(u[i+5*stride]+u[i-5*stride]);
    if (CDCP_RADIUS >= 6) result += c->fd[6]*(u[i+6*stride]+u[i-6*stride]);
    return result/h/h;
}


/* Modes: bit 0 enables the derivative of a memory across the interface;
 * bit 1 enables damping. Runs use the actual initialized coefficients, so
 * even underflow-to-zero coefficients and overlapping tiny-domain interfaces
 * preserve the old conditionals exactly. Stage/modes are compile-time literals
 * after inlining: no per-cell region decisions remain in the SIMD loop. */
static inline __attribute__((always_inline)) void CDCP_NAME(run)(
    Cpml2D *c, int z, int lo, int hi, const float *previous,
    const float *pressure, float *next, float *history,
    int stage, int xm, int zm)
{
    int x;
    #pragma omp simd
    for (x = lo; x < hi; ++x) {
        ptrdiff_t i = (ptrdiff_t)z*c->nx+x;
        if (stage == 0) {
            if (xm & 2) c->psi[0][i] = c->b[0][i]*c->psi[0][i] +
                c->a[0][i]*CDCP_NAME(first_derivative)(c, pressure, i, 0);
            if (zm & 2) c->psi[1][i] = c->b[1][i]*c->psi[1][i] +
                c->a[1][i]*CDCP_NAME(first_derivative)(c, pressure, i, 1);
        } else if (stage == 1) {
            float hx = CDCP_NAME(second_derivative)(c, pressure, i, 0);
            float hz = CDCP_NAME(second_derivative)(c, pressure, i, 1);
            if (xm & 1) hx += CDCP_NAME(first_derivative)(c, c->psi[0], i, 0);
            if (zm & 1) hz += CDCP_NAME(first_derivative)(c, c->psi[1], i, 1);
            if (xm & 2) {
                c->eta[0][i] = c->b[0][i]*c->eta[0][i]+c->a[0][i]*hx;
                hx += c->eta[0][i];
            }
            if (zm & 2) {
                c->eta[1][i] = c->b[1][i]*c->eta[1][i]+c->a[1][i]*hz;
                hz += c->eta[1][i];
            }
            float sum = hx;
            sum += hz;
            c->rhs[i] = sum;
            if (history)
                history[(size_t)(z-c->halo)*(c->nx-2*c->halo)+x-c->halo] = sum;
            next[i] = 2*pressure[i]-previous[i]+c->pressure_scale[i]*sum;
        } else if (stage == 2) {
            if (xm & 2) {
                float psi = c->psi[0][i]-CDCP_NAME(first_derivative)(c, c->hbar[0], i, 0);
                c->temporary[0][i] = c->a[0][i]*psi;
                c->psi[0][i] = c->b[0][i]*psi;
            } else {
                c->temporary[0][i] = 0.0f; c->psi[0][i] = 0.0f;
            }
            if (zm & 2) {
                float psi = c->psi[1][i]-CDCP_NAME(first_derivative)(c, c->hbar[1], i, 1);
                c->temporary[1][i] = c->a[1][i]*psi;
                c->psi[1][i] = c->b[1][i]*psi;
            } else {
                c->temporary[1][i] = 0.0f; c->psi[1][i] = 0.0f;
            }
        } else {
            float correction = 0.0f;
            if (xm & 1) correction = CDCP_NAME(first_derivative)(c, c->temporary[0], i, 0);
            next[i] += CDCP_NAME(second_derivative)(c, c->hbar[0], i, 0)-correction;
            correction = 0.0f;
            if (zm & 1) correction = CDCP_NAME(first_derivative)(c, c->temporary[1], i, 1);
            next[i] += CDCP_NAME(second_derivative)(c, c->hbar[1], i, 1)-correction;
        }
    }
}

static inline __attribute__((always_inline)) void CDCP_NAME(row)(
    Cpml2D *c, int z, const float *previous, const float *pressure,
    float *next, float *history, int stage, int zm)
{
    int s;
    for (s = 0; s < c->spans; ++s) {
        int lo = c->xcut[s], hi = c->xcut[s+1];
        switch (c->xflag[s]) {
        case 0: CDCP_NAME(run)(c,z,lo,hi,previous,pressure,next,history,stage,0,zm); break;
        case 1: CDCP_NAME(run)(c,z,lo,hi,previous,pressure,next,history,stage,1,zm); break;
        case 3: CDCP_NAME(run)(c,z,lo,hi,previous,pressure,next,history,stage,3,zm); break;
        }
    }
}

static inline __attribute__((always_inline)) void CDCP_NAME(row_dispatch)(
    Cpml2D *c, int z, const float *previous, const float *pressure,
    float *next, float *history, int stage)
{
    switch (c->zflag[z]) {
    case 0: CDCP_NAME(row)(c,z,previous,pressure,next,history,stage,0); break;
    case 1: CDCP_NAME(row)(c,z,previous,pressure,next,history,stage,1); break;
    case 3: CDCP_NAME(row)(c,z,previous,pressure,next,history,stage,3); break;
    }
}

/* Called collectively. Barriers preserve all neighbouring-memory dependencies. */
static inline __attribute__((always_inline)) void CDCP_NAME(cpml_pressure_workshare)(
    Cpml2D *c, const float *previous, const float *pressure, float *next, float *history)
{
    int z;
    #pragma omp for schedule(static)
    for (z = c->halo; z < c->nz-c->halo; ++z)
        CDCP_NAME(row_dispatch)(c,z,previous,pressure,next,history,0);
    #pragma omp for schedule(static)
    for (z = c->halo; z < c->nz-c->halo; ++z)
        CDCP_NAME(row_dispatch)(c,z,previous,pressure,next,history,1);
}

static inline __attribute__((always_inline)) void CDCP_NAME(cpml_reverse_workshare)(
    Cpml2D *c, const float *next, float *now, float *old,
    const float *history, double *gradient)
{
    int x,z;
    #pragma omp for schedule(static)
    for (z = c->halo; z < c->nz-c->halo; ++z) {
        #pragma omp simd
        for (x = c->halo; x < c->nx-c->halo; ++x) {
            ptrdiff_t i = (ptrdiff_t)z*c->nx+x;
            float q = c->pressure_scale[i]*next[i];
            if (gradient) {
                size_t h = (size_t)(z-c->halo)*(c->nx-2*c->halo)+x-c->halo;
                gradient[i] += c->gradient_scale[i]*history[h]*next[i];
            }
            now[i] += 2*next[i];
            old[i] = -next[i];
            for (int d = 0; d < 2; ++d) {
                float eta = c->eta[d][i]+q;
                c->hbar[d][i] = q+c->a[d][i]*eta;
                c->eta[d][i] = c->b[d][i]*eta;
            }
        }
    }
    #pragma omp for schedule(static)
    for (z = c->halo; z < c->nz-c->halo; ++z)
        CDCP_NAME(row_dispatch)(c,z,NULL,NULL,NULL,NULL,2);
    /* Caller publication barrier follows pointer rotation, as before. */
    #pragma omp for schedule(static) nowait
    for (z = c->halo; z < c->nz-c->halo; ++z)
        CDCP_NAME(row_dispatch)(c,z,NULL,NULL,now,NULL,3);
}
