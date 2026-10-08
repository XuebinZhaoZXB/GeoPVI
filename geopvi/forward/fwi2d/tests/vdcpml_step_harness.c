/* Include the implementation to test its actual internal step/transpose. */
#include "vdcpml2D.c"

int vdcpml_test_threads(int n)
{
    int actual = 0;
    omp_set_dynamic(0);
    omp_set_num_threads(n);
    #pragma omp parallel
    {
        #pragma omp single
        actual = omp_get_num_threads();
    }
    return actual;
}

static float sample(int i, int seed)
{
    return (float)(sin(0.731*(i+1)+seed)*0.7+cos(0.337*i-seed)*0.3);
}

double vdcpml_test_transpose(int radius)
{
    VdCpml c = {0};
    float *p, *old, *next, *lambda, *carry, *mem_before[4];
    double lhs = 0, rhs = 0;
    int i, m, x, z;
    c.r = radius; c.halo = 2*radius; c.offset = c.halo+4;
    c.nx = 9; c.nz = 7; c.X = c.nx+2*c.offset; c.Z = c.nz+2*c.offset;
    c.N = c.X*c.Z; c.dx = 1.1f; c.dz = 1.3f; c.dt = 0.07f;
    vd_staggered_coefficients(radius, c.d, c.interp);
    #define A(ptr) ptr = vdcp_alloc(&c, c.N, sizeof(float))
    A(p); A(old); A(next); A(lambda); A(carry); A(c.scale);
    for (m = 0; m < 2; ++m) {
        A(c.B[m]); A(c.flux[m]); A(c.hbar[m]);
        for (i = 0; i < c.N; ++i) c.B[m][i] = 1.0f+0.2f*sample(i, 2+m);
    }
    for (m = 0; m < 4; ++m) {
        int horizontal = !(m&1), n = horizontal ? c.X : c.Z;
        int xl = c.halo, xh = c.X-c.halo, zl = c.halo, zh = c.Z-c.halo;
        A(c.a[m]); A(c.b[m]); A(c.mem[m]); A(c.bar[m]); A(mem_before[m]);
        vdcp_profile(n, c.offset, 4, m < 2 ? 0.5f : 0.0f,
                      horizontal ? c.dx : c.dz, c.dt, 1.5f, 2.0f, c.a[m], c.b[m]);
        if (m == 0) { xl -= radius; xh += radius-1; }
        if (m == 1) { zl -= radius; zh += radius-1; }
        for (z = zl; z < zh; ++z)
            for (x = xl; x < xh; ++x) {
                i = z*c.X+x;
                if (c.a[m][horizontal ? x : z] != 0) {
                    c.mem[m][i] = mem_before[m][i] = sample(i, 10+m);
                    c.bar[m][i] = sample(i, 20+m);
                }
            }
    }
    for (z = c.halo; z < c.Z-c.halo; ++z)
        for (x = c.halo; x < c.X-c.halo; ++x) {
            i = z*c.X+x;
            p[i] = sample(i, 1); old[i] = sample(i, 2);
            lambda[i] = sample(i, 3); carry[i] = sample(i, 4);
            c.scale[i] = 0.01f*(1.0f+0.2f*sample(i, 5));
        }
    #pragma omp parallel
    { vdcp_step(&c, old, p, next); }
    for (i = 0; i < c.N; ++i) {
        lhs += (double)next[i]*lambda[i]+(double)p[i]*carry[i];
        for (m = 0; m < 4; ++m) lhs += (double)c.mem[m][i]*c.bar[m][i];
    }
    #pragma omp parallel
    { vdcp_reverse(&c, lambda, carry, 0); }
    for (i = 0; i < c.N; ++i) {
        rhs += (double)old[i]*lambda[i]+(double)p[i]*carry[i];
        for (m = 0; m < 4; ++m) rhs += (double)mem_before[m][i]*c.bar[m][i];
    }
    for (i = 0; i < c.allocation_count; ++i) free(c.allocations[i]);
    return fabs(lhs-rhs)/fmax(fmax(fabs(lhs), fabs(rhs)), 1.e-30);
    #undef A
}
