#ifndef GEOPVI_VDCPML2D_H
#define GEOPVI_VDCPML2D_H

/* Second-order pressure CPML.  Returns zero on success.  Optional snapshots
 * contain pressure and its discrete cotangent (not the Liao scaled adjoint). */
int vdcpml_solve(char *path, const float *velocity, const float *density,
                 float *synthetic, const float *observed,
                 float *grad_velocity, float *grad_density, const float *mask,
                 int snapshot_count, const int *snapshot_steps,
                 float *forward_snapshots, float *adjoint_snapshots, int verbose);

/* Shared model preparation with the legacy variable-density solver. */
int vd_staggered_coefficients(int radius, float *derivative, float *interpolation);
int vd_extend_models(int nx, int nz, int offset, int ntx, int ntz,
                     const float *v, const float *rho, float *velocity,
                     float *density, float *bulk, float *buoyancy, float *vmax);
void vd_prepare_face_buoyancies(int ntx, int ntz, int radius,
                                const float *interpolation, const float *buoyancy,
                                float *face_bx, float *face_bz);
#endif
