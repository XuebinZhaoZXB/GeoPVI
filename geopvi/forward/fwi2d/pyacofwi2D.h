extern
void fwi_2D(char input_file[200], float *vel_inner, float *record_syn, float *record_obs, float *grad, 
                float *data_mask, int run_fwi, int verbose);
void forward_2D(char input_file[200], float *vel_inner, float *record_syn, int run_fwi, int verbose);
void forward_vd_2D(char input_file[200], const float *vel_inner, const float *rho_inner,
                   float *record_syn, int verbose);
void fwi_vd_2D(char input_file[200], const float *vel_inner, const float *rho_inner,
               float *record_syn, const float *record_obs, float *grad_velocity,
               float *grad_density, const float *data_mask, int verbose);
void vd_spatial_operator_2D(int nx, int nz, int lc, float dx, float dz,
                            const float *rho_inner, const float *pressure_inner,
                            float *operator_inner);
void vd_wavefield_snapshots_2D(char input_file[200], const float *vel_inner,
                               const float *rho_inner, float *record_syn,
                               const float *record_obs, int snapshot_count,
                               const int *snapshot_steps, float *forward_snapshots,
                               float *adjoint_snapshots, int verbose);
