#ifndef VDACOUSTIC2D_MODELLING_H
#define VDACOUSTIC2D_MODELLING_H

void forward_vd_2D(char input_file[200], const float *vel_inner,
                   const float *rho_inner, float *record_syn, int verbose);
void fwi_vd_2D(char input_file[200], const float *vel_inner,
               const float *rho_inner, float *record_syn,
               const float *record_obs, float *grad_velocity,
               float *grad_density, const float *data_mask, int verbose);

#endif
