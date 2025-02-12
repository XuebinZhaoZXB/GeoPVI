import numpy as np
import torch
from torch.autograd import Function
from torch.multiprocessing import Pool

from pysurf96 import surf96     # this needs to be changed if the forward model is intergrated into GeoPVI
from geopvi.forward.tomo2d.fmm import fm2d

MAXIMUM_ITER = 10
MULTIPLY_FACTOR = 5


class ForwardModel(Function):
    @staticmethod
    def forward(ctx, input, func):
        output, grad = func(input)
        ctx.save_for_backward(input, torch.tensor(grad))
        return torch.tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        input, grad = ctx.saved_tensors
        # grad_input = (grad_output[...,None] * grad).sum(axis = -2)
        grad_input = (grad_output[...,None] * grad)
        return grad_input, None


# def forward_sw(vs, periods, thick, vp_vs = 1.76, relative_step = 0.01, wave = 'love', mode = 1, 
#                        velocity = 'group', requires_grad = True):
#     vp = vp_vs * vs
#     rho = 2.35 + 0.036 * (vp - 3.)**2
#     d_syn = surf96(thick, vp, vs, rho, periods, wave = wave, mode = mode, velocity = velocity)

#     gradient = np.zeros((len(periods), len(vs)))
#     if requires_grad:
#         for i in range(len(vs)):
#             vs_tmp = vs.copy()
#             step = relative_step * vs[i]
#             vs_tmp[i] += step
#             vp_tmp = vp_vs * vs_tmp
#             rho_tmp = 2.35 + 0.036 * (vp_tmp - 3.)**2
#             d_tmp = surf96(thick, vp_tmp, vs_tmp, rho_tmp, periods, wave = wave, mode = mode, velocity = velocity)
#             derivative = (d_tmp - d_syn) / abs(step)
#             gradient[:, i] = derivative

#     return d_syn, gradient


def forward_sw(vel, periods, thick, vp_vs = 1.76, relative_step = 0.005, wave = 'love', mode = 1, 
                       velocity = 'group', requires_grad = True, lower = None, upper = None):
    vs = vel.copy()
    perturb = relative_step
    while True:
        for ite in range(MAXIMUM_ITER):
            vp = vp_vs * vs
            rho = 2.35 + 0.036 * (vp - 3.)**2
            # vp = 1.16*vs + 1.36
            # rho = 1.74*vp**0.25
            d_syn = surf96(thick, vp, vs, rho, periods, wave = wave, mode = mode, velocity = velocity, flat_earth = False)
            if (d_syn == 0).all():
                vs = vel + np.random.normal(0, vel * perturb)
                if lower is not None:
                    vs = np.maximum(vs, lower)
                if upper is not None:    
                    vs = np.minimum(vs, upper)
            else:
                break
        if (d_syn == 0).all():
            perturb *= MULTIPLY_FACTOR
        else:
            break

    gradient = np.zeros((len(periods), len(vs)))
    if requires_grad:
        for i in range(len(vs)):
            vs_tmp = vs.copy()
            step = relative_step * vs[i]
            for ite in range(MAXIMUM_ITER):
                vs_tmp[i] = vs[i] + step
                if lower is not None:
                    vs_tmp[i] = np.maximum(vs_tmp[i], lower[i])
                if upper is not None:
                    vs_tmp[i] = np.minimum(vs_tmp[i], upper[i])
                vp_tmp = vp_vs * vs_tmp
                rho_tmp = 2.35 + 0.036 * (vp_tmp - 3.)**2
                # vp_tmp = 1.16 * vs_tmp + 1.36
                # rho_tmp = 1.74 * vp_tmp ** 0.25
                d_tmp = surf96(thick, vp_tmp, vs_tmp, rho_tmp, periods, wave = wave, mode = mode, velocity = velocity, flat_earth = False)   
                if (d_tmp == 0).all():
                    step = np.random.normal(0, 1.) * MULTIPLY_FACTOR * relative_step * vs[i]
                    gradient[:, i] = 0.
                else:
                    derivative = (d_tmp - d_syn) / step
                    gradient[:, i] = derivative
                    break

    return d_syn, gradient


class Posterior3D_dc():
    '''
    Calculate the log posterior and its gradient for 3D surface wave inversion
    Model is 3D S-wave velocity model, and data are dispersion curves at multiple locations
    Args:
        data: observed dispersion data at each frequency (m,): 1D array representing the observed phase/group velocities
        periods: period of each frequency (m,): 1D array representing the period of each frequency
        thick: thickness of each layer (n,): 1D array representing the thickness of each layer
        sigma: standard deviation of each data point (float): either array or scalar value
        log_prior: function to calculate the log prior (function): function to calculate the log prior of a given model sample
        num_processes: number of processes for parallel computation (int)
        wave: type of modelled wave (str): 'rayleigh' or 'love' or 'joint' representing joint inversion of Rayleigh and Love waves
        mode: mode of modelled wave (int): representing fundamental or first overtone mode
        velocity: type of modelled dispersion data (str): 'phase' or 'group' representing phase or group velocity
    '''
    def __init__(self, data, thick, periods, sigma = 0.003, log_prior = None, num_processes = 1, relative_step_grad = 0.001,
                         wave = 'rayleigh', mode = 1, velocity = 'phase'):
        self.log_prior = log_prior
        self.num_processes = num_processes

        if len(data) % len(periods) != 0:
            raise ValueError('Observed dispersion curve and defined periods have different sizes.')
        else:
            self.npoints = len(data) // len(periods)    # number of dispersion curves to be inverted

        # ensure sigma has the same size as data
        if np.isscalar(sigma):
            self.sigma = sigma * np.ones_like(data)
        elif isinstance(sigma, np.ndarray):
            if len(sigma) == len(periods):
                self.sigma = np.tile(sigma, self.npoints)
            elif len(sigma) == len(data):
                self.sigma = sigma
            else:
                raise ValueError('Incorrect size for data error sigma.')
        else:
            raise ValueError('Incorrect type for data error sigma.')
        self.data = data
        self.periods = periods
        self.thick = thick
        self.wave = wave
        self.mode = mode
        self.velocity = velocity
        self.relative_step_grad = relative_step_grad

    def solver1D(self, i, vs):
        '''
        1D forward simulation and gradient calculation given 1 single s-wave velocity profile
        by calling the external forward function (surf96)
        Input:
            vs: 1D array representing the S-wave velocity profile (nlayer,)
        Return:
            log_like: log likelihood of the given model, scalar
            dlog_like: gradient of log_like w.r.t the given model, 1D array (nlayer,)
        '''
        data = self.data[i * len(self.periods): (i + 1) * len(self.periods)]
        sigma = self.sigma[i * len(self.periods): (i + 1) * len(self.periods)]

        phase, gradient = forward_sw(vs, self.periods, self.thick, relative_step = self.relative_step_grad, wave = self.wave, 
                                            mode = self.mode, velocity = self.velocity, requires_grad = True)
        log_like = - 0.5 * np.sum(((data - phase)/sigma) ** 2, axis = -1)
        dlog_like = np.sum(((data - phase)/sigma**2)[..., None] * gradient, axis = -2)
        return log_like, dlog_like

    def solver3D(self, x):
        '''
        3D forward simulation and gradient calculation by calling multiple self.solver1D in parallel
        '''
        m, _ = x.shape
        total_profile = m * self.npoints
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy().reshape(total_profile, -1)
        log_like = np.zeros(total_profile)
        grad = np.zeros(x.shape)

        if self.num_processes == 1:
            for i in range(total_profile):
                log_like[i], grad[i] = self.solver1D(i % self.npoints, x[i])
        else:
            with Pool(processes = self.num_processes) as pool:
                results = pool.starmap(self.solver1D, [(i % self.npoints, x[i]) for i in range(total_profile)])
            for i in range(total_profile):
                log_like[i] = results[i][0]
                grad[i] = results[i][1]

        return log_like.reshape(m, -1).sum(axis = 1), grad.reshape(m, -1)
    
    def log_prob(self, x):
        """
        Calculate log posterior and its gradient directly from model x
        """
        if x.shape[1] != len(self.thick) * self.npoints:
            raise ValueError('Thickness and vs have different sizes.')
        log_like = ForwardModel.apply(x, self.solver3D)
        log_prior = self.log_prior(x)
        # # set a prior information: the top layer has minimum shear-velocity
        # # This ensures the computed phase velocities are phase velocities of Rayleigh or Love waves  
        # for i in range(x.shape[0]):
        #     if x[i, 1:].min() < x[i,0]:
        #         log_prior[i] -= 10.

        logp = log_like + log_prior
        return logp


class Posterior3D_tt():
    '''
    Calculate the log posterior and its gradient for 3D surface wave inversion
    Model is 3D S-wave velocity model, and data are travel times at different periods/frerquencies
    Args:
        data: travel time data at each frequency (nperiods, nt)
    '''
    def __init__(self, data, config, src, rec, mask, thick, periods, sigma = 0.003, log_prior = None, 
                        num_processes = 1, relative_step_grad = 0.001, wave = 'rayleigh', mode = 1, velocity = 'phase',
                        lower = None, upper = None):
        self.log_prior = log_prior
        self.num_processes = num_processes
        self.lower = lower
        self.upper = upper

        # data array is supposed to have a shape of (nperiods, nt)
        # sigma can be either a scalar, or an 1D array with the same size as periods, or a 2D array with the same size as data
        if data.shape[0] != len(periods):
            raise ValueError('Observed travel time data and defined periods have different sizes.')
        # ensure sigma has the same size as data
        if np.isscalar(sigma):
            self.sigma = sigma * np.ones_like(data)
        elif isinstance(sigma, np.ndarray):
            if sigma.ndim == 1 and len(sigma) == len(periods):
                self.sigma = np.broadcast_to(sigma[:,None], data.shape)
            elif sigma.shape == data.shape:
                self.sigma = sigma
            else:
                raise ValueError('Incorrect size for data error sigma.')
        else:
            raise ValueError('Incorrect type for data error sigma.')
        self.data = data

        # define parameters for fast marching method
        # mask is expected to have a shape of (nperiods, 2 * nsrc * nrec), each row defines a mask for each period
        self.mask = np.ascontiguousarray(mask)

        # the following parameters are the same as 2D travel time tomography
        # won't be changed in 3D inversion for each period
        self.nx = config.getint('FMM','nx')
        self.ny = config.getint('FMM','ny')
        self.xmin = config.getfloat('FMM','xmin')
        self.ymin = config.getfloat('FMM','ymin')
        self.dx = config.getfloat('FMM','dx')
        self.dy = config.getfloat('FMM','dy')
        self.gdx = config.getint('FMM','gdx')
        self.gdy = config.getint('FMM','gdy')
        self.sdx = config.getint('FMM','sdx')
        self.sext = config.getint('FMM','sext')
        self.earth = config.getfloat('FMM','earth')
        
        self.src = src
        self.rec = rec
        self.srcx = np.ascontiguousarray(src[:,0])
        self.srcy = np.ascontiguousarray(src[:,1])
        self.recx = np.ascontiguousarray(rec[:,0])
        self.recy = np.ascontiguousarray(rec[:,1])

        # define parameters for dispersion curve calculation
        self.npoints = self.nx * self.ny    # number of dispersion curves to be inverted
        self.periods = periods
        self.thick = thick
        self.wave = wave
        self.mode = mode
        self.velocity = velocity
        self.relative_step_grad = relative_step_grad

    def dispersion_modelling(self, vs):
        '''
        1D forward simulation and gradient calculation given 1 single s-wave velocity profile
        by calling the external forward function (surf96)
        '''
        phase, gradient = forward_sw(vs, self.periods, self.thick, relative_step = self.relative_step_grad, wave = self.wave, 
                                            mode = self.mode, velocity = self.velocity, requires_grad = True,
                                            lower = self.lower[:len(vs)], upper = self.upper[:len(vs)])
        return phase, gradient

    def fmm2d(self, index, vel):
        """
        2D fast marching method for travel time calculation and its gradient w.r.t. velocity
        Directly return the log-likelihood (negative misfit function) and gradient of the given model to save memory
        Args:
            index: index of which periods to be calculated (int)
            vel: 1D array representing the 2D velocity profile (nx*ny,)
        """
        time, dtdv = fm2d(vel, self.srcx, self.srcy, self.recx, self.recy, self.mask[index].reshape(2, -1),
                        self.nx, self.ny, self.xmin, self.ymin, self.dx, self.dy, 
                        self.gdx, self.gdy, self.sdx, self.sext, self.earth)

        data = self.data[index]
        sigma = self.sigma[index]
        log_like = - 0.5 * np.sum(((data - time)/sigma) ** 2, axis = -1)
        dlog_like = np.sum(((data - time)/sigma**2)[..., None] * dtdv, axis = -2)
        return log_like, dlog_like

    def solver3D(self, x):
        '''
        3D forward simulation and gradient calculation
        First perform surface wave dispersion curve calculation using forward_sw
        Then calculate the travel time data using fast marching method
        '''
        m, _ = x.shape
        total_profile = m * self.npoints
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy().reshape(total_profile, -1)
        disp_curve = np.zeros([total_profile, len(self.periods)])       # shape of (m * npoints, nperiods)
        grad_vs = np.zeros([total_profile, len(self.periods), x.shape[1]])      # shape of (m * npoints, nperiods, nz)

        if self.num_processes == 1:
            for i in range(total_profile):
                disp_curve[i], grad_vs[i] = self.solver1D(i % self.npoints, x[i])
            # TODO: implement the following FMM without parallelisation
        else:
            # 1. calculate dispersion curve data by calling dispersion_modelling
            with Pool(processes = self.num_processes) as pool:
                results = pool.map(self.dispersion_modelling, [x[i] for i in range(total_profile)])
                # results = pool.starmap(self.solver1D, [(i % self.npoints, x[i]) for i in range(total_profile)])
            for i in range(total_profile):
                disp_curve[i] = results[i][0]
                grad_vs[i] = results[i][1]
                # grad_vs now has a shape of (m * npoints, nperiods, nz)
            if (disp_curve == 0.).any():
                raise ValueError('0 occured in dispersion curve, meaning velocity model for FMM is not valid.')

            # 2. transpose dispersion curve to get 2D phase/group velocity maps
            disp_curve = disp_curve.reshape(m, self.npoints, -1).transpose(0, 2, 1).reshape(-1, self.npoints)
            # disp_curve now has a shape of (m * nperiods, npoints)

            # 3. calculate travel time data using fast marching method
            total_periods = m * len(self.periods)
            log_like = np.zeros(total_periods)      # shape of (m * nperiods,)
            grad_cg = np.zeros(disp_curve.shape)    # shape of (m * nperiods, npoints)
            with Pool(processes = self.num_processes) as pool:
                results = pool.starmap(self.fmm2d, [(i % len(self.periods), disp_curve[i]) for i in range(total_periods)])
            for i in range(total_periods):
                log_like[i] = results[i][0]
                grad_cg[i] = results[i][1]

            # 4. get final log likelihood and gradient
            log_like = log_like.reshape(m, -1).sum(axis = 1)
            grad_cg = grad_cg.reshape(m, len(self.periods), self.npoints).transpose(0, 2, 1).reshape(total_profile, -1)
            # grad_cg now has a shape of (m * npoints, nperiods)
            grad = np.sum(grad_cg[..., None] * grad_vs, axis = -2)      # shape of (m * npoints, nz)

        return log_like, grad.reshape(m, -1)

    def log_prob(self, x):
        """
        Calculate log posterior and its gradient directly from model x
        """
        if x.shape[1] != len(self.thick) * self.npoints:
            raise ValueError('Thickness and vs have different sizes.')
        log_like = ForwardModel.apply(x, self.solver3D)
        log_prior = self.log_prior(x)
        logp = log_like + log_prior
        return logp


class Posterior1D():
    '''
    Calculate the log posterior and its gradient for 1D surface wave inversion
    Input model is 1D S-wave velocity at one location, and output data are dispersion curves at the save location
    Args:
        data: observed dispersion data at each frequency (m,): 1D array representing the observed phase/group velocities
        periods: period of each frequency (m,): 1D array representing the period of each frequency
        thick: thickness of each layer (n,): 1D array representing the thickness of each layer
        sigma: standard deviation of each data point (float)
        log_prior: function to calculate the log prior (function): function to calculate the log prior of a given model sample
        num_processes: number of processes for parallel computation (int)
        wave: type of modelled wave (str): 'rayleigh' or 'love' or 'joint' representing joint inversion of Rayleigh and Love waves
        mode: mode of modelled wave (int): representing fundamental or first overtone mode
        velocity: type of modelled dispersion data (str): 'phase' or 'group' representing phase or group velocity
    '''
    def __init__(self, data, thick, periods, sigma = 0.003, log_prior = None, num_processes = 1, relative_step_grad = 0.005,
                         wave = 'rayleigh', mode = 1, velocity = 'phase'):
        self.log_prior = log_prior
        self.num_processes = num_processes
        self.sigma = sigma

        if len(data) != len(periods):
            raise ValueError('Observed dispersion curve and defined periods have different sizes.')
        self.data = data
        self.periods = periods
        self.thick = thick
        self.wave = wave
        self.mode = mode
        self.velocity = velocity
        self.relative_step_grad = relative_step_grad

    def solver(self, x):
        '''
        Calculate modelled data and data-model gradient by calling the external forward function (surf96)
        '''
        m, n = x.shape
        phase = np.zeros([m, self.data.shape[0]])
        gradient = np.zeros([m, self.data.shape[0], n])
        for i in range(m):
            vs = x.data.numpy()[i].squeeze()
            phase[i], gradient[i] = forward_sw(vs, self.periods, self.thick, relative_step = self.relative_step_grad, wave = self.wave, 
                                                mode = self.mode, velocity = self.velocity, requires_grad = True)
        log_like = - 0.5 * np.sum(((self.data - phase)/self.sigma) ** 2, axis = -1)
        dlog_like = np.sum(((self.data - phase)/self.sigma**2)[..., None] * gradient, axis = -2)
        return log_like, dlog_like
    
    def log_prob(self, x):
        """
        Calculate log posterior and its gradient directly from model x
        No paramellisation on different samples is needed for 1D inversion
        """
        if x.shape[1] != len(self.thick):
            raise ValueError('Thickness and vs have different sizes.')
        log_like = ForwardModel.apply(x, self.solver)
        log_prior = self.log_prior(x)
        # # set a prior information: the top layer has minimum shear-velocity
        # # This ensures the computed phase velocities are phase velocities of Rayleigh or Love waves  
        # for i in range(x.shape[0]):
        #     if x[i, 1:].min() < x[i,0]:
        #         log_prior[i] -= 10.

        logp = log_like + log_prior
        return logp