import numpy as np
import torch
from torch.autograd import Function
from torch.multiprocessing import Pool

from pysurf96 import surf96     # this needs to be changed if the forward model is intergrated into GeoPVI


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


def forward_sw(vs, periods, thick, vp_vs = 1.76, relative_step = 0.01, wave = 'love', mode = 1, 
                       velocity = 'group', requires_grad = True):
    vp = vp_vs * vs
    rho = 2.35 + 0.036 * (vp - 3.)**2
    d_syn = surf96(thick, vp, vs, rho, periods, wave = wave, mode = mode, velocity = velocity)

    gradient = np.zeros((len(periods), len(vs)))
    if requires_grad:
        for i in range(len(vs)):
            vs_tmp = vs.copy()
            step = relative_step * vs[i]
            vs_tmp[i] += step
            vp_tmp = vp_vs * vs_tmp
            rho_tmp = 2.35 + 0.036 * (vp_tmp - 3.)**2
            d_tmp = surf96(thick, vp_tmp, vs_tmp, rho_tmp, periods, wave = wave, mode = mode, velocity = velocity)
            derivative = (d_tmp - d_syn) / abs(step)
            gradient[:, i] = derivative

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
    def __init__(self, data, thick, periods, sigma = 0.003, log_prior = None, num_processes = 1, 
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

        phase, gradient = forward_sw(vs, self.periods, self.thick, relative_step = 0.005, 
                                            wave = self.wave, mode = self.mode, velocity = self.velocity, requires_grad = True)
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
    pass


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
    def __init__(self, data, thick, periods, sigma = 0.003, log_prior = None, num_processes = 1, 
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

    def solver(self, x):
        '''
        Calculate modelled data and data-model gradient by calling the external forward function (surf96)
        '''
        m, n = x.shape
        phase = np.zeros([m, self.data.shape[0]])
        gradient = np.zeros([m, self.data.shape[0], n])
        for i in range(m):
            vs = x.data.numpy()[i].squeeze()
            phase[i], gradient[i] = forward_sw(vs, self.periods, self.thick, relative_step = 0.005, 
                                                wave = self.wave, mode = self.mode, velocity = self.velocity, requires_grad = True)
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