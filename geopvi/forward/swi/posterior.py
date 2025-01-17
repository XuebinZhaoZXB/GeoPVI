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
        grad_input = (grad_output[...,None] * grad).sum(axis = -2)
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


class Posterior_1D():
    '''
    Class to calculate the log posterior and its gradient for 1D surface wave inversion
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
                                                   wave = self.wave, mode = self.mode, velocity = self.velocity)
        return phase, gradient
    
    def log_prob(self, x):
        """
        Calculate log posterior and its gradient directly from model x
        """
        d_syn = ForwardModel.apply(x, self.solver)
        log_like = - 0.5 * torch.sum(((torch.from_numpy(self.data) - d_syn)/self.sigma) ** 2, axis = -1)
        log_prior = self.log_prior(x)
        # print(log_prior)
        # # set a prior information: the top layer has minimum shear-velocity
        # # This ensures the computed phase velocities are phase velocities of Rayleigh or Love waves  
        # for i in range(x.shape[0]):
        #     if x[i, 1:].min() < x[i,0]:
        #         log_prior[i] -= 10.

        logp = log_like + log_prior
        return logp