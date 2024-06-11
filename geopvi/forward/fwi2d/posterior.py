import numpy as np
import torch
from torch.autograd import Function
from torch.multiprocessing import Pool

from geopvi.forward.fwi2d import aco2d

class ForwardModel(Function):
    @staticmethod
    def forward(ctx, input, func):
        output, grad = func(input)
        ctx.save_for_backward(input, torch.tensor(grad))
        return torch.tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        this function returns the gradient w.r.t the input tensor in the forward function
        therefore, the return shape should be the same as the shape of input tensor
        '''
        input, grad = ctx.saved_tensors
        # grad_input = (grad_output[...,None] * grad).sum(axis = -2)
        grad_input = (grad_output[...,None] * grad)
        return grad_input, None


def prepare_fwi_parameters(paramfile, config):
    with open(paramfile,'w') as f:
        f.write('--grid points in x direction (nx)\n')
        f.write(config.get('FWI','nx')+'\n')
        f.write('--grid points in z direction (nz)\n')
        f.write(config.get('FWI','nz')+'\n')
        f.write('--pml points (pml0)\n')
        f.write(config.get('FWI','pml')+'\n')
        f.write('--Finite difference order (Lc)\n')
        f.write(config.get('FWI','Lc')+'\n')
        f.write('--Method to calculate Laplace operator: 0 for FDM and 1 for PSM, (laplace_slover)\n')
        f.write(config.get('FWI','laplace_slover')+'\n')
        f.write('--Total number of sources (ns)\n')
        f.write(config.get('FWI','ns')+'\n')
        f.write('--Total time steps (nt)\n')
        f.write(config.get('FWI','nt')+'\n')
        f.write('--Shot interval in grid points (ds)\n')
        f.write(config.get('FWI','ds')+'\n')
        f.write('--Grid number of the first shot to the left of the model (ns0)\n')
        f.write(config.get('FWI','ns0')+'\n')
        f.write('--Depth of source in grid points (depths)\n')
        f.write(config.get('FWI','depths')+'\n')
        f.write('--Depth of receiver in grid points (depthr)\n')
        f.write(config.get('FWI','depthr')+'\n')
        f.write('--Receiver interval in grid points (dr)\n')
        f.write(config.get('FWI','dr')+'\n')
        f.write('--Time step interval of saved wavefield during forward (nt_interval)\n')
        f.write(config.get('FWI','nt_interval')+'\n')
        f.write('--Grid spacing in x direction (dx)\n')
        f.write(config.get('FWI','dx')+'\n')
        f.write('--Grid spacing in z direction (dz)\n')
        f.write(config.get('FWI','dz')+'\n')
        f.write('--Time step (dt)\n')
        f.write(config.get('FWI','dt')+'\n')
        f.write('--Donimate frequency (f0)\n')
        f.write(config.get('FWI','f0')+'\n')

    return


class Posterior():
    def __init__(self, data, config, vel_fixed = 1950, sigma = 0.1, num_processes = 1, 
                            log_prior = None, mask = None, paramfile = 'input_params.txt'):
        '''
        data: observed data with shape: (ns*nt*nr,)
        config: configure file used to define nuisance parameters for 2D fwi code
        vel_fixed: velocity value for fixed (normally water layer) regions
        log_prior: a function that takes samples as input and calculates their log-prior values (using PyTorch)
                    Return: y = log_prior(x)
        mask: a mask array where the parameters with mask = 0 will be fixed 
        sigma: data error/uncertainty
        num_processes: number of process for parallelisation
        paramfile: filename (and full path) that defines parameters for 2D FWI
        '''
        self.data = data
        self.mask = mask
        self.log_prior = log_prior
        self.num_processes = num_processes
        self.sigma = sigma
        self.vel_water = vel_fixed
        self.paramfile = paramfile
        # create mask matrix for model parameters that are fixed during inversion (e.g., water layer)
        if mask is None:
            nx = config.getint('FWI','nx')
            nz = config.getint('FWI','nz')
            mask = np.full((nx*nz),True)
        self.mask = mask

        prepare_fwi_parameters(self.paramfile, config)

    def solver(self, vel):
        """
        warpper function for multi-processing.pool
        data_syn: synthetic waveform data with shape of (ns*nt*nr,) 1 dim array
        grad: gradient of l2_loss w.r.t. input velocity model with shape of (nz, nx)
        """
        data_syn, grad = aco2d.fwi(vel, self.data, paramfile = self.paramfile)
        grad = grad[self.mask]
        loss = 0.5 * np.sum((data_syn - self.data)**2)
        return loss, -grad

    def fwi(self, x):
        '''
        Calculate modelled waveform data and fwi data-model gradient using finite difference
        '''
        m, _ = x.shape
        loss = np.zeros(m)
        grad = np.zeros(x.shape)
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy().astype(np.float64)
        # vel = np.full(self.mask.shape, self.vel_water, dtype = np.float64)
        vel = np.full([m, self.mask.size], self.vel_water, dtype = np.float64)
        vel[:,self.mask] = x
        for i in range(m):
            loss[i], grad[i] = self.solver(vel[i])
            # data_syn, gradient = aco2d.fwi(vel[i], self.data, paramfile = self.paramfile)
            # grad[i] = gradient[self.mask]
            # loss[i] = 0.5 * np.sum((data_syn - self.data)**2)
        return loss, grad

    def fwi_parallel(self, x):
        '''
        Parallelised version of self.fwi using torch.multiprocessing
        '''
        m, _ = x.shape
        loss = np.zeros(m)
        grad = np.zeros(x.shape)
        if not isinstance(x, np.ndarray):
            x = x.detach().numpy().astype(np.float64)
        vel = np.full([m, self.mask.size], self.vel_water, dtype = np.float64)
        vel[:,self.mask] = x

        with Pool(processes = self.num_processes) as pool:
            results = pool.map(self.solver, [vel[i] for i in range(m)])

        for i in range(m):
            loss[i] = results[i][0]
            grad[i] = results[i][1]
        return loss, grad

    def log_prob(self, x):
        """
        calculate log likelihood and its gradient directly from model x
        Input
            x: 2D array with dimension of nsamples * ndim
        Return
            logp: a vector of (unnormalised) log-posterior value for each sample
        """    
        if(torch.isnan(x).any()):
            raise ValueError('NaN occured in sample!')
        if self.num_processes == 1:
            loss = ForwardModel.apply(x, self.fwi)
        else:
            loss = ForwardModel.apply(x, self.fwi_parallel)
        logp = -loss/self.sigma**2 + self.log_prior(x)
        return logp