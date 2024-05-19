import numpy as np
import torch
from torch.multiprocessing import Pool
from torch.autograd import Function

from geopvi.tomo2d.fmm import fm2d

class ForwardModel(Function):
    @staticmethod
    def forward(ctx, input, func):
        output, dtdv = func(input)
        ctx.save_for_backward(input, torch.tensor(dtdv))
        return torch.tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        input, dtdv = ctx.saved_tensors
        grad_input = (grad_output[...,None] * dtdv).sum(axis = -2)
        return grad_input, None


class Posterior():
    def __init__(self, data, config, src, rec, mask = None, sigma=0.05, lower = np.NINF, upper = np.PINF, 
                        num_processes = 1, log_prior = None):
        self.lower = lower
        self.upper = upper
        self.num_processes = num_processes
        self.log_prior = log_prior

        self.data = data
        self.sigma = torch.tensor(sigma)

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

        self.mask = np.ascontiguousarray(mask)
        self.src = src
        self.rec = rec
        self.srcx = np.ascontiguousarray(src[:,0])
        self.srcy = np.ascontiguousarray(src[:,1])
        self.recx = np.ascontiguousarray(rec[:,0])
        self.recy = np.ascontiguousarray(rec[:,1])

    def real_2_const(self, x):
        if (np.isneginf(self.lower) and np.isposinf(self.upper)):
            z = x
            log_det = 0
        elif (not np.isneginf(self.lower) and np.isposinf(self.upper)):
            z = self.lower + torch.exp(x)
            log_det = x.sum(1)
        elif (np.isneginf(self.lower) and not np.isposinf(self.upper)):
            z = self.upper - torch.exp(x)
            log_det = x.sum(1)
        else:
            z = self.lower + (self.upper - self.lower) / (1 + torch.exp(-x))
            log_det = (np.log(self.upper - self.lower) - x - 
                        2 * torch.log(1 + torch.exp(-x))).sum(1)
        return z, log_det

    def solver(self, vel):
        """
        warpper function for multi-processing.pool
        """
        time, dtdv = fm2d(vel, self.srcx, self.srcy, self.recx, self.recy, self.mask,
                        self.nx, self.ny, self.xmin, self.ymin, self.dx, self.dy, 
                        self.gdx, self.gdy, self.sdx, self.sext)
        return time, dtdv

    def fmm(self, x):
        '''
        Calculate modelled data and data-model gradient by solving forward function
        '''
        m, n = x.shape
        time = np.zeros([m, self.data.shape[0]])
        dtdv = np.zeros([m, self.data.shape[0], n])
        for i in range(m):
            vel = x.data.numpy()[i].squeeze().astype(np.float64)
            time[i], dtdv[i] = fm2d(vel, self.srcx, self.srcy, self.recx, self.recy, self.mask,
                            self.nx, self.ny, self.xmin, self.ymin, self.dx, self.dy, 
                            self.gdx, self.gdy, self.sdx, self.sext)
        return time, dtdv

    def fmm_parallel(self, x):
        '''
        Parallelised version of self.fmm using torch.multiprocessing
        '''
        m, n = x.shape
        time = np.zeros([m, self.data.shape[0]])
        dtdv = np.zeros([m, self.data.shape[0], n])

        pool = Pool(processes = self.num_processes)
        results = pool.map(self.solver, [x.detach().numpy()[i].squeeze().astype(np.float64) for i in range(m)])
        pool.close()
        pool.join()
        for i in range(m):
            time[i] = results[i][0]
            dtdv[i] = results[i][1]
        return time, dtdv

    def log_prob(self, x):
        """
        calculate log likelihood and its gradient directly from model x
        This version considers the transform from real to constrained space
        thus returns log_loke + log_det
        """
        # model, log_det = self.real_2_const(x)
        if self.num_processes == 1:
            d_syn = ForwardModel.apply(x, self.fmm)
        else:
            d_syn = ForwardModel.apply(x, self.fmm_parallel)
        log_like = - 0.5 * torch.sum(((torch.from_numpy(self.data) - d_syn)/self.sigma) ** 2, axis = -1)
        logp = log_like + self.log_prior(x)
        return logp