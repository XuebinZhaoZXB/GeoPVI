import numpy as np
import torch
import torch.nn as nn
from torch.multiprocessing import Pool
from torch.autograd import Function

from fmm.fmm import fm2d

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
    def __init__(self, data, src, rec, mask, sigma=0.05, nx=16, ny=16, 
                    xmin=-4, ymin=-4, dx=0.5, dy=0.5, gdx=2, gdy=2, sdx=4, sext=4, 
                    lower = np.NINF, upper = np.PINF, dim = 1, num_processes = 1):
        self.lower = lower
        self.upper = upper
        self.dim = dim
        self.num_processes = num_processes

        self.data = data
        self.sigma = torch.tensor(sigma)
        self.nx = nx
        self.ny = ny
        self.xmin = xmin
        self.ymin = ymin
        self.dx = dx
        self.dy = dy
        self.gdx = gdx
        self.gdy = gdy
        self.sdx = sdx
        self.sext = sext

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
        m = x.shape[0]
        time = np.zeros([m, self.data.shape[0]])
        dtdv = np.zeros([m, self.data.shape[0], self.dim])
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
        m = x.shape[0]
        time = np.zeros([m, self.data.shape[0]])
        dtdv = np.zeros([m, self.data.shape[0], self.dim])

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
        return log_like