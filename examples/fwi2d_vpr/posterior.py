import numpy as np
import time
import torch
from torch.autograd import Function
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


class LogNormal(Function):
    @staticmethod
    def forward(ctx, input, mu, L):
        '''
        Estimate log-probability of normal distribution given a lower triangular matrix L
        input: input tensor (nsampls * ndim)
        mu: mean of Gaussian distribution
        L: a lower triangular matrix with cov = L@LT
        grad: gradient of logp w.r.t. input (nsamples * ndim)
        '''
        diag = L.diagonal()
        tmp = input.detach().numpy() - mu
        epsilons = linalg.spsolve_triangular(L.tocsr(), tmp.T, lower = True)
        logp = - np.log(diag).sum().repeat(input.shape[0]) - 0.5 * (epsilons**2).sum(axis = 0)
        grad = linalg.spsolve_triangular(L.T.tocsr(), -epsilons, lower = False)
        ctx.save_for_backward(input, torch.tensor(grad.T))
        return torch.tensor(logp)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        this function returns the gradient w.r.t the input tensor in the forward function
        therefore, the return shape should be the same as the shape of input tensor
        '''
        input, grad = ctx.saved_tensors
        grad_input = (grad_output[...,None] * grad)
        return grad_input, None, None


class Posterior():
    def __init__(self, param, lower, upper, log_prior, covariance = 'diagonal', cov_inverse = True, inv = None):
        self.dim = lower.size
        self.upper = torch.from_numpy(upper)
        self.lower = torch.from_numpy(lower)
        self.mus = torch.from_numpy(param[:self.dim])
        self.sigmas = torch.from_numpy(param[self.dim:self.dim*2])
        self.log_prior = log_prior
        self.covariance = covariance
        self.cov_inverse = cov_inverse
        if param.size == 2 * self.dim:
            # means that param only contains mean and std, therefore defines a mean field variational pdf
            self.covariance = 'diagonal'
        if self.covariance == 'structured':
            diags = param[self.dim:].reshape(-1, self.dim)
            offset = (diags == 0).sum(axis = 1)
            self.L = sparse.diags(diags, -offset, shape = (self.dim, self.dim))
            if cov_inverse:
                if inv is None:
                    self.L_inverse = torch.from_numpy(linalg.spsolve_triangular(self.L.tocsr(), 
                                                        np.eye(self.dim), lower = True))
                else:
                    self.L_inverse = torch.from_numpy(inv)

    def const_2_real(self, x):
        z = torch.log(x - self.lower) - torch.log(self.upper - x)
        log_det = torch.log(1. / (x - self.lower) + 1. / (self.upper - x)).sum(axis = -1)
        return z, log_det

    def normalise(self, x):
        z = (x - self.mus) / self.sigmas
        log_det = - torch.log(self.sigmas).sum().repeat(x.shape[0])
        return z, log_det

    def log_prob(self, x):
        log_prior = self.log_prior(x)
        # return log_prior
        x, log_det1 = self.const_2_real(x)
        if self.covariance == 'diagonal':
            x, log_det2 = self.normalise(x)
            log_post = -0.5 * (x**2).sum(axis = 1) + log_det2 + log_det1
        elif self.covariance == 'structured':
            if self.cov_inverse:
                epsilons = self.L_inverse @ (x - self.mus).T
                log_post = - torch.log(self.sigmas).sum().repeat(x.shape[0]) \
                            - 0.5 * (epsilons**2).sum(axis = 0) + log_det1
            else:
                log_post = LogNormal.apply(x, self.mus.numpy(), self.L) + log_det1
        return log_post + log_prior
