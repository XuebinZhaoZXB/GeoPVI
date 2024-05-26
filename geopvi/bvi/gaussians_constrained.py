import torch
import math
import numpy as np

from numpy.random import choice

class Gaussian_constrained(object):
    def __init__(self, d, lower = 0., upper = 1., fullrank = False, init_dist = 'Normal'):
        # d is dimension of the random variable space
        self.d = d
        self.upper = torch.tensor(upper)
        self.lower = torch.tensor(lower)
        self.fullrank = fullrank
        self.init_dist = init_dist
        if self.init_dist == 'Normal':
            self.base = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
    
    def real_2_const(self, x):
        # z = self.lower + (self.upper - self.lower) * torch.sigmoid(x)
        z = self.lower + (self.upper - self.lower) / (1 + torch.exp(-x))
        log_det = (np.log(self.upper - self.lower) - x - 2 * torch.log(1 + torch.exp(-x))).sum(axis = -1)
        return z, log_det
    
    def const_2_real(self, x):
        z = torch.log(x - self.lower) - torch.log(self.upper - x)
        log_det = torch.log(1 / (x - self.lower) + 1 / (self.upper - x)).sum(axis = -1)
        return z, log_det

    def _atleast_2d(self, tensor):
        if len(tensor.shape) < 2:
            return tensor[None, :]
        return tensor

    def unflatten(self, params):
        params = self._atleast_2d(params)
        N = params.shape[0]
        # first d params are mu values
        mus = params[:, :self.d]
        # following params define the covariance matrix
        covariances = params[:, self.d:]
        return {"mus": mus, "covariances": covariances}

    def create_lower_triangular(self, values, dim, diagonal=0):
        if values.dim() == 1:
            lower = torch.zeros((dim, dim))
            lower[np.tril_indices(dim, diagonal)] = values
            return lower
        elif values.dim() == 2:
            lower = torch.zeros((dim, dim, values.shape[0]))
            lower[np.tril_indices(dim, diagonal)] = values.T
            return torch.permute(lower, (2,0,1))
        
    def log_normal(self, mu, matrix, X):
        b = torch.transpose(X - mu[:,None,:], dim0 = 1, dim1 = 2)       # n_comp * ndim * n_sample
        res = torch.linalg.solve_triangular(matrix, b, upper=False)     # n_comp * ndim * n_sample
        epsilons = torch.permute(res, (2,0,1))                          # n_sample * n_comp * ndim
        
        log_pdf = -0.5 * self.d * torch.log(torch.tensor(2 * math.pi)) - \
                  torch.sum(torch.log(torch.diag(matrix)), axis=1) - \
                  0.5 * torch.sum((X[:, None, :] - mus) ** 2 / sigmas, axis=2)
        
        if log_pdf.shape[1] == 1:
            log_pdf = log_pdf[:, 0]
        return log_pdf
    
    def log_pdf(self, param, x):
        if self.init_dist == 'Uniform':
            epsilons, log_det1 = self.const_2_real(x)
        elif self.init_dist == 'Normal':
            epsilons = x
            log_det1 = -self.base.log_prob(x)
        mu = param[:self.d]
        std = torch.log(torch.exp(param[self.d:self.d*2]) + 1)
        if self.fullrank:
            non_diag = param[self.d*2:]
            L = torch.diag(std) + self.create_lower_triangular(non_diag, self.d, -1)
            z = mu + torch.matmul(L.double(), epsilons.T).T
        else:
            z = mu + epsilons * std
        log_det2 = torch.log(std).sum().repeat(x.shape[0])
        m, log_det3 = self.real_2_const(z)
        
        return m, -(log_det1 + log_det2 + log_det3)

    def log_pdf_prev(self, param, m):
        z, log_det3 = self.const_2_real(m)     # dim of log_det3: n_sample
        mu = param[:,:self.d]
        std = torch.log(torch.exp(param[:,self.d:self.d*2]) + 1)
        if self.fullrank:
            non_diag = param[:,self.d*2:]
            ## create diagonal matrix diag with shape: n_comp * ndim * ndim
            diag = std.unsqueeze(2).expand(*std.size(), std.size(1)) * torch.eye(std.size(1))
            L = diag + self.create_lower_triangular(non_diag, self.d, -1)       # n_comp * ndim * ndim
            b = torch.transpose(z - mu[:,None,:], dim0 = 1, dim1 = 2)       # n_comp * ndim * n_sample
            res = torch.linalg.solve_triangular(L, b, upper=False)                 # n_comp * ndim * n_sample
            epsilons = torch.permute(res, (2,0,1))                           # n_sample * n_comp * ndim
        else:
            epsilons = (z[:,None,:] - mu) / std     # dim of epsilons: n_sample * n_comp * ndim
        log_det2 = - torch.log(std).sum(axis = -1).repeat(m.shape[0], 1)  # dim of log_det2: n_sample * n_comp

        if self.init_dist == 'Uniform':
            x, log_det1 = self.real_2_const(epsilons)       # dim of log_det1: n_sample * n_comp
        elif self.init_dist == 'Normal':
            x = epsilons
            # log_det1 = 0
            # log_det1 = -0.5 * torch.sum(epsilons**2, axis = -1)       # dim of log_det1: n_sample * n_comp
            log_det1 = self.base.log_prob(x)
            # this log_det1 is acutally the log-probability value of epsilons of standard normal distribution
        return x, log_det1 + log_det2 + log_det3[:,None]
        # return dim: n_sample * n_comp

    def generate_samples_for_one_component(self, param, num_samples):
        if self.init_dist == 'Uniform':
            x = torch.as_tensor(np.random.uniform(self.lower, self.upper, size = (num_samples, self.d)))
        elif self.init_dist == 'Normal':
            x = torch.as_tensor(np.random.normal(0., 1., size = (num_samples, self.d)))
            # x = torch.randn(num_samples, self.d)
        return x

    def params_init(self, params, weights, inflation):
        # initialization with heuristics
        params = self._atleast_2d(params)
        i = params.shape[0]
        if i == 0:
            # mu = torch.normal(torch.zeros(self.d), inflation * torch.ones(self.d))
            mu = torch.zeros(self.d)
            log_sigma = torch.zeros(self.d)
            new_param = torch.cat((mu, log_sigma), dim=0)
            if self.fullrank:
                non_diag = torch.zeros(int(self.d*(self.d - 1)/2))
                new_param = torch.cat((mu, log_sigma, non_diag), dim=0)
        else:
            mus = params[:, :self.d]
            probs = ((weights) / (weights).sum()).detach().numpy()
            k = choice(range(i), p=probs)
            log_sigmas = params[:, self.d:]
            # mu = mus[k,:] + torch.randn(self.d) * torch.sqrt(torch.tensor(inflation, dtype=torch.float32))\
            #                                         * torch.exp(log_sigmas[k, :])
            mu = mus[k,:] + torch.randn(self.d) * inflation
            # log_sigma = log_sigmas[k, :] + torch.randn(self.d) * inflation
            # log_sigma = torch.ones(self.d)
            log_sigma = torch.zeros(self.d)
            if self.fullrank:
                non_diag = torch.zeros(int(self.d*(self.d - 1)/2))
                new_param = torch.cat((mu, log_sigma, non_diag), dim=0)
            else:
                new_param = torch.cat((mu, log_sigma), dim=0)
        return new_param