import math
import numpy as np 
import torch


def real_2_const(x, lower = 0, upper = 1):
    z = lower + (upper - lower) / (1 + torch.exp(-x))
    log_det = (torch.log(upper - lower) - x - 2 * torch.log(1 + torch.exp(-x))).sum(axis = -1)
    return z, log_det
    
def const_2_real(x, lower = 0, upper = 1):
    z = torch.log(x - lower) - torch.log(upper - x)
    log_det = torch.log(1 / (x - lower) + 1 / (upper - x)).sum(axis = -1)
    return z, log_det


class GaussianComponent():
    '''
    Define a Gaussian component distribution for boosting variational inference (BVI)
    '''
    def __init__(self, dim, kernel = 'diagonal', mask = None, base = 'Normal', 
                        constrained = True, lower = None, upper = None):
        self.dim = dim
        self.kernel = kernel
        if base == 'Uniform' or base == 'Normal':
            self.base = base
        else:
            raise NotImplementedError("Base distribution provided not currently supported")

        self.constrained = constrained
        if constrained:
            if lower is None and upper is None:
                raise ValueError('Constrained component, at least one of the lower and upper bounds should not be None!')
            if np.isscalar(upper):
                upper = np.full((dim,) upper)
            self.upper = torch.from_numpy(upper)
            if np.isscalar(lower):
                lower = np.full((dim,) lower)
            self.lower = torch.from_numpy(lower)
        if kernel == 'strucutred':
            if mask is None:
                self.mask = np.ones([1, dim])
                self.mask[0, -1] = 0.
            else:
                self.mask = mask
            self.offset = (self.mask == 0).sum(axis = 1)

    def _atleast_2d(self, tensor):
        if len(tensor.shape) < 2:
            return tensor[None, :]
        return tensor

    def _create_lower_triangular(self, values, dim, diagonal=0):
        if values.dim() == 1:
            lower = torch.zeros((dim, dim))
            lower[np.tril_indices(dim, diagonal)] = values
            return lower
        elif values.dim() == 2:
            lower = torch.zeros((dim, dim, values.shape[0]))
            lower[np.tril_indices(dim, diagonal)] = values.T
            return torch.permute(lower, (2,0,1))

    def sample_from_base(self, nsamples):
        'Sample from base distribution, either standard normal N(0,1) or Uniform U(0,1)'
        if self.base == 'Normal':
            return torch.randn(nsamples, self.dim)
        elif self.base == 'Uniform':
            return torch.rand(nsamples, self.dim)

    def log_prob_gt(self, param, x):
        'Calculate the log probability value g_t for a given component parameter: param and samples x'
        if self.base == 'Uniform':
            x, log_det = const_2_real(x, lower = 0, upper = 1)
            log_base = 0
        elif self.base == 'Normal':
            log_det = torch.zeros(x.shape[0])
            log_base = -0.5 * self.dim * np.log(2*np.pi) - 0.5 * (x**2).sum(axis = 1)

        mu, std = param[:self.dim * 2].chunk(2)
        std = torch.log(torch.exp(std) + 1)
        if self.kernel == 'fullrank':
        # TODO kernel == 'fullrank' and kernel == 'structured'
            pass
        elif self.kernel == 'structured':
            pass
        elif self.kernel == 'diagonal':
            x = mu + x * std
        log_det += torch.log(std).sum().repeat(x.shape[0])

        lg = 0.
        if self.constrained:
            x, lg = real_2_const(x, lower = self.lower, upper = self.upper)
        
        return x, log_base - (log_det + lg)

    def log_prob_qt_minus_one(self, param, m):
        '''
        Calculate the log probability value q_t-1 using components parameter param from previous t-1 components
        and get the corresponding sample in base distribution
        '''


    def component_init(self, params, weights, perturb_std):
        'Initialization Gaussian component with heuristics'



class boostingGaussian():
    '''
    A class that defines a variational distribution through boosting variational inference
    which essentially builds a mixture of distributions (e.g., Gaussians) to approximate the true posterior pdf
    '''
    def __init__(self, flows, kernel = 'diagonal', mask = None, constrained = True):