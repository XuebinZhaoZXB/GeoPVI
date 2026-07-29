import torch
import numpy as np


def real_2_const(x, lower = None, upper = None):
    if lower is not None:
        lower = torch.as_tensor(lower, dtype=x.dtype, device=x.device)
    if upper is not None:
        upper = torch.as_tensor(upper, dtype=x.dtype, device=x.device)

    if lower is None and upper is None:
        z = x
        log_det = 0
    elif lower is not None and upper is None:
        z = lower + torch.exp(x)
        log_det = x.sum(axis = -1)
    elif lower is None and upper is not None:
        z = upper - torch.exp(x)
        log_det = x.sum(axis = -1)
    else:    
        z = lower + (upper - lower) / (1 + torch.exp(-x))
        log_det = (torch.log(upper - lower) - x - 2 * torch.log(1 + torch.exp(-x))).sum(axis = -1)
    return z, log_det
    
def const_2_real(x, lower = None, upper = None):
    if lower is not None:
        lower = torch.as_tensor(lower, dtype=x.dtype, device=x.device)
    if upper is not None:
        upper = torch.as_tensor(upper, dtype=x.dtype, device=x.device)

    if lower is None and upper is None:
        z = x
        log_det = 0
    elif lower is not None and upper is None:
        z = torch.log(x - lower)
        log_det = torch.log(1 / (x - lower)).sum(axis = -1)
    elif lower is None and upper is not None:
        z = torch.log(upper - x)
        log_det = torch.log(1 / (upper - x)).sum(axis = -1)
    else:
        z = torch.log(x - lower) - torch.log(upper - x)
        log_det = torch.log(1 / (x - lower) + 1 / (upper - x)).sum(axis = -1)
    return z, log_det


class GaussianComponent():
    '''
    Define a Gaussian component distribution for Stein Mixture Variational Inference (SMVI)
    '''
    def __init__(self, dim, kernel = 'diagonal', mask = None, base = 'Normal', std_type = 1,
                        constrained = False, lower = None, upper = None):
        '''
        Input:
            dim: dimension of the model parameter space
            kernel: type of covariance matrix for Gaussian component, including 'fullrank', 'structured' and 'diagonal'
                    'fullrank': full rank covariance matrix, with no structure constraint
                    'structured': structured covariance matrix, with non-diagonal entries only between parameters that are not fixed (mask = 0)
                    'diagonal': diagonal covariance matrix
            mask: an auxiliary mask array used for structured covariance matrix
            base: base distribution for reparameterisation, either standard normal N(0,1) or Uniform U(0,1)
            constrained: whether to apply constraints on the model parameters by transforming the unconstrained real space to a constrained space defined by lower and upper bounds
            lower: lower bound for constrained parameters, can be a scalar or a vector with shape (dim,)
            upper: upper bound for constrained parameters, can be a scalar or a vector with shape (dim,)
        '''
        self.dim = dim
        self.kernel = kernel
        self.std_type = std_type
        if base == 'Uniform' or base == 'Normal':
            self.base = base
        else:
            raise NotImplementedError("Base distribution provided not currently supported")

        self.constrained = constrained
        if constrained:
            if lower is None and upper is None:
                raise ValueError('Constrained Gaussian component, at least one of the lower and upper bounds should not be None!')
            if upper is None:
                self.upper = None
            else:
                if np.isscalar(upper):
                    upper = np.full((dim,), upper)
                self.upper = torch.from_numpy(upper)
            if lower is None:
                self.lower = None
            else:
                if np.isscalar(lower):
                    lower = np.full((dim,), lower)
                self.lower = torch.from_numpy(lower)
        if kernel == 'structured':
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
        '''
        Sample from base distribution, either standard normal N(0,1) or Uniform U(0,1)
        '''
        if self.base == 'Normal':
            return torch.randn(nsamples, self.dim)
        elif self.base == 'Uniform':
            return torch.rand(nsamples, self.dim)

    def log_prob2(self, param, x, chunks = None):
        '''
        Calculate the log probability value for samples x under the Gaussian component with parameter param
        The difference between this function and log_prob is that this function calculates the log probability value 
        within the real space, rather than constrained space
        Input:
            param: parameters for a Gaussian component
            x: samples drawn from base distribution, with shape (ncomponents, nsamples, dim)
            chunks: chunk size used to calculate mixture density for all samples, 
                    if None, then calculate all samples together; otherwise, calculate in chunks to save memory
        Output:
            z: transformed samples given each component's parameter, with shape (ncomponents, nsamples, dim)
            log_prob: log probability values of mixture density for sample z, with shape (ncomponents, nsamples)
        '''
        n_components = param.shape[0]
        # First, transfrom x based on component parameters to get samples distributed according to each component distribution
        if self.base == 'Uniform':
            x, _ = const_2_real(x, lower = 0, upper = 1)

        mus = param[:, :self.dim].unsqueeze(1)    # ncomponents * 1 * dim
        if self.std_type == 1:
            stds = torch.log(torch.exp(param[:, self.dim:2*self.dim]) + 1).unsqueeze(1)    # ncomponents * 1 * dim
        elif self.std_type == 2:
            stds = torch.exp(param[:, self.dim:2*self.dim]).unsqueeze(1)    # ncomponents * 1 * dim

        if self.kernel == 'fullrank':
            pass
        elif self.kernel == 'structured':
            non_diag = param[:, 2 * self.dim:].unsqueeze(1)    # ncomponents * 1 * num_non_diag
        elif self.kernel == 'diagonal':
            x = mus + x * stds  # ncomponents * nsamples * dim
        
        # Second, calculate log probability values for each sample under different component distributions
        # such that log probability values for the mixture distribution can be calculated
        if self.kernel == 'fullrank':
            pass
        elif self.kernel == 'structured':
            pass
        elif self.kernel == 'diagonal':
            '''TODO: chunk the number of samples to save memory'''
            theta = (x - mus.unsqueeze(1)) / stds.unsqueeze(1)   # ncomponents * ncomponents * nsamples * dim
        lg2 = - torch.log(stds.unsqueeze(1)).sum(axis = -1)     # ncomponents * 1 * 1

        if self.base == 'Uniform':
            _, lg3 = real_2_const(theta, lower = 0, upper = 1)   # ncomponents * ncomponents * nsamples
            log_base = 0.
        elif self.base == 'Normal':
            lg3 = torch.zeros_like(lg2)
            log_base = - 0.5 * self.dim * np.log(2*np.pi) - 0.5 * (theta**2).sum(axis = -1)   # ncomponents * ncomponents * nsamples

        # If the component distribution is defined on a constrained space, transform the samples back to the constrained space
        lg1 = torch.zeros([n_components, x.shape[1]])   # ncomponents * nsamples
        if self.constrained:
            x, lg1 = real_2_const(x, lower = self.lower, upper = self.upper)    
            # x: ncomponents * nsamples * dim, lg1: ncomponents * nsamples
            # lg1 = lg1.unsqueeze(0)    # 1 * ncomponents * nsamples
        
        log_gt = log_base + (lg2 + lg3)   # ncomponents * ncomponents * nsamples
        log_q = torch.logsumexp(log_gt, dim = 0) - np.log(n_components)   # ncomponents * nsamples

        return x, log_q, lg1

    def log_prob(self, param, x, chunks = None):
        '''
        Calculate the log probability value for samples x under the Gaussian component with parameter param
        Input:
            param: parameters for a Gaussian component
            x: samples drawn from base distribution 
            chunks: chunk size used to calculate mixture density for all samples, 
                    if None, then calculate all samples together; otherwise, calculate in chunks to save memory
        Output:
            z: transformed samples given each component's parameter, with shape (ncomponents, nsamples, dim)
            log_prob: log probability values of mixture density for sample z, with shape (ncomponents, nsamples)
        '''
        n_components = param.shape[0]
        # First, transfrom x based on component parameters to get samples distributed according to each component distribution
        if self.base == 'Uniform':
            x, _ = const_2_real(x, lower = 0, upper = 1)

        mus = param[:, :self.dim].unsqueeze(1)    # ncomponents * 1 * dim
        if self.std_type == 1:
            stds = torch.log(torch.exp(param[:, self.dim:2*self.dim]) + 1).unsqueeze(1)    # ncomponents * 1 * dim
        elif self.std_type == 2:
            stds = torch.exp(param[:, self.dim:2*self.dim]).unsqueeze(1)    # ncomponents * 1 * dim

        if self.kernel == 'fullrank':
            pass
        elif self.kernel == 'structured':
            non_diag = param[:, 2 * self.dim:].unsqueeze(1)    # ncomponents * 1 * num_non_diag
        elif self.kernel == 'diagonal':
            x = mus + x * stds  # ncomponents * nsamples * dim
        
        # Second, calculate log probability values for each sample under different component distributions
        # such that log probability values for the mixture distribution can be calculated
        if self.kernel == 'fullrank':
            pass
        elif self.kernel == 'structured':
            pass
        elif self.kernel == 'diagonal':
            '''TODO: chunk the number of samples to save memory'''
            theta = (x - mus.unsqueeze(1)) / stds.unsqueeze(1)   # ncomponents * ncomponents * nsamples * dim
        lg2 = - torch.log(stds.unsqueeze(1)).sum(axis = -1)     # ncomponents * 1 * 1

        if self.base == 'Uniform':
            _, lg3 = real_2_const(theta, lower = 0, upper = 1)   # ncomponents * ncomponents * nsamples
            log_base = 0.
        elif self.base == 'Normal':
            lg3 = torch.zeros_like(lg2)
            log_base = - 0.5 * self.dim * np.log(2*np.pi) - 0.5 * (theta**2).sum(axis = -1)   # ncomponents * ncomponents * nsamples

        # If the component distribution is defined on a constrained space, transform the samples back to the constrained space
        if self.constrained:
            x, lg1 = real_2_const(x, lower = self.lower, upper = self.upper)
            # x: ncomponents * nsamples * dim, lg1: ncomponents * nsamples
            lg1 = - lg1.unsqueeze(0)    # 1 * ncomponents * nsamples
        else:
            lg1 = torch.zeros(x.shape[0], x.shape[1])  # (ncomponents, nsamples)
        
        log_gt = log_base + (lg1 + lg2 + lg3)   # ncomponents * ncomponents * nsamples
        log_q = torch.logsumexp(log_gt - torch.log(torch.tensor(n_components)), dim = 0)   # ncomponents * nsamples

        return x, log_q

