import torch
import numpy as np


def real_2_const(x, lower = None, upper = None):
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
        log_det = (np.log(upper - lower) - x - 2 * torch.log(1 + torch.exp(-x))).sum(axis = -1)
    return z, log_det
    
def const_2_real(x, lower = None, upper = None):
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
    Define a Gaussian component distribution for boosting variational inference (BVI)
    '''
    def __init__(self, dim, kernel = 'diagonal', mask = None, base = 'Normal', perturb = 1,
                        constrained = False, lower = None, upper = None):
        self.dim = dim
        self.perturb = perturb
        self.kernel = kernel
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
        '''
        Sample from base distribution, either standard normal N(0,1) or Uniform U(0,1)
        '''
        if self.base == 'Normal':
            return torch.randn(nsamples, self.dim)
        elif self.base == 'Uniform':
            return torch.rand(nsamples, self.dim)

    def log_prob_gt(self, param, x):
        '''
        Calculate the log probability value g_t for a given component parameter: param and samples x
        Input:
            param: parameters for a Gaussian component
            x: samples drawn from base distribution 
        Output:
            x: samples distributed according to the current Gaussian component with param
            log_prob: log probability values for sample x
        '''
        if self.base == 'Uniform':
            x, lg1 = const_2_real(x, lower = 0, upper = 1)
            log_base = 0
        elif self.base == 'Normal':
            lg1 = torch.zeros(x.shape[0])
            log_base = -0.5 * self.dim * np.log(2*np.pi) - 0.5 * (x**2).sum(axis = -1)

        mu = param[:self.dim]
        # std = torch.exp(param[self.dim: 2*self.dim])
        std = torch.log(torch.exp(param[self.dim: 2*self.dim]) + 1)
        non_diag = param[2 * self.dim : ]
        if self.kernel == 'fullrank':
            L = torch.diag(std) + self._create_lower_triangular(non_diag, self.dim, -1)
            x = mu + torch.matmul(L, x.T).T
        elif self.kernel == 'structured':
            # TODO: use torch.sparse to represent L
            # Current torch version used does not support torch.sparse.spdiags
            non_diag = non_diag.reshape(-1, self.dim)
            tmp = x[:,None,:] * non_diag
            tmp1 = torch.zeros(tmp.shape)
            tmp1[:, np.sort(self.mask, axis = 1)] = tmp[:, self.mask]
            x = mu + std * x + tmp1.sum(axis = 1)
        elif self.kernel == 'diagonal':
            x = mu + x * std
        lg2 = torch.log(std).sum().repeat(x.shape[0])

        lg3 = 0.
        if self.constrained:
            x, lg3 = real_2_const(x, lower = self.lower, upper = self.upper)
        
        return x, log_base - (lg1 + lg2 + lg3)

    def log_prob_qt_minus_one(self, params, x):
        '''
        Calculate log probability value q_t-1 using components parameter params from previous t-1 components
        and get the corresponding sample in base distribution (transform back to base distribution)
        Essentially, this should be the inverse of the above log_prob_gt function
        Input:
            x:      samples in model parameter space with shape: (nsamples, dim)
            params: parameters for obtained Gaussian components: g1, g2, ..., gt-1
        Output:
            x:      samples in base distribution with shape: (nsamples * ncomponents * dim)
            lg:     log probability values for the input tensor evaluated under different components g1, ... gt-1
        '''
        x = x[:, None, :]           # first add one dimension for different components
        lg1 = torch.zeros(x.shape[:2])
        if self.constrained:
            x, lg1 = const_2_real(x, lower = self.lower, upper = self.upper)    # lg1: nsamples * 1

        mu = params[:, :self.dim]
        # std = torch.exp(params[:, self.dim:2*self.dim])
        std = torch.log(torch.exp(params[:, self.dim:2*self.dim]) + 1)
        if self.kernel == 'fullrank':
        # TODO kernel == 'fullrank' and kernel == 'structured'
            pass
        elif self.kernel == 'structured':
            pass
        elif self.kernel == 'diagonal':
            x = (x - mu) / std                                          # nsamples * ncomponents * ndim
        lg2 = - torch.log(std).sum(axis = -1).repeat(x.shape[0], 1)     # nsamples * ncomponents

        if self.base == 'Uniform':
            x, lg3 = real_2_const(x, lower = 0, upper = 1)              # nsamples * ncomponents * ndim
                                                                        # lg3: nsamples * ncomponents
            log_base = 0
        elif self.base == 'Normal':
            lg3 = torch.zeros_like(lg2)
            log_base = - 0.5 * self.dim * np.log(2*np.pi) - 0.5 * (x**2).sum(axis = -1)
        
        return x, log_base + (lg1 + lg2 + lg3)

    def component_init(self, params, weights):
        '''
        Initialize Gaussian component with heuristics
        The mean vector is initialised by randomly perturbing the mean vector of one of previously obtained components
        The first component is initialised as N(0, 1)
        The std vector is initialised as 0
        Input:
            params: parameters for each component
            weights: weights for each component
        Output:
            new_param: initial parameter values for the new component
        '''
        params = self._atleast_2d(params)
        i = params.shape[0]
        if i == 0:
            # Initialise the first component: ADVI initilisation
            mu = torch.zeros(self.dim)
            # std = log(exp(log_sigma) + 1)
            log_sigma = torch.zeros(self.dim)
            if self.kernel == 'fullrank':
                non_diag = torch.zeros(int(self.dim*(self.dim - 1)/2))
                new_param = torch.cat((mu, log_sigma, non_diag), dim=0)
            elif self.kernel == 'structured':
                non_diag = torch.zeros(self.mask.shape).reshape(-1)
                new_param = torch.cat((mu, log_sigma, non_diag), dim=0)
            elif self.kernel == 'diagonal':
                new_param = torch.cat((mu, log_sigma), dim=0)
        else:
            # Initialise new components by randomly perturbing one of the previously obtained components
            mus = params[:, :self.dim]
            probs = ((weights) / (weights).sum()).detach().numpy()
            k = np.random.choice(range(i), p=probs)
            log_sigmas = params[:, self.dim : 2*self.dim]
            mu = mus[k,:] + torch.randn(self.dim) * self.perturb
            # mu = mus[k,:] + torch.randn(self.dim) * torch.log(1 + torch.exp(log_sigmas[k, :])) * self.perturb
            # mu = mus[k,:] + torch.randn(self.dim) * torch.exp(log_sigmas[k, :]) * self.perturb
            # log_sigma = log_sigmas[k, :] + torch.randn(self.dim) * inflation
            # log_sigma = torch.ones(self.dim)
            log_sigma = torch.zeros(self.dim)

            if self.kernel == 'fullrank':
                non_diag = torch.zeros(int(self.dim*(self.dim - 1)/2))
                new_param = torch.cat((mu, log_sigma, non_diag), dim=0)
            elif self.kernel == 'structured':
                non_diag = torch.zeros(self.mask.shape).reshape(-1)
                new_param = torch.cat((mu, log_sigma, non_diag), dim=0)
            elif self.kernel == 'diagonal':
                new_param = torch.cat((mu, log_sigma), dim=0)
        return new_param

    def unflatten(self, params):
        params = self._atleast_2d(params)
        # first d params are mu values
        mus = params[:, :self.dim]
        # following params define the covariance matrix
        # covariances = params[:, self.dim:]
        # covariances[:, :self.dim] = torch.exp(params[:, self.dim:2*self.dim])
        # stds = torch.exp(params[:, self.dim:2*self.dim])
        stds = torch.log(torch.exp(params[:, self.dim:2*self.dim]) + 1)
        non_diags = params[:, 2 * self.dim:]
        # covariance = torch.cat([torch.exp(params[:, self.dim:2*self.dim]), params[:, 2*self.dim:]], dim = 1)
        if self.kernel != 'diagonal':
            return {"mus": mus, "stds": stds, 'off_diags': non_diags}
        return {"mus": mus, "stds": stds}
