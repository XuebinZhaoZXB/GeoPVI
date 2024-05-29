import numpy as np
import torch
from torch.autograd import Function
import scipy
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
        tmp = input.detach().numpy().T - mu[:,None]
        epsilons = linalg.spsolve_triangular(L.tocsr(), tmp, lower = True)
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


class Smoothing(Function):
    @staticmethod
    def forward(ctx, input, smooth_matrix):
        '''
        Apply smooth matrix on model input
        input: input tensor (nsampls * ndim)
        smooth_matrix: a smooth matrix that performs spatial smooth
        grad: gradient of logp w.r.t. input (nsamples * ndim)
        '''
        tmp = smooth_matrix * input.detach().numpy().T
        logp = -0.5 * (tmp**2).sum(axis = 0)
        grad = (-smooth_matrix.transpose() * tmp).T
        ctx.save_for_backward(input, torch.tensor(grad))
        return torch.tensor(logp)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        this function returns the gradient w.r.t the input tensor in the forward function
        therefore, the return shape should be the same as the shape of input tensor
        '''
        input, grad = ctx.saved_tensors
        grad_input = (grad_output[...,None] * grad)
        return grad_input, None


class Uniform():
    '''
    A class that defines Uniform prior distribution used for inversion
    '''
    def __init__(self, lower = np.array([0.]), upper = np.array([1.]), smooth_matrix = None):
        '''
        lower (numpy array): lower bound of the Uniform distribution 
        upper (numpy array): upper bound of the Uniform distribution
        smooth_matrix (scipy sparse): smooth matrix applied on model samples, used to define smooth prior pdf
        '''
        self.lower = torch.from_numpy(lower)
        self.upper = torch.from_numpy(upper)
        self.dim = lower.size
        self.smooth_matrix = smooth_matrix

    def log_prob(self, x):
        '''
        Compute log probability using PyTorch, such that the result can be back propagated
        '''
        logp = - torch.log(self.upper - self.lower).sum()

        if self.smooth_matrix is not None:
            logp_smooth = Smoothing.apply(x, self.smooth_matrix)
            return logp + logp_smooth
        return logp


class Normal():
    '''
    A class that defines Normal prior distribution used for inversion, 
    parametrised by a mean vector and a covariance matrix
    '''
    def __init__(self, loc = np.array([0.]), std = np.array([1.]), covariance = None, scale_tril = None,
                    precision = None, inverse = False, smooth_matrix = None):
        '''
        loc (numpy array): mean vector
        std (numpy array): standard deviation - ignore parameter correlations (diagonal Gaussian)
        covariance (numpy array): covariance matrix
        scale_tril (numpy array): L - scale lower triangular matrix (cov = L * L.T)
        precision (numpy array): precision matrix
        inverse (Bool): calculate inverse of covariance/scale_tril, so that log_prob can be evaluated easily
        smooth_matrix (scipy sparse): smooth matrix applied on model samples, used to define smooth prior pdf
        Note: only one of std, covariance or precision or scale_tril can be specified
        '''
        self.loc = torch.from_numpy(loc)
        self.dim = loc.size
        self.inverse = inverse
        # check which option is provided as the second parameter of the Normal distribution
        if (covariance is not None) + (scale_tril is not None) + (precision is not None) + (std is not None) != 1:
            raise ValueError("Exactly one of std, covariance or precision or scale_tril may be specified.")
        if std is not None:
            self.param2 = std
            self.param2_type = 'std'
        elif covariance is not None:
            # if covariance matrix is provided, covert covariance matrix to either scale_tril or precision
            # so that log-probability value can be calculated easily
            self.param2 = np.linalg.cholesky(covariance)
            self.param2_type = 'scale_tril'
        elif scale_tril is not None:
            self.param2 = scale_tril
            self.param2_type = 'scale_tril'
        else:
            self.param2 = precision
            self.param2_type = 'precision'
            # If precision matrix is provided, logdet_cov is not easy to calculate, thus simply ignored
            self.logdet_cov = None  
        if self.param2_type == 'scale_tril' and inverse is True:
            diag = np.diagonal(self.param2)
            self.logdet_cov = np.log(diag).sum() * 2
            self.param2 = scipy.linalg.solve_triangular(self.param2, np.eye(self.dim), lower = True)
            self.param2 = self.param2.T @ self.param2
            self.param2_type = 'precision'
        self.param2 = torch.from_numpy(self.param2)

        self.smooth_matrix = smooth_matrix

    def log_prob(self, x):
        '''
        Compute log probability using PyTorch, such that the result can be back propagated
        '''
        x_minus_loc = x- self.loc
        if self.param2_type == 'std':
            logp = - 0.5 * torch.sum((x_minus_loc / self.param2)**2, axis = 1) \
                    - torch.log(self.param2).sum() - 0.5 * self.dim * np.log(2*np.pi)
        elif self.param2_type == 'precision':
            logp = - 0.5 * (torch.tensordot(x_minus_loc, self.param2, dims = 1) * x_minus_loc).sum(axis = -1) \
                    - 0.5 * self.logdet_cov - 0.5 * self.dim * np.log(2*np.pi) 
        elif self.param2_type == 'scale_tril':
            # For newer Pytorch version, torch.triangular_solve is deprecated 
            # thus use torch.linalg.solve_triangular()
            # epsilons = torch.linalg.solve_triangular(self.param2, x_minus_loc.T, upper = False).T
            epsilons = torch.triangular_solve(x_minus_loc.T, self.param2, upper = False)[0].T
            logp = - 0.5 * torch.sum(epsilons**2, axis = 1) \
                    - torch.log(self.param2.diagonal()).sum() - 0.5 * self.dim * np.log(2*np.pi)
            # logp = LogNormal.apply(x, self.loc.numpy(), self.param2)

        if self.smooth_matrix is not None:
            logp_smooth = Smoothing.apply(x, self.smooth_matrix)
            return logp + logp_smooth
        return logp

class User_defined_prior():
    '''
    A class that defines user-defined prior distribution used for inversion
    '''
    def __init__(self):
        pass

    def log_prob(self, x):
        '''
        Calculate log prior probability value for samples x
        The results should be back-propagated by PyTorch auto-grad engine.
        For example:
            logp = log_prob_of_prior_samples(x)
        '''
        pass
        return logp