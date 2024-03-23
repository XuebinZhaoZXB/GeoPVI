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


class Prior():
    '''
    A class that defines prior distribution used for inversion
    Currently, only support Uniform and diagonal Gaussian distributions
    '''
    def __init__(self, param1 = 0., param2 = 1., param3 = None, patch_size = 100, param2_type = 'std', 
                    smooth_matrix = None, prior_type = 'Uniform'):
        '''
        param1: lower bound (Uniform) or mean (Gaussian)
        param2: upper bound (Uniform) or covariance /precision matrix (Gaussian) 
        param2_type: type for param2, 
                     support: std (param2 is a 1D array defining standard deviation)
                              covariance (param2 is a 2D array defining full covariance matrix)
                              scale_tril (param2 is a 2D array defining lower triangular matrix L)
                              precision  (param2 is a 2D array defining precision (inverse of covariance) matrix)
        param1 & param2 should match the dimensionality of the problem
        prior_type: 'Uniform' or 'Normal'
        '''
        self.param1 = torch.as_tensor(param1)
        self.param2_type = param2_type
        self.dim = param1.size
        self.prior_type = prior_type
        self.smooth_matrix = smooth_matrix
        if param2_type == 'scale_tril':
            offset = (param2[1:] == 0).sum(axis = 1)
            # R_full = sparse.diags(param2[1:], -offset, shape = (self.dim, self.dim))
            # self.param2 = sparse.diags(3*param2[0], shape = (self.dim, self.dim)) @ R_full
            self.param2 = sparse.diags(param2[0], shape = (self.dim, self.dim))
        else:
            self.param2 = torch.as_tensor(param2)
        if param2_type == 'local_cor':
            self.param3 = torch.as_tensor(param3)
            self.patch_size = patch_size
            self.get_patch(self.patch_size)

    def get_patch(self, patch_size):
        blockz, blockx = 30, 60

        z_index = np.random.randint(0, 100 + blockz + 1, size = (patch_size))
        x_index = np.random.randint(0, 250 + blockx + 1, size = (patch_size))
        self.patch = np.zeros([patch_size, (100+blockz*2)*(250+blockx*2)], 
                                dtype = 'bool').reshape(patch_size, 100+blockz*2, 250+blockx*2)

        for i in range(patch_size):
            self.patch[i, z_index[i]:z_index[i]+blockz, x_index[i]:x_index[i]+blockx] = True
        self.patch = self.patch.reshape(patch_size, -1)

    def extend_model(self, x):
        blockz, blockx = 30, 60

        model = x.reshape(-1, 100, 250)
        z = torch.zeros((model.shape[0], self.patch.shape[-1])).reshape(-1, 100+blockz*2, 250+blockx*2)
        z[:, blockz:-blockz, blockx:-blockx] = model
        z[:,:blockz, blockx:-blockx] = model[:,0][:,None].detach()
        z[:,-blockz:, blockx:-blockx] = model[:,-1][:,None].detach()
        z[:,:,:blockx] = z[:,:,blockx][...,None].detach()
        z[:,:,-blockx:] = z[:,:,-blockx-1][...,None].detach()
        if x.dim() == 1:
            z = torch.squeeze(z.reshape([model.shape[0], -1]), dim = 0)
        elif x.dim() == 2:
            z = z.reshape([model.shape[0], -1])
        return z

    def log_prob(self, x):
        '''
        Compute log probability using PyTorch, such that the result can be back propagated
        '''
        if self.prior_type == 'Uniform':
            # if prior is a Uniform distribution, log_prior value is a constant 
            logp = - torch.log(self.param2 - self.param1).sum()
        elif self.prior_type == 'Normal':
            # the calculation of logp depends on param2_type
            # log_normal requires det(cov), currently only diagonal cov correctly calculate det(cov)
            # other param2_type options simply ignore the  calculation of det(cov) 
            # since this constant value won't acturally affect trainging
            # tmp = x - self.param1
            if self.param2_type == 'std':
                logp = -0.5 * torch.sum((tmp / self.param2)**2, axis = 1) \
                        - torch.log(self.param2).sum() - 0.5 * self.dim * np.log(2*np.pi)
            elif self.param2_type == 'precision':
                inv_cov = torch.diag(1 / self.param2**2)
                logp = - 0.5 * (tmp @ inv_cov @ tmp.T).diag() - 0.5 * self.dim * np.log(2*np.pi)

                # logp = -0.5 * (tmp @ self.param2 @ tmp.T).diag() - 0.5 * self.dim * np.log(2*np.pi)
                # logp = -0.5 * (torch.matmul(torch.matmul(tmp, self.param2)), tmp.T).diag() \
                #         - 0.5 * self.dim * np.log(2*np.pi)
            elif self.param2_type == 'scale_tril':
                logp = LogNormal.apply(x, self.param1.numpy(), self.param2)
            elif self.param2_type == 'local_cor':
                m = x.shape[0]
                # tmp = (x - self.param1) / self.param2
                tmp = (self.extend_model(x) - self.extend_model(self.param1)) / self.extend_model(self.param2)

                self.get_patch(self.patch_size)
                x_patch = torch.broadcast_to(
                                            tmp[:,None], (m, self.patch_size, self.patch.shape[-1])
                                        )[:,self.patch].reshape([m, self.patch_size, -1])
                logp_patch = - 0.5 * (torch.tensordot(x_patch, self.param3, dims = 1) * x_patch).sum(axis = -1)
                logp = logp_patch.mean(axis = -1) / self.param3.shape[0] * self.dim
                # logp = (logp_patch * self.scale).mean(axis = -1) / self.param3.shape[0]*self.dim

        if self.smooth_matrix is not None:
            logp_smooth = Smoothing.apply(x, self.smooth_matrix)
            return logp + logp_smooth
        return logp


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
        self.loc = torch.from_numpy(param1)
        self.dim = param1.size
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


class Gaussian():
    '''
    A class that defines Gaussian prior distribution used for inversion
    '''
    def __init__(self, param1 = 0., param2 = 1., param3 = None, patch_size = 100, param2_type = 'std', 
                    smooth_matrix = None, prior_type = 'Uniform'):
        '''
        param1: lower bound (Uniform) or mean (Gaussian)
        param2: upper bound (Uniform) or covariance /precision matrix (Gaussian) 
        param2_type: type for param2, 
                     support: std (param2 is a 1D array defining standard deviation)
                              covariance (param2 is a 2D array defining full covariance matrix)
                              scale_tril (param2 is a 2D array defining lower triangular matrix L)
                              precision  (param2 is a 2D array defining precision (inverse of covariance) matrix)
        param1 & param2 should match the dimensionality of the problem
        prior_type: 'Uniform' or 'Normal'
        '''
        self.param1 = torch.as_tensor(param1)
        self.param2_type = param2_type
        self.dim = param1.size
        self.prior_type = prior_type
        self.smooth_matrix = smooth_matrix
        if param2_type == 'scale_tril':
            offset = (param2[1:] == 0).sum(axis = 1)
            # R_full = sparse.diags(param2[1:], -offset, shape = (self.dim, self.dim))
            # self.param2 = sparse.diags(3*param2[0], shape = (self.dim, self.dim)) @ R_full
            self.param2 = sparse.diags(param2[0], shape = (self.dim, self.dim))
        else:
            self.param2 = torch.as_tensor(param2)
        if param2_type == 'local_cor':
            self.param3 = torch.as_tensor(param3)
            self.patch_size = patch_size
            self.get_patch(self.patch_size)

    def get_patch(self, patch_size):
        blocky, blockx, blockz = 10, 10, 8
        ny, nx, nz = 101, 101, 62

        y_index = np.random.randint(0, ny + blocky + 1, size = (patch_size))
        x_index = np.random.randint(0, nx + blockx + 1, size = (patch_size))
        z_index = np.random.randint(0, nz + blockz + 1, size = (patch_size))
        self.patch = np.zeros([patch_size, (ny+blocky*2), (nx+blockx*2), (nz+blockz*2)], dtype = 'bool')

        for i in range(patch_size):
            self.patch[i, y_index[i]:y_index[i]+blocky, x_index[i]:x_index[i]+blockx
                        , z_index[i]:z_index[i]+blockz] = True
        self.patch = self.patch.reshape(patch_size, -1)

    def extend_model(self, x):
        blocky, blockx, blockz = 10, 10, 8
        ny, nx, nz = 101, 101, 62
        nty, ntx, ntz = ny+blocky*2, nx+blockx*2, nz+blockz*2

        model = x.reshape(-1, ny, nx, nz)
        z = torch.zeros((model.shape[0], nty, ntx, ntz))
        z[:, blocky:-blocky, blockx:-blockx, blockz:-blockz] = model
        z[:, :blocky] = z[:,blocky][:,None].detach()
        z[:, -blocky:] = z[:,-blocky-1][:,None].detach()
        z[:, :, :blockx] = z[:, :, blockx][:,:,None].detach()
        z[:, :, -blockx:] = z[:, :, -blockx-1][:,:,None].detach()
        z[:, :, :, :blockz] = z[:,:,:,blockz][...,None].detach()
        z[:, :, :, -blockz:] = z[:,:,:,-blockz-1][...,None].detach()
        if x.dim() == 1:
            z = torch.squeeze(z.reshape([model.shape[0], -1]), dim = 0)
        elif x.dim() == 2:
            z = z.reshape([model.shape[0], -1])
        return z

    def log_prob(self, x):
        '''
        Compute log probability using PyTorch, such that the result can be back propagated
        '''
        if self.prior_type == 'Uniform':
            # if prior is a Uniform distribution, log_prior value is a constant 
            logp = - torch.log(self.param2 - self.param1).sum()
        elif self.prior_type == 'Normal':
            # the calculation of logp depends on param2_type
            # log_normal requires det(cov), currently only diagonal cov correctly calculate det(cov)
            # other param2_type options simply ignore the  calculation of det(cov) 
            # since this constant value won't acturally affect trainging
            # tmp = x - self.param1
            if self.param2_type == 'std':
                logp = -0.5 * torch.sum((tmp / self.param2)**2, axis = 1) \
                        - torch.log(self.param2).sum() - 0.5 * self.dim * np.log(2*np.pi)
            elif self.param2_type == 'precision':
                inv_cov = torch.diag(1 / self.param2**2)
                logp = - 0.5 * (tmp @ inv_cov @ tmp.T).diag() - 0.5 * self.dim * np.log(2*np.pi)

                # logp = -0.5 * (tmp @ self.param2 @ tmp.T).diag() - 0.5 * self.dim * np.log(2*np.pi)
                # logp = -0.5 * (torch.matmul(torch.matmul(tmp, self.param2)), tmp.T).diag() \
                #         - 0.5 * self.dim * np.log(2*np.pi)
            elif self.param2_type == 'scale_tril':
                logp = LogNormal.apply(x, self.param1.numpy(), self.param2)
            elif self.param2_type == 'local_cor':
                m = x.shape[0]
                # tmp = (x - self.param1) / self.param2
                tmp = (self.extend_model(x) - self.extend_model(self.param1)) / self.extend_model(self.param2)

                self.get_patch(self.patch_size)
                x_patch = torch.broadcast_to(
                                            tmp[:,None], (m, self.patch_size, self.patch.shape[-1])
                                        )[:,self.patch].reshape([m, self.patch_size, -1])
                logp_patch = - 0.5 * (torch.tensordot(x_patch, self.param3, dims = 1) * x_patch).sum(axis = -1)
                logp = logp_patch.mean(axis = -1) / self.param3.shape[0] * self.dim
                # logp = (logp_patch * self.scale).mean(axis = -1) / self.param3.shape[0]*self.dim

        if self.smooth_matrix is not None:
            logp_smooth = Smoothing.apply(x, self.smooth_matrix)
            return logp + logp_smooth
        return logp