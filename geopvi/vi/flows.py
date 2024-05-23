import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from vi.utils import *

"""
The field of normalising flows is developing very fast, in this package we 
just implemented several flows that are tested to be effective as of 2021.
Feel free to add more flows!
"""

# supported non-linearities: note that the function must be invertible
functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) + \
                            (x < 0).type(torch.FloatTensor) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor) + \
                     (x < 0).type(torch.FloatTensor) * torch.exp(x)
}

class TriangularSolve(Function):
    @staticmethod
    def forward(ctx, input, L):
        '''
        Solve triangular system L*x = b for x given L and b
        L is parametrised by structured kernel with sparse diagonals
        this assumes dimensionality of L is high such that you can't solve the above problem using triangular_solve
        therefore L is represented as scipy.sparse.diags object and solved using scipy.linalg.spsolve_triangular
        input: input tensor b (nsampls * ndim)
        L: a sparse lower triangular matrix with cov = L@LT
        grad: gradient of output (x) w.r.t. input (b) (nsamples * ndim)
        '''
        epsilons = linalg.spsolve_triangular(L.tocsr(), input.detach().numpy().T, lower = True)
        grad = linalg.spsolve_triangular(L.tocsr(), np.eye(L.shape[0]), lower = True)
        ctx.save_for_backward(input, torch.tensor(grad))
        return torch.tensor(logp)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        this function returns the gradient w.r.t the input tensor in the forward function
        therefore, the return shape should be the same as the shape of input tensor
        '''
        input, grad = ctx.saved_tensors
        grad_input = (grad_output @ grad)
        return grad_input, None, None


class Real2Constr(nn.Module):
    """
    Transform from space of real numbers to a constrained space
    """
    def __init__(self, lower = None, upper = None):
        ### TODO update with dim attribute
        super().__init__()
        # self.dim = dim
        self.upper = torch.tensor(upper)
        self.lower = torch.tensor(lower)

    def forward(self, x, train = True):
        if self.lower is None and self.upper is None:
            z = x
            log_det = 0
        elif self.lower is not None and self.upper is None:
            z = self.lower + torch.exp(x)
            log_det = x.sum(axis = -1)
        elif self.lower is None and self.upper is not None:
            z = self.upper - torch.exp(x)
            log_det = x.sum(axis = -1)
        else:
            z = self.lower + (self.upper - self.lower) / (1 + torch.exp(-x))
            log_det = (torch.log(self.upper - self.lower) - x - 
                        2 * torch.log(1 + torch.exp(-x))).sum(axis = -1)
        return z, log_det

    def inverse(self, z):
        if self.lower is None and self.upper is None:
            x = z
            log_det = 0
        elif self.lower is not None and self.upper is None:
            x = torch.log(z - self.lower)
            log_det = torch.log(1 / (z - self.lower)).sum(axis = -1)
        elif self.lower is None and self.upper is not None:
            x = torch.log(self.upper - z)
            log_det = torch.log(1 / (self.upper - z)).sum(axis = -1)
        else:
            x = torch.log(z - self.lower) - torch.log(self.upper - z)
            log_det = torch.log(1. / (z - self.lower) + 1. / (self.upper - z)).sum(axis = -1)
        return x, log_det


class Constr2Real(nn.Module):
    """
    Transform from a constrained space to space of real numbers
    """
    def __init__(self, lower = None, upper = None):
        ### TODO update with dim attribute
        super().__init__()
        # self.dim = dim
        self.upper = torch.tensor(upper)
        self.lower = torch.tensor(lower)

    def forward(self, x, train = True):
        if self.lower is None and self.upper is None:
            z = x
            log_det = 0
        elif self.lower is not None and self.upper is None:
            z = torch.log(x - self.lower)
            log_det = torch.log(1 / (x - self.lower)).sum(axis = -1)
        elif self.lower is None and self.upper is not None:
            z = torch.log(self.upper - x)
            log_det = torch.log(1 / (self.upper - x)).sum(axis = -1)
        else:
            z = torch.log(x - self.lower) - torch.log(self.upper - x)
            log_det = torch.log(1. / (x - self.lower) + 1. / (self.upper - x)).sum(axis = -1)
        return z, log_det

    def inverse(self, z):
        if self.lower is None and self.upper is None:
            x = z
            log_det = 0
        elif self.lower is not None and self.upper is None:
            x = self.lower + torch.exp(z)
            log_det = z.sum(axis = -1)
        elif self.lower is None and self.upper is not None:
            x = self.upper - torch.exp(z)
            log_det = z.sum(axis = -1)
        else:
            x = self.lower + (self.upper - self.lower) / (1 + torch.exp(-z))
            log_det = (torch.log(self.upper - self.lower) - 
                        z - 2 * torch.log(1 + torch.exp(-z))).sum(axis = -1) 
        return x, log_det


class Linear(nn.Module):
    """
    Pytorch implementation for a linear transform: z = u + Lx
    If this flow is used for ADVI, then the covariance matrix is \Sigma = L^T.L
    Different methods (kernels) are provided to constrcut L, including:
    diagonal: only diagonal of L is considered, ignore correlations - mean field ADVI
    structured: part of disgonals are considered - PSVI
    fullrank: all elements of L - full rank ADVI
    """
    def __init__(self, dim, kernel = 'diagonal', mask = None, param = None):
        super().__init__()
        self.dim = dim
        self.kernel = kernel
        self.u = nn.Parameter(torch.zeros(dim))
        self.diag = nn.Parameter(torch.zeros(dim))
        if kernel == 'fullrank':
            self.non_diag = nn.Parameter(torch.zeros(int(dim * (dim - 1)/2)))
        elif kernel == 'diagonal':
            self.non_diag = None
        elif kernel == 'structured':
            if mask is None:
                self.mask = np.ones([1, dim])
                self.mask[0, -1] = 0.
            else:
                self.mask = mask
            self.offset = (self.mask == 0).sum(axis = 1)
            self.non_diag = nn.Parameter(torch.zeros(self.mask.shape))

        # Load previous results: can be used to resume from previous run or for multiscale FWI 
        if param is not None:
            if param.size >= self.dim * 1:
                self.u = nn.Parameter(torch.from_numpy(param[:self.dim]))
            if param.size >= self.dim * 2:
                tmp = np.log(param[self.dim : self.dim * 2])
                self.diag = nn.Parameter(torch.from_numpy(tmp))
            if param.size > self.dim * 2:
                if kernel == 'fullrank': 
                    if param[self.dim*2:].size == int(dim * (dim - 1)/2):
                        self.non_diag = nn.Parameter(torch.from_numpy(param[self.dim*2:]))
                    # else:
                    #     TODO: consider param only provides part of off-diagonal elements
                    #     raise ValueError("Shape of mask and parameter does not match!")
                elif kernel == 'structured':
                    if param[self.dim*2:].size == self.mask.size:
                        self.non_diag = nn.Parameter(torch.from_numpy(param[self.dim*2:].reshape(self.mask.shape)))
                    # else:
                    #     # TODO: consider param only provides part of off-diagonal blocks
                    #     raise ValueError("Shape of mask and parameter does not match!")

    def create_lower_triangular(self, diagonal=0):
        lower = torch.zeros((self.dim, self.dim))
        if self.kernel == 'fullrank':
            lower[np.tril_indices(self.dim, diagonal)] = self.non_diag
        return lower

    def forward(self, x):
        diag = torch.exp(self.diag)
        if self.kernel == 'fullrank':
            L = torch.diag(diag) + self.create_lower_triangular(diagonal=-1)
            z = self.u + torch.matmul(L, x.T).T
        elif self.kernel == 'diagonal':
            z = self.u + diag * x
        elif self.kernel == 'structured':
            # TODO: use torch.sparse to represent L
            # Current torch version used does not support torch.sparse.spdiags
            tmp = x[:,None,:] * self.non_diag
            tmp1 = torch.zeros(tmp.shape)
            tmp1[:, np.sort(self.mask, axis = 1)] = tmp[:, self.mask]
            z = self.u + diag * x + tmp1.sum(axis = 1)
        log_det = torch.log(diag).sum().repeat(x.shape[0])
        return z, log_det

    def inverse(self, z):
        diag = torch.exp(self.diag)
        if self.kernel == 'fullrank':
            b = (z - self.u).T
            L = torch.diag(diag) + self.create_lower_triangular(diagonal=-1)
            # For newer Pytorch version, torch.triangular_solve is deprecated 
            # thus use torch.linalg.solve_triangular()
            # x = torch.linalg.solve_triangular(L, b, upper=False).T
            x = torch.triangular_solve(b, L, upper=False)[0].T  
        elif self.kernel == 'diagonal':
            x = (z - self.u) / diag
        elif self.kernel == 'structured':
            # Current torch version used doesn't support torch.sparse.spdiags
            # thus the inverse for this case is implemented using scipy.sparse.diags
            sparse_diagonals = torch.vstack([diag, self.non_diag]).detach().numpy()
            L = sparse.diags(sparse_diagonals, -self.offset, shape = (self.dim, self.dim))
            x = TriangularSolve.apply(z - self.u, L)
            # raise NotImplementedError("Inverse transform not supported for sturctured kernel")
        log_det = - torch.log(diag).sum().repeat(x.shape[0])
        return x, log_det


class Planar(nn.Module):
    """
    Planar flow.

        z = f(x) = x + u h(wᵀx + b)

    [Rezende and Mohamed, 2015]
    """
    def __init__(self, dim, nonlinearity=torch.tanh):
        super().__init__()
        self.dim = dim
        self.h = nonlinearity
        self.w = nn.Parameter(torch.Tensor(dim))
        self.u = nn.Parameter(torch.Tensor(dim))
        self.b = nn.Parameter(torch.Tensor(1))
        self.reset_parameters(dim)

    def reset_parameters(self, dim):
        init.uniform_(self.w, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.u, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.b, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x, train = True):
        """
        Given x, returns z and the log-determinant log|df/dx|.

        Returns
        -------
        """
        if self.h in (F.elu, F.leaky_relu):
            u = self.u
        elif self.h == torch.tanh:
            scal = torch.log(1+torch.exp(self.w @ self.u)) - self.w @ self.u - 1
            u = self.u + scal * self.w / torch.norm(self.w) ** 2
        else:
            raise NotImplementedError("Non-linearity is not supported.")
        lin = torch.unsqueeze(x @ self.w, 1) + self.b
        z = x + u * self.h(lin)
        phi = functional_derivatives[self.h](lin) * self.w
        log_det = torch.log(torch.abs(1 + phi @ u) + 1e-4)
        return z, log_det

    def inverse(self, z):
        raise NotImplementedError("Planar flow has no algebraic inverse.")


class Radial(nn.Module):
    """
    Radial flow.

        z = f(x) = = x + β h(α, r)(z − z0)

    [Rezende and Mohamed 2015]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.x0 = nn.Parameter(torch.Tensor(dim))
        self.log_alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.reset_parameters(dim)

    def reset_parameters(self, dim):
        init.uniform_(self.x0, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.log_alpha, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.beta, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x, train = True):
        """
        Given x, returns z and the log-determinant log|df/dx|.
        """
        m, n = x.shape
        r = torch.norm(x - self.x0, dim=1).unsqueeze(1)
        h = 1 / (torch.exp(self.log_alpha) + r)
        beta = -torch.exp(self.log_alpha) + torch.log(1 + torch.exp(self.beta))
        z = x + beta * h * (x - self.x0)
        log_det = ((n - 1) * torch.log(1 + beta * h) + \
                  torch.log(1 + beta * h - \
                            beta * r / (torch.exp(self.log_alpha) + r) ** 2)).squeeze()
        return z, log_det

    def inverse(self, z):
        raise NotImplementedError("Radial flow has no algebraic inverse.")


class RealNVP(nn.Module):
    """
    RealNVP: real-valued non-volume preserving flow
    Calling one RealNVP layer is actually applying two coupling layers

    [Dinh et. al. 2017 - ICLR]
    """
    def __init__(self, dim, hidden_dim = [100], base_network = FCNN):
        super().__init__()
        self.dim = dim
        self.t0 = base_network(dim - dim // 2, dim // 2, hidden_dim)
        self.s0 = base_network(dim - dim // 2, dim // 2, hidden_dim)
        self.t1 = base_network(dim // 2, dim - dim // 2, hidden_dim)
        self.s1 = base_network(dim // 2, dim - dim // 2, hidden_dim)

    def forward(self, x, train = True):
        # z = torch.zeros_like(x)
        # x0, x1 = x[:,::2], x[:,1::2]
        x0, x1 = x.chunk(2, dim = 1)
        t0_transformed = self.t0(x0)
        s0_transformed = torch.tanh(self.s0(x0))
        x1 = t0_transformed + x1 * torch.exp(s0_transformed)
        t1_transformed = self.t1(x1)
        s1_transformed = torch.tanh(self.s1(x1))
        x0 = t1_transformed + x0 * torch.exp(s1_transformed)
        log_det = torch.sum(s0_transformed, dim=1) + \
                  torch.sum(s1_transformed, dim=1)

        return torch.cat([x0, x1], dim = 1), log_det

    def inverse(self, z):
        # x = torch.zeros_like(z)
        # z0, z1 = z[:,::2], z[:,1::2]
        z0, z1 = z.chunk(2, dim = 1)
        t1_transformed = self.t1(z1)
        s1_transformed = torch.tanh(self.s1(z1))
        z0 = (z0 - t1_transformed) * torch.exp(-s1_transformed)
        t0_transformed = self.t0(z0)
        s0_transformed = torch.tanh(self.s0(z0))
        z1 = (z1 - t0_transformed) * torch.exp(-s0_transformed)
        log_det = torch.sum(-s0_transformed, dim=1) + \
                  torch.sum(-s1_transformed, dim=1)
        
        return torch.cat([z0, z1], dim = 1), log_det



class SIAF(nn.Module):
    """
    Inverse auto-regressive flow using slow version with explicit networks per dim

    [Kingma et al. 2016]
    """
    def __init__(self, dim, hidden_dim = [8], base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.initial_param = nn.Parameter(torch.Tensor(2))
        for i in range(1, dim):
            self.layers += [base_network(i, 2, hidden_dim)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, train = True):
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0])
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            # z[:, i] = (x[:, i] - mu) / torch.exp(alpha)
            z[:, i] = x[:, i] * torch.exp(alpha) + mu
            log_det += alpha
        z = z.flip(dims=(1,))
        return z, log_det

    def inverse(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0])
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            # x[:, i] = mu + torch.exp(alpha) * z[:, i]
            x[:, i] = (z[:, i] - mu) * torch.exp(-alpha)
            log_det -= alpha
        return x, log_det


class IAF(nn.Module):
    """
    Inverse auto-regressive flow using Masked matrix for fast forward

    [Kingma et al. 2016]
    """
    
    def __init__(self, dim, hidden_dim = [100], base_network=MaskedNN, \
                    renew_mask_every = 20, nmasks = 10):
        super().__init__()
        self.dim = dim
        self.nmasks = nmasks
        self.renew_mask_every = renew_mask_every
        self.net = base_network(dim, dim*2, hidden_dim, \
                    num_masks = self.nmasks, natural_ordering = True)
        self.seed = 0

    def forward(self, x, train = True):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        out = torch.zeros((x.shape[0], 2 * self.dim))
        nsamples = 1 if train else self.nmasks

        # fetch the next seed to decide update mask or not during training
        self.seed = (self.seed + 1) % self.renew_mask_every

        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if self.seed == 0 or (not train): # if in test, cycle masks every time
                self.net.net.update_masks()
            # forward the model
            out += self.net(x)
        out /= nsamples

        # out = self.net(x)
        mu, alpha = out.split(self.dim, dim=1)
        z = x * torch.exp(alpha) + mu
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,))
        log_det = torch.sum(alpha, dim=1)
        return z, log_det
    
    def inverse(self, z, train = False):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0])
        z = z.flip(dims=(1,)) 
        for i in range(self.dim):
            out = self.net(x.clone()) # clone to avoid in-place op errors if using IAF
            mu, alpha = out.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - mu[:, i]) * torch.exp(-alpha[:, i])
            log_det -= alpha[:, i]
        return x, log_det


class ActNorm(nn.Module):
    """
    ActNorm layer.

    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype = torch.float))
        self.log_sigma = nn.Parameter(torch.zeros(dim, dtype = torch.float))

    def forward(self, x, train = True):
        z = x * torch.exp(self.log_sigma) + self.mu
        log_det = torch.sum(self.log_sigma)
        return z, log_det

    def inverse(self, z):
        x = (z - self.mu) / torch.exp(self.log_sigma)
        log_det = -torch.sum(self.log_sigma)
        return x, log_det


class OneByOneConv(nn.Module):
    """
    Invertible 1x1 convolution.

    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        W, _ = sp.linalg.qr(np.random.randn(dim, dim))
        P, L, U = sp.linalg.lu(W)
        self.P = torch.tensor(P, dtype = torch.float)
        self.L = nn.Parameter(torch.tensor(L, dtype = torch.float))
        self.S = nn.Parameter(torch.tensor(np.diag(U), dtype = torch.float))
        self.U = nn.Parameter(torch.triu(torch.tensor(U, dtype = torch.float),
                              diagonal = 1))
        self.W_inv = None

    def forward(self, x, train = True):
        L = torch.tril(self.L, diagonal = -1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal = 1)
        z = x @ self.P @ L @ (U + torch.diag(self.S))
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def inverse(self, z):
        if not self.W_inv:
            L = torch.tril(self.L, diagonal = -1) + \
                torch.diag(torch.ones(self.dim))
            U = torch.triu(self.U, diagonal = 1)
            W = self.P @ L @ (U + torch.diag(self.S))
            self.W_inv = torch.inverse(W)
        x = z @ self.W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


class Invertible1x1Conv(nn.Module):
    """ 
    As introduced in Glow paper.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, train = True):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def inverse(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


class Reverse_order(nn.Module):
    """
    Reverse the order of the input vector
    """
    def __init__(self, dim):
        super().__init__()
        self.index = np.linspace(dim - 1,0, dim)

    def forward(self, x, train = True):
        z = x[:, self.index]
        log_det = torch.zeros(x.shape[0])
        return z, log_det

    def inverse(self, z):
        x = z[:, np.argsort(self.index)]
        log_det = torch.zeros(x.shape[0])
        return x, log_det


class Permute(nn.Module):
    """
    Random permute (or re-order) model parameters
    """
    def __init__(self, dim, seed = 1, nx = 1, ny = 1, block_x = 1, block_y = 1, last_flow = False):
        super().__init__()
        subdomain = block_x * block_y
        self.seed = seed
        self.last_flow = last_flow
        rng = np.random.RandomState(seed)

        x_dim = np.linspace(0, nx, block_x+1, dtype = 'int32')[1:]
        x_dim[1:] -= x_dim[:-1].copy()
        y_dim = np.linspace(0, ny, block_y+1, dtype = 'int32')[1:]
        y_dim[1:] -= y_dim[:-1].copy()
        sub_dim = (x_dim.reshape(-1, 1) * y_dim).reshape(-1)
        cum_dim = np.insert(np.cumsum(sub_dim), 0, 0)

        if self.last_flow is False:
            self.index = np.zeros(dim, dtype = 'int32')
            for i in range(subdomain):
                self.index[cum_dim[i]:cum_dim[i+1]] = rng.permutation(sub_dim[i]) + cum_dim[i]
        else:
            self.index = np.concatenate(
                [np.concatenate(
                    [np.arange(cum_dim[i*block_y + j], cum_dim[i*block_y + j + 1]).reshape(x_dim[i], y_dim[j])
                    for j in range(block_y)], axis = 1) for i in range(block_x)], axis = 0).reshape(-1)
    
    def forward(self, x, train = True):
        z = x[:, self.index]
        log_det = torch.zeros(x.shape[0])
        return z, log_det

    def inverse(self, z):
        x = z[:, np.argsort(self.index)]
        log_det = torch.zeros(x.shape[0])
        return x, log_det


class NSF_SAR(nn.Module):
    """
    Neural spline flow, auto-regressive, slow version with explicit networks per dim

    [Durkan et al. 2019]
    """
    def __init__(self, dim, K = 8, B = 3, hidden_dim = 50, base_network = FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.layers = nn.ModuleList()
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        for i in range(1, dim):
            self.layers += [base_network(i, 3 * K - 1, hidden_dim)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2)

    def forward(self, x, train = True):
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0])
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(
                x[:, i], W, H, D, inverse=False, tail_bound=self.B)
            log_det += ld
        z = z.flip(dims = (1,))
        return z, log_det

    def inverse(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(x.shape[0])
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(
                z[:, i], W, H, D, inverse = True, tail_bound = self.B)
            log_det += ld
        return x, log_det


class NSF_AR(nn.Module):
    """
    Neural spline flow, auto-regressive, masked for fast forward

    [Durkan et al. 2019]
    """
    def __init__(self, dim, K = 8, B = 3, hidden_dim = 100, base_network = MaskedNN, 
                    renew_mask_every = 100, nmasks = 1):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.nmasks = nmasks
        self.renew_mask_every = renew_mask_every
        self.net = base_network(dim, dim * (3 * self.K - 1), hidden_dim, 
                    num_masks = self.nmasks, natural_ordering = False)
        self.seed = 0

    def forward(self, x, train = True):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        log_det = torch.zeros(x.shape[0])
        z = torch.zeros_like(x)
        out = torch.zeros((x.shape[0], (3 * self.K - 1) * self.dim))
        nsamples = 1 if train else self.nmasks

        # fetch the next seed to decide update mask or not during training
        self.seed = (self.seed + 1) % self.renew_mask_every
        # print(self.seed)

        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if self.seed == 0 or (not train): # if in test, cycle masks every time
                self.net.net.update_masks()
            # forward the model
            out += self.net(x)
        out /= nsamples

        out = out.reshape(-1, self.dim, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        z, ld = unconstrained_RQS(
            x, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        z = z.flip(dims=(1,))
        return z, log_det

    def inverse(self, z, train = False):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0])
        z = z.flip(dims=(1,)) 
        for i in range(self.dim):
            out = self.net(x.clone()).reshape(-1, self.dim, 3 * self.K - 1)
            W, H, D = torch.split(out, self.K, dim = 2)
            W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)            
            x[:, i], ld = unconstrained_RQS(
                z[:, i], W[:, i, :], H[:, i, :], D[:, i, :], inverse=True, tail_bound=self.B)
            log_det += ld
        return x, log_det


class NSF_CL(nn.Module):
    """
    Neural spline flow, coupling layer.

    [Durkan et al. 2019]
    """
    # def __init__(self, dim, K = 5, B = 3, subdomain = 1, hidden_dim = 150, base_network = 'FCNN'):
    def __init__(self, dim, K = 5, B = 3, nx = 1, ny = 1, block_x = 1, block_y = 1, 
            hidden_dim = [50], conv_filter = [32, 16], conv_kernel = [9, 9], pool = 2,
            base_network = 'FCNN'):
        super().__init__()
        self.dim = dim
        self.subdomain = block_x * block_y
        x_dim = np.linspace(0, nx, block_x+1, dtype = 'int32')[1:]
        x_dim[1:] -= x_dim[:-1].copy()
        y_dim = np.linspace(0, ny, block_y+1, dtype = 'int32')[1:]
        y_dim[1:] -= y_dim[:-1].copy()
        self.sub_dim = (x_dim.reshape(-1, 1) * y_dim).reshape(-1)
        self.cum_dim = np.insert(np.cumsum(self.sub_dim), 0, 0)

        self.K = K
        self.B = B

        self.f0s = nn.ModuleList()
        self.f1s = nn.ModuleList()
        for i in range(self.subdomain):
            if base_network == 'FCNN':
                self.f0s += [FCNN(self.sub_dim[i] - self.sub_dim[i] // 2, \
                                self.sub_dim[i] // 2 * (3 * K - 1), hidden_dim)]
                self.f1s += [FCNN(self.sub_dim[i] // 2, \
                                (self.sub_dim[i] - self.sub_dim[i] // 2) * (3 * K - 1), hidden_dim)]
            else:
                self.f0s += [
                    CNN1D(self.sub_dim[i] - self.sub_dim[i] // 2, self.sub_dim[i] // 2 * (3 * K - 1),
                        hidden_dim, conv_filter = conv_filter, conv_kernel = conv_kernel, pool = pool
                    )
                ]
                self.f1s += [
                    CNN1D(self.sub_dim[i] // 2, (self.sub_dim[i] - self.sub_dim[i] // 2) * (3 * K - 1),
                        hidden_dim, conv_filter = conv_filter, conv_kernel = conv_kernel, pool = pool
                    )
                ]

    def forward(self, x, train = True):
        log_det = torch.zeros(x.shape[0])
        z = torch.zeros_like(x)
        for i in range(self.subdomain):
            x0, x1 = x[:, self.cum_dim[i]:self.cum_dim[i+1]].chunk(2, dim = 1)
            out = self.f0s[i](x0).reshape(-1, self.sub_dim[i] // 2, 3 * self.K - 1)
            W, H, D = torch.split(out, self.K, dim = 2)
            W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x1, ld = unconstrained_RQS(
                x1, W, H, D, inverse=False, tail_bound=self.B)
            log_det += torch.sum(ld, dim = 1)

            out = self.f1s[i](x1).reshape(-1, self.sub_dim[i] - self.sub_dim[i] // 2, 3 * self.K - 1)
            W, H, D = torch.split(out, self.K, dim = 2)
            W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x0, ld = unconstrained_RQS(
                x0, W, H, D, inverse=False, tail_bound=self.B)
            log_det += torch.sum(ld, dim = 1)

            z[:, self.cum_dim[i]:self.cum_dim[i+1]] = torch.cat([x0, x1], dim =1)

        return z, log_det
        
    def inverse(self, z):
        log_det = torch.zeros(z.shape[0])
        x = torch.zeros_like(z)
        for i in range(self.subdomain):
            z0, z1 = z[: self.cum_dim[i]:self.cum_dim[i+1]].chunk(2, dim = 1)
            out = self.f1s[i](z1).reshape(-1, self.sub_dim[i] - self.sub_dim[i] // 2, 3 * self.K - 1)
            W, H, D = torch.split(out, self.K, dim = 2)
            W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z0, ld = unconstrained_RQS(
                z0, W, H, D, inverse=True, tail_bound=self.B)
            log_det += torch.sum(ld, dim = 1)

            out = self.f0s[i](z0).reshape(-1, self.sub_dim[i] // 2, 3 * self.K - 1)
            W, H, D = torch.split(out, self.K, dim = 2)
            W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z1, ld = unconstrained_RQS(
                z1, W, H, D, inverse = True, tail_bound = self.B)
            log_det += torch.sum(ld, dim = 1)

            x[:, self.cum_dim[i]:self.cum_dim[i+1]] = torch.cat([z0, z1], dim =1)

        return x, log_det
