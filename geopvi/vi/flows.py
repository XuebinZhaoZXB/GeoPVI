import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


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
