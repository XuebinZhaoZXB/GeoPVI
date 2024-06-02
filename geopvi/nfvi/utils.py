import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3



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
        return torch.tensor(epsilons.T)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        this function returns the gradient w.r.t the input tensor in the forward function
        therefore, the return shape should be the same as the shape of input tensor
        '''
        input, grad = ctx.saved_tensors
        grad_input = (grad_output @ grad)
        return grad_input, None, None


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse=False,
                      tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    # if inputs[inside_intvl_mask].nelement() != 0:
    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet


def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives \
            + input_derivatives_plus_one - 2 * input_delta) \
            + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) \
            * (input_derivatives + input_derivatives_plus_one \
            - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
                      + ((input_derivatives + input_derivatives_plus_one \
                      - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * root.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) \
                    + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives \
                      + input_derivatives_plus_one - 2 * input_delta) \
                      * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * theta.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


class MaskedLinear(nn.Linear):
    """ 
    same as Linear except has a configurable mask on the weights 
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, in_dim, hidden_sizes, out_dim, num_masks=1, natural_ordering=False):
        """
        in_dim: integer; number of inputs
        hidden_sizes: a list of integers; number of units in each hidden layers
        out_dim: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_sizes = hidden_sizes
        assert self.out_dim % self.in_dim == 0, "out_dim must be integer multiple of in_dim"
        
        # define a simple MLP neural net
        self.net = []
        hs = [in_dim] + hidden_sizes + [out_dim]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
        
        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings
        
        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.
        
    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)
        
        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        
        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.in_dim) if self.natural_ordering \
                                            else rng.permutation(self.in_dim)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.in_dim-1, size=self.hidden_sizes[l])
        
        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])
        
        # handle the case where nout = in_dim * k, for integer k > 1
        if self.out_dim > self.in_dim:
            k = int(self.out_dim / self.in_dim)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
        
        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x):
        return self.net(x)


class MaskedNN(nn.Module):
    """ 
    Masked Neural Network for Auto-regressive MLP, wrapper around MADE net 
    """

    def __init__(self, in_dim, out_dim, hidden_dim, num_masks = 1, natural_ordering = True):
        super().__init__()
        self.net = MADE(in_dim, [hidden_dim, hidden_dim], out_dim, \
                        num_masks=num_masks, natural_ordering=natural_ordering)
        
    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        hs = [in_dim] + hidden_dim
        self.network = []
        for h0, h1 in zip(hs, hs[1:]):
            self.network.extend([
                    nn.Linear(h0, h1),
                    # nn.BatchNorm1d(h1),
                    nn.Tanh(),
                    # nn.LeakyReLU(0.2),
                ])
        self.network.extend([nn.Linear(h1, out_dim)])
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        return self.network(x)


class CNN1D(nn.Module):
    """
    1D Convolutional neural network (plus one fully connected layer)
    to construct NN in the coupling flow
    """
    def __init__(self, in_dim, out_dim, hidden_dim, 
                    conv_filter = [32, 32], conv_kernel = [9, 9], pool = 2):
        super().__init__()
        if len(conv_filter) != len(conv_kernel):
            print('Convolutional layer: filter size and kernel size mismatch')
            exit()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.conv_filter = conv_filter
        self.conv_kernel = conv_kernel
        
        # Convolutional layer
        self.conv = []
        h_conv = [1] + conv_filter
        for h0, h1, k in zip(h_conv, h_conv[1:], conv_kernel):
            self.conv.extend([
                    nn.Conv1d(h0, h1, k, 1, k // 2),
                    nn.ReLU(),
                    nn.MaxPool1d(pool),
                ])
        if self.conv == []:
            h1 = 1
        self.conv = nn.Sequential(*self.conv)
        conv_out_dim = self.calculate_size(self.in_dim)
        # Simple MLP neural net
        self.fcnn = []
        if len(hidden_dim) > 0:
            # hs = [conv_out_dim * h1] + hidden_dim + [out_dim*kernel*2 + kernel]
            hs = [conv_out_dim * h1] + hidden_dim
            for h0, h1 in zip(hs, hs[1:]):
                self.fcnn.extend([
                        nn.Linear(h0, h1),
                        # nn.BatchNorm1d(h1),
                        nn.ReLU(),
                    ])
            self.fcnn.extend([nn.Linear(h1, out_dim)])
        else:
            self.fcnn.extend([nn.Linear(conv_out_dim * h1, out_dim)])
        # self.fcnn.pop() # pop out the last ReLU for the output layer
        self.fcnn = nn.Sequential(*self.fcnn)

    def calculate_size(self, input_size):
        """
        Calculate the output size of the convolutional layer through one forward pass of the net
        """
        x = torch.randn(input_size).reshape(-1, 1, input_size)
        output = self.conv(x)
        return output.size()[-1]

    def forward(self, x):
        x = self.conv(torch.unsqueeze(x, 1))
        x = x.view(x.size(0), -1)
        return self.fcnn(x)
