import math
import time
import numpy as np 
import torch


def projection_on_simplex(x):
    u = torch.sort(x, descending=True)[0]
    indices = torch.arange(1, u.shape[0] + 1)
    rho_nz = u + 1. / indices * (1. - torch.cumsum(u, dim=0)) > 0
    rho = indices[rho_nz].max()
    lmbda = 1. / rho * (1. - u[:rho].sum())
    out = torch.max(
        torch.stack([x + lmbda, torch.zeros_like(x, dtype=torch.float32)]), dim=0).values
    return out / out.sum()

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
        log_det = (torch.log(upper - lower) - x - 2 * torch.log(1 + torch.exp(-x))).sum(axis = -1)
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
    def __init__(self, dim, kernel = 'diagonal', mask = None, base = 'Normal', perturb = 0.5,
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
            else
                if np.isscalar(upper):
                    upper = np.full((dim,) upper)
                self.upper = torch.from_numpy(upper)
            if lower is None:
                self.lower = None
            else:
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

        mu, std = param[:self.dim * 2].chunk(2)
        std = torch.log(torch.exp(std) + 1)
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
        lg1 = 0.
        if self.constrained:
            x, lg1 = const_2_real(x, lower = self.lower, upper = self.upper)    # lg1: nsamples * 1

        mu, std = params[:, :self.dim * 2].chunk(2, dim = -1)            # ncomponents * ndim
        std = torch.log(torch.exp(std) + 1)
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
        
        return x, log_base + lg1[:, None] + lg2 + lg3

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
            mu = mus[k,:] + torch.randn(self.dim) * torch.log(1 + torch.exp(log_sigmas[k, :])) * self.perturb
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
        covariances = params[:, self.dim:]
        covariances[:, :self.dim] = torch.log(torch.exp(covariances[:, :self.dim]) + 1)
        return {"mus": mus, "covariances": covariances}


class BoostingGaussian():
    '''
    A class that defines a variational distribution through boosting variational inference
    which essentially builds a mixture of distributions (e.g., Gaussians) to approximate the true posterior pdf
    '''
    def __init__(
        self, componentDistribution, log_posterior, lmb = lambda itr : 1, start_component = 0,
        weight_cal = 0, weight_init = 'equal', lr_weight = 0.005, niter_weight = 100, nsample_weight = 1
    ):
        self.N = start_component # current num of components
        self.log_posterior = log_posterior
        self.component_dist = componentDistribution
        self.lmb = lmb
        self.weight_cal = weight_cal
        self.weight_init = weight_init
        self.lr_w = lr_weight
        self.niter_weight = niter_weight
        self.nsample_weight = nsample_weight

        self.weights = torch.empty(0) # weights
        self.params = torch.empty((0, 0)) # components' parameters

    def _init_one_weight(self, N):
        if self.weight_init == 'decreasing':
            weight = 0.75 / N
        elif self.weight_init == 'equal':
            weight = 1./N
        elif self.weight_init == 'increasing':
            weight = 1.25 / N
        return weight

    def _update_weight_0(self):
        '''
        Calculate weight coefficient with heuristics: decreasing, equal or increasing weights
        '''
        N = self.params.shape[0]
        if N == 1:
            return torch.tensor([1.])
        
        new_weight = self._init_one_weight(N)
        return torch.cat([self.weights * (1 - new_weight), torch.tensor([new_weight])])

    def _update_weight_1(self):
        '''
        Update weight coefficient for the new component only, using stochastic gradient descent - Guo et al., 2016
        In our tests, we found that this update method is sometimes numerically unstable
        and requires additional forward evaluations to perform gradient descent,
        therefore, we don't recommend to use this method for weight update.
        '''
        N = self.params.shape[0]
        if N == 1:
            return torch.tensor([1.])

        with torch.no_grad():
            tolerance = 0.001
            init_step_size = self.lr_w
            new_weight = torch.tensor([self._init_one_weight(N)])

            min_iters = 5
            for i in range(self.niter_weight):
                weights = torch.cat([self.weights * (1. - new_weight), new_weight])
                gradient = self._weight_gradient(self.params, weights, self.nsample_weight)
                step_size = init_step_size / (0.05 * i + 1)
                w = min(max(new_weight - step_size * gradient, torch.tensor([0.0005])), torch.tensor([0.9995]))
                dif = abs(new_weight - w)
                new_weight = w
                if i > min_iters and (dif < tolerance):
                    break
            return torch.cat([self.weights * (1. - new_weight), new_weight])

    def _update_weight_2(self):
        '''
        Update weight coefficient for every component
        In our tests, we found that this update method is sometimes numerically unstable
        and requires additional forward evaluations to perform gradient descent,
        therefore, we don't recommend to use this method for weight update.
        '''
        N = self.params.shape[0]
        if N == 1:
            return torch.tensor([1.])
        
        new_weight = torch.tensor([self._init_one_weight(N)])
        weights = torch.cat([self.weights * (1. - new_weight), new_weight])
        # weights = torch.ones(N)
        # weights /= weights.sum()

        for _ in range(self.niter_weight):
            weights.requires_grad_()
            optimizer = torch.optim.SGD([weights], lr=self.lr_w, weight_decay = 0.)
            optimizer.zero_grad()
            self._kl_estimate(self.params, weights, self.nsample_weight).backward()
            optimizer.step()

            weights = projection_on_simplex(weights.detach())
        return weights

    def _weight_gradient(self, params, weights, nsamples):
        '''
        TODO: need to recheck this function again
        '''
        out = 0.
        for k in range(weights.shape[0]):
            samples = self.component_dist.generate_samples_for_one_component(params[k, :], nsamples)
            if self.constrained:
                samples, _ = self.component_dist.log_pdf(params[k, :], samples)
                _, lg = self.component_dist.log_pdf_prev(params, samples)
            else:
                lg = self.component_dist.log_pdf(params, samples)
            if len(lg.shape) == 1:
                lg = lg[:,None]
             
            lg = torch.logsumexp(lg[:, weights > 0] + torch.log(weights[weights > 0]), dim=1)

            if self.constrained:
                log_det = 0
            else:
                samples, log_det = self.component_dist.real_2_const(samples)
            lf = self.logp(samples)
            # out += weights[k] * (lg.mean() - lf.mean())

            if k < (weights.shape[0] - 1):
                out += self.weights[k] * ((lg - log_det).mean() - lf.mean())
            else:
                out = (lg - log_det).mean() - lf.mean() - out
        return out

    def _kl_estimate(self):
        # TODO update his function to perform _update_weight_2
        pass

    def _compute_weights(self):
        if self.weight_cal == 0:
            return self._update_weight_0()
        elif self.weight_cal == 1:
            return self._update_weight_1()
        elif self.weight_cal == 2:
            return self._update_weight_2()
        else:
            raise Exception("Invalid weight update method")

    def _get_mixture(self):
        # get the unflattened params and weights
        output = self.component_dist.unflatten(self.params)
        output.update([('weights', self.weights)])
        return output

    def _objective(self, param, itr, nsamples):
        '''
        Objective function for training each component
        Negative ELBO for the first component
        Negative RELBO for later components
        Input:
            param: parameters for current component to be trained
            itr: iteration number for components, used to calculate regularisation value lmb
            nsamples: number of samples per iteration to calculate the objective function
        Return:
            ELBO for the first component or RELBO for later components
        '''
        # Draw samples from base distribution
        samples = self.component_dist.sample_from_base(nsamples)
        # First, forward pass of component distribution to get model samples and its probability value log_gt
        samples, log_gt = self.component_dist.log_prob_gt(param, samples)
        # Second, backward pass model samples through obtained components to get log_qt-1
        if self.weights.shape[0] > 0:
            # 1. backward pass to get log_gi i \in [0, t-1]
            _, log_gi = self.component_dist.log_prob_qt_minus_one(self.params, samples)
            # 2. mix each component to get log_qt-1 = logsumexp(log(wi) + log(gi))
            if len(log_gi.shape) == 1:
                # Need to add a dim such that first dim corresponds to samples, and second dim to components
                log_gi = log_gi[:, None]
            log_qt_minus_1 = log_gi[:, self.weights > 0] + torch.log(self.weights[self.weights > 0])
            log_qt_minus_1 = torch.logsumexp(log_qt_minus_1, dim = -1)
        else:
            log_qt_minus_1 = 0.
        
        # Third, calculate log posterior probability values for model samples by performing forward evaluation
        logp = self.log_posterior(samples)
        return - torch.mean(logp - log_qt_minus_1 - self.lmb(itr) * log_gt)

    def _initialize(self, itr, nsamples = 100, n_init = 1):
        '''
        Randomly initialize component parameters and evaluate negative (R)ELBO value
        Retrive the best parameter set which provides the minimal negative (R)ELBO value 
        Note that such an initialize requires many additional forward evaulations
        therefore, we don't suggest to use this method in high-D seismic inversion problems
        Alternatively, this method might be used for problems where dimensionality is low or where forward evaluation is cheap,
        to provide good initial values for every new component
        This function also provides fast initialize method by calling component_dist.component_init directly
        To do this, you need to set n_init = 1
        Input:
            itr: which component distribution being initialised
            nsamples: number of samples used to perform Monte Carlo integration to estimate negative (R)ELBO
            n_init: number of random initialisation; if = 1, then no additional forward evaluation is performed
        Output:
            best_param: obtained initialize values for new component
        '''
        if n_init == 1:
            best_param = self.component_dist.component_init(self.params, self.weights)
        else:
            best_param = None
            best_objective = float('inf')
            # try initializing several times
            for n in range(n_init):
                current_param = self.component_dist.component_init(self.params, self.weights)
                current_objective = self._objective(current_param, itr, nsamples)
                if current_objective < best_objective or best_param is None:
                    best_param = current_param
                    best_objective = current_objective
            if best_param is None:
                # if every single initialization had an infinite objective, just raise an error
                raise ValueError
        return best_param

    def _update_component_param(self, param, component, optimizer, n_out = 1, n_iter = 1000, nsample = 10, verbose = False):
        '''
        Update parameter for one component distribution
        Input:
            param: parameters need to be uupdated
            optimizer: torch.optim object used to perform optimization
            component: which component being updated / component ID
            n_out: number of outputing intermediate training results for quality control
        Return:
            loss: loss function value for each iteration
            Note that param is updated during iterations thus no need to explicit return
        '''
        loss = []
        output_interval = math.ceil(n_iter / n_out)
        start = time.time()
        for i in range(n_iter):
            optimizer.zero_grad()
            negative_obj = self._objective(param, component, nsample)
            negative_obj.backward()
            optimizer.step()
            loss.append(negative_obj.data.numpy())

            if (i % output_interval == 0 or i == n_iter - 1) and verbose:
                iteration = i if i < n_iter - 1 else n_iter
                end = time.time()
                print(f'Iteration: {iteration:>5d}, \tLoss: {negative_obj.data.numpy():>5f}')
                print('The elapsed time is: ', end-start)          
        return loss
            
            
    def update(self, ncomponent, n_init = 1, n_iter = 1000, nsample = 10, n_out = 1, 
                optimizer = 'torch.optim.Adam', lr = 0.001, verbose = False):
        '''
        Update/build BVI up to n components
        '''
        # if self.N != 0:
        #     self._load_previous_results()
        for i_comp in range(self.N, ncomponent):
            # First, initialize the next component:
            x0 = self._initialize(i_comp, n_init = n_init)
            # if this is the first component, set the dimension of self.params
            if self.params.size == 0:
                self.params = torch.empty((0, x0.shape[0]))

            # Second, build (train) the next component
            if verbose:
                print('----------------------------------------\n')
                print("Optimizing component " + str(i_comp + 1) + "... \n")
            current_param = x0.detach().requires_grad_()
            optim = eval(optimizer)([current_param], lr = lr)
            loss = self._update_component_param(current_param, i_comp, optim, n_out = 1, n_iter = n_iter,
                                nsample = nsample, verbose = verbose)

            new_param = current_param.detach()
            if verbose:
                print("Optimization of component " + str(i + 1) + " complete\n")
            
            if self.params.shape[0] == 0:       # first component
                self.params = new_param[None]   # make sure self.params is a 2-dim tensor
            else:
                self.params = torch.cat((self.params, new_param[None]), 0)
        
            # compute the new weights and add to the list
            if verbose and self.weight_cal != 0:
                print('Updating weights...')
            self.weights_prev = self.weights.clone()
            self.weights = self._compute_weights()

            if verbose and self.weight_method != 0:
                print('Weight update complete...')
            
            if verbose:
                output = self._get_mixture()                
                weights = output['weights'].detach().numpy()
                mus = output['mus'].detach().numpy()
                covariances = output['covariances'].detach().numpy()
                np.savetxt(f'./output/BVI_weights.txt', weights)
                np.savetxt(f'./output/BVI_mus.txt', mus)
                np.savetxt(f'./output/BVI_covariances.txt', covariances)
                np.savetxt(f'./output/BVI_loss_component{i_comp}.txt', loss)


        # update self.N to the new # components
        self.N = N

        output = self._get_mixture()
        return output
