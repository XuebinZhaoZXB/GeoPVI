import math
import time
import numpy as np 
import torch
import torch.nn as nn


class VariationalModel(nn.Module):
    '''
    A class that creates a variational model that uses a set of invertible transforms (normalising flows)
    inherited from torch.nn.Module
    This is used to build and sample from the variational pdf
    '''
    def __init__(self, flows, base_dist = 'Normal'):
        '''
        flows: a list that defines a flow model to construct a parmetric variational distribution
        base_dist: base (initial) distribution. Currently support Standard Normal N(0,1) and Uniform U(0,1)
        Future release might support other types of base distribution
        '''
        super().__init__()
        self.flows = nn.ModuleList(flows)
        if base_dist == 'Uniform' or base_dist == 'Normal':
            self.base_dist = base_dist
        else:
            raise NotImplementedError("Base distribution provided not currently supported")
        for flow in flows:
            if hasattr(flow, 'dim'):
                self.dim = flow.dim
                break
    
    def _log_prob_base(self, x):
        if self.base_dist == 'Normal':
            return -0.5 * self.dim * np.log(2*np.pi) - 0.5 * (x**2).sum(axis = 1)
        elif self.base_dist = 'Uniform':
            return torch.zeros(x.shape[0])

    def sample_from_base(self, nsamples):
        if self.base_dist == 'Normal':
            return torch.randn(nsamples, self.dim)
        elif self.base_dist = 'Uniform':
            return torch.rand(nsamples, self.dim)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        logq0 = self._log_prob_base(x)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        return x, logq0 - log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det
 
    def sample(self, x):
        for flow in self.flows:
            x, _ = flow.forward(x, train = False)
        return x


class VariationalInversion():
    '''
    A class that performs variational inversion by maximising ELBO 
    or minimising KL divergence between variationa and posterior distributions
    '''
    def __init__(self, variationalModel, log_posterior):
        '''
        variationalModel: a class that defines the variational distribution (variational model)
        log_post: a function that calculates unnormalised posterior probability value for any given model sample
        '''
        self.log_posterior = log_posterior
        self.variationalModel = variationalModel

    def update(self, optimizer = None, lr = 0.001, n_iter = 1000, nsample = 1, n_out = 1, 
                    verbose = True, save_intermediate_result = False):
        '''
        Update variational model by optimising the variational objective function
        Input
            optimizer: torch.optim used to perform optimization; default is torch.optim.Adam
            lr: learning rate, default is 0.001
            n_iter: number of iterations
            nsample: number of samples per iteration to perform Monte Carlo integration
            n_out: number of outputing intermediate training results for quality control
            verbose: whether print intermediate training process or not
        Return
            loss: average loss value for each iteration, vector of length n
        '''

        loss = []
        output_interval = math.ceil(n_iter / n_out)
        # if no optimizer is provided, default is Adam optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(self.variationalModel.parameters(), lr = lr)
        if verbose:
            print('----------------------------------------\n')

        start = time.time()
        for i in range(n_iter):
            x = self.variationalModel.sample_from_base(nsample)
            z, logq = self.variationalModel(x)
            logp = self.log_posterior(z)

            negative_elbo = -torch.mean(logp - logq) # mean: Expectation term using Monte Carlo
            optimizer.zero_grad()
            negative_elbo.backward()
            optimizer.step()
            loss.append(negative_elbo.data.numpy())

            if i % output_interval == 0 and verbose:
                print(f'Iteration: {i:>5d},\tLoss: {negative_elbo.data:>10.2f}')
                end = time.time()
                print(f'The elapsed time is: {end-start:.2f} s')

                # Save intermediate model parameters
                if save_intermediate_result:
                    # param = get_flow_param(model.flows[-2])
                    # name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_ite{i}_parameter.npy')
                    # np.save(name, param)
                    
                    np.savetxt('loss.txt', loss)

                    # # If you want to get posterior samples and save them, you can use the following:
                    # x = torch.as_tensor(gen_sample(2000, ndim, para1 = lower, para2 = upper, ini=args.ini_dist))
                    # z = model.sample(x)
                    # z = z.data.numpy()
                    # name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_ite{i}_sample.npy')
                    # np.save(name, z)

                    # save intermediate normalising flows model
                    torch.save({
                                'iteration': i,
                                'model_state_dict': self.variationalModel.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                }, 'model.pt')
                
        if verbose:
            print(f'Iteration: {n_iter:>5d},\tLoss: {negative_elbo.data:>10.2f}')
            end = time.time()
            print(f'The elapsed time is: {end-start:.2f} s')
            print('----------------------------------------\n')        

        return loss