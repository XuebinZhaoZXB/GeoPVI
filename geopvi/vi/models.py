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
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z = x
        return z, log_det

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

    # def update(self, log_prob, optimizer, n_iter = 1000, nsample = 4, start_ite = 0, noutput = 1, verbose = True):
    #     start = time.time()
    #     for i in range(start_ite, n_iter):
    #         optimizer.zero_grad()
    #         # model.train()
    #         x = torch.as_tensor(gen_sample(nsample, ndim, para1 = lower, para2 = upper, ini=args.ini_dist))
    #         z, log_det = self.forward(x)
    #         logp = posterior.log_prob(z)

    #         loss = -torch.mean(logp + log_det) # mean: Expectation term using Monte Carlo
    #         loss.backward()
    #         optimizer.step()
    #         loss_his.append(loss.data.numpy())


class BayesianInversion():
    '''
    A class that defnies the objective function in variational inference, i.e., maximising ELBO or minimising KL
    '''
    def __init__(self, variational, log_posterior):
        '''
        variational: a class that defines the variational distribution (variational model)
        log_post: a function that calculates posterior unnormalised probability value for any given model sample
        '''
        self.log_posterior = log_posterior
        self.variational = variational

    def update(self, optimizer = None, lr = 0.001, n_iter = 1000, nsample = 4, start_ite = 0, n_out = 1, verbose = True):
        '''
        Update variational model by optimising the variational objective function
        Input
            optimizer: torch.optim used to perform optimization; default is torch.optim.Adam
            lr: learning rate, default is 0.001
            n_iter: number of iterations
            nsample: number of samples per iteration to perform Monte Carlo integration
            n_out: number of outputing intermediate training results for quality control
            verbose: whether print intermediate training or not
        Return
            loss: mean loss value for each iterations, vector of length n
        '''

        loss = []
        output_interval = math.ceil(n_iter / n_out)
        
        start = time.time()
        for i in range(n_iter):
            # model.train()
            # x = torch.as_tensor(gen_sample(args.nsample, ndim, para1 = lower, para2 = upper, ini=args.ini_dist))

            z, logq = self.variational(nsample)
            logp = self.log_posterior(z)

            negative_elbo = -torch.mean(logp - logq) # mean: Expectation term using Monte Carlo
            optimizer.zero_grad()
            negative_elbo.backward()
            optimizer.step()
            loss.append(negative_elbo.data.numpy())

            if i % output_interval == 0 and verbose:
                # Save intermediate model parameters
                param = get_flow_param(model.flows[-2])
                name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_ite{i}_parameter.npy')
                np.save(name, param)

                name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_loss.txt')
                np.savetxt(name, loss_his)

                # # If you want to get posterior samples and save them, you can use the following:
                # x = torch.as_tensor(gen_sample(2000, ndim, para1 = lower, para2 = upper, ini=args.ini_dist))
                # z = model.sample(x)
                # z = z.data.numpy()
                # name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_ite{i}_sample.npy')
                # np.save(name, z)

                # save intermediate normalising flows model
                if args.save_intermediate_result:
                    name = os.path.join(args.basepath, args.outdir, f'{args.flow}_{args.kernel}_model.pt')
                    torch.save({
                                'iteration': i,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_his,
                                }, name)
                    
                print(f'Iteration: {i:>5d},\tLoss: {loss.data:>10.2f}')
                end = time.time()
                print(f'The elapsed time is: {end-start:.2f} s')

        print(f'Iteration: {n_iter:>5d},\tLoss: {negative_elbo.data:>10.2f}')
        end = time.time()
        print(f'The elapsed time is: {end-start:.2f} s')
        print('----------------------------------------\n')        

        return loss