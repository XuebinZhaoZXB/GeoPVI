import torch
import torch.nn as nn


class VariationalModel(nn.Module):
    '''
    A class that creates a variational model that uses a set of invertible transforms (normalising flows)
    inherited from torch.nn.Module
    This is used to train and sample from the variational pdf
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

    # def update(self, iteration, optimiser, nsample = 1, start_ite = 0, noutput = 1, verbose = True):
    #     for i in range(start_ite, iterations):
    #         optimizer.zero_grad()
    #         # model.train()
    #         x = torch.as_tensor(gen_sample(nsample, ndim, para1 = lower, para2 = upper, ini=args.ini_dist))
    #         z, log_det = model(x)
    #         logp = posterior.log_prob(z)

    #         loss = -torch.mean(logp + log_det) # mean: Expectation term using Monte Carlo
    #         loss.backward()
    #         optimizer.step()
    #         loss_his.append(loss.data.numpy())

