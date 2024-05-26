import numpy as np
import torch
from geopvi.boostingVI.bvi import BVI


def projection_on_simplex(x):
    u = torch.sort(x, descending=True)[0]
    indices = torch.arange(1, u.shape[0] + 1)
    rho_nz = u + 1. / indices * (1. - torch.cumsum(u, dim=0)) > 0
    rho = indices[rho_nz].max()
    lmbda = 1. / rho * (1. - u[:rho].sum())
    out = torch.max(
        torch.stack([x + lmbda, torch.zeros_like(x, dtype=torch.float32)]), dim=0).values
    return out / out.sum()


class BBVI(BVI):
    def __init__(
        self, component_dist, logp, lmb = lambda itr : 1, weight_method = 0, weight_init = 0,
        n_samples = 1, n_simplex_iters = 100, n_samples_w = 1, lr_w = 0.01, constrained = False, eps = None, **kw
    ):
        super().__init__(component_dist, **kw)
        self.logp = logp
        self.lmb = lmb
        self.n_samples = n_samples
        self.n_simplex_iters = n_simplex_iters
        self.eps = eps
        self.weight_method = weight_method
        self.weight_init = weight_init
        self.n_samples_w = n_samples_w
        self.lr_w = lr_w
        self.constrained = constrained
    
    def _update_weights_0(self):
        N = self.params.shape[0]
        if N == 1:
            return torch.tensor([1.])
        else:
            if self.weight_init == 0: 
                new_weight = 0.75/(N)       # decreasing initial weight value
                # new_weight = 0.5/(N-1)
            elif self.weight_init == 1:
                new_weight = 1. / N         # equal initial weight value
            elif self.weight_init == 2:
                # new_weight = 2. / (N+1)     # increasing initial weight value
                new_weight = 1.25 / (N)     # increasing initial weight value

            old_weight = self.weights * (1. - new_weight)
        return torch.cat([old_weight, torch.tensor([new_weight])])

    def _update_weights_1(self):
        N = self.params.shape[0]
        if N == 1:
            return torch.tensor([1.])

        with torch.no_grad():
            tolerance = 0.001
            init_step_size = self.lr_w
            if self.weight_init == 0: 
                prev_w = torch.tensor([0.75/(N)])        # decreasing initial weight value
                # new_weight = 0.5/(N-1)
            elif self.weight_init == 1:
                prev_w = torch.tensor([1. / N])          # equal initial weight value
            elif self.weight_init == 2:
                prev_w = torch.tensor([2. / (N+1)])      # increasing initial weight value

            min_iters = 10
            for i in range(self.n_simplex_iters):
                weights = torch.cat([self.weights * (1. - prev_w), prev_w])
                gradient = self._w_gradient(self.params, weights, self.n_samples_w)
                step_size = init_step_size / (0.05 * i + 1)
                w = min(max(prev_w - step_size * gradient, torch.tensor([0.0005])), torch.tensor([0.9995]))
                dif = abs(prev_w - w)
                prev_w = w
                if i > min_iters and (dif < tolerance):
                    break
            # print(i, 1./N, w[0])
            # print()
            return torch.cat([self.weights * (1. - prev_w), prev_w])

    def _update_weights_2(self):
        if self.params.shape[0] == 1:
            return torch.tensor([1.])
        else:
            weights = torch.ones(self.params.shape[0], dtype=torch.float32)
            weights /= weights.sum()

            for _ in range(self.n_simplex_iters):
                weights.requires_grad_()
                optimizer = torch.optim.SGD([weights], lr=self.lr_w, weight_decay = 0.)
                optimizer.zero_grad()
                self._kl_estimate(self.params, weights, self.n_samples_w).backward()
                optimizer.step()

                weights = projection_on_simplex(weights.detach())
        return weights

    def _compute_weights(self):
        if self.weight_method == 0:
            return self._update_weights_0()
        elif self.weight_method == 1:
            return self._update_weights_1()
        elif self.weight_method == 2:
            return self._update_weights_2()
        else:
            raise Exception("Invalid weight update method")

    def _error(self):
        return "KL Divergence", self._kl_estimate(self.params, self.weights, self.n_samples_w)

    def _objective(self, x, itr, n_samples):
        h_samples = self.component_dist.generate_samples_for_one_component(x, n_samples)
        # compute log target density under samples
        if self.constrained:
            h_samples, lh = self.component_dist.log_pdf(x, h_samples)
        else:
            lh = self.component_dist.log_pdf(x, h_samples)
        # compute current log mixture density
        if self.weights.shape[0] > 0:
            if self.constrained:
                _, lg = self.component_dist.log_pdf_prev(self.params, h_samples)
            else:
                lg = self.component_dist.log_pdf(self.params, h_samples)
            if len(lg.shape) == 1:
                # need to add a dimension so that each sample corresponds to a row in lg
                lg = lg[:, None]
            lg = lg[:, self.weights > 0] + torch.log(self.weights[self.weights > 0])
            # if self.eps:
            #     lg = torch.cat((lg, np.log(self.eps) * torch.ones_like((lg.shape[0], 1))), 1)
            lg = torch.logsumexp(lg, dim=1)
        else:
            lg = 0.

        if self.constrained:
            log_det = 0
        else:
            h_samples, log_det = self.component_dist.real_2_const(h_samples)
            
        lf = self.logp(h_samples)
        return torch.mean((lg - log_det*(len(self.weights) > 0)) + self.lmb(itr) * (lh - log_det) - lf)
    
    def _w_gradient(self, params, weights, n_samples):
        out = 0.
        for k in range(weights.shape[0]):
            samples = self.component_dist.generate_samples_for_one_component(params[k, :], n_samples)
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

    def _kl_estimate(self, params, weights, n_samples):
        out = 0.
        for k in range(weights.shape[0]):
            samples = self.component_dist.generate_samples_for_one_component(params[k, :], n_samples)
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
            out += weights[k] * ((lg - log_det).mean() - lf.mean())
        return out
    
    def _print_perf_w(self, itr, x, obj, grd):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'W', 'GradNorm', 'KL'))
        print("{:^30}|{:^30}|{:^30.3f}|{:^30.3f}".format(itr, str(x), np.sqrt((grd**2).sum()), obj))

    def _get_mixture(self):
        #just get the unflattened params and weights; for KL BVI these correspond to mixture components
        output = self.component_dist.unflatten(self.params)
        output.update([('weights', self.weights)])
        return output
