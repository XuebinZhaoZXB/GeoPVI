import torch
import time
import numpy as np

class BVI(object):
    def __init__(
        self, component_dist, start = 0, postfix = 0, n_init = 10, init_inflation = 10, n_sample_init = 10,
        lr = 0.001, num_opt_steps = 5000, verbose = False):

        self.N = start # current num of components
        self.postfix = postfix
        self.component_dist = component_dist # component distribution object
        self.weights = torch.empty(0) # weights
        self.params = torch.empty((0, 0)) # components' parameters
        self.error = float('inf') # error for the current mixture
        self.n_init = n_init # number of initializations for each component
        self.init_inflation = init_inflation # noise multiplier for initializations
        self.lr = lr
        self.num_opt_steps = num_opt_steps
        self.verbose = verbose
        self.n_sample_init = n_sample_init

    def build(self, N):
        # build the approximation up to N components
        if self.N != 0:
            self._load_previous_results()
        for i in range(self.N, N):
            # initialize the next component
            # x0 = self._initialize(i, self.n_sample_init)
            x0 = self.component_dist.params_init(self.params, self.weights, self.init_inflation)
            # if this is the first component, set the dimension of self.params
            if self.params.size == 0:
                self.params = torch.empty((0, x0.shape[0]))
            
            # build the next component
            if self.verbose:
                print("Optimizing component " + str(i + 1) + "... ")
            current_param = x0.detach().requires_grad_()
            optimizer = torch.optim.Adam([current_param], lr=self.lr)
            elbo = np.zeros(self.num_opt_steps)
            
            # Staring training component
            start = time.time()
            for j in range(self.num_opt_steps):
                optimizer.zero_grad()
                loss = self._objective(current_param, i, self.n_samples)
                loss.backward()
                optimizer.step()
                elbo[j] = loss.data.numpy()
                
                if self.verbose and (j % (self.num_opt_steps // 5) == 0 or j == self.num_opt_steps - 1):
                    print(f'Iteration: {j:>5d}, Loss: {elbo[j]:>8f}')
                    # np.savetxt(f'./output/BVI_parameters_comp{i}_ite{j}_{self.postfix}.txt', current_param.detach().numpy())
                    # np.savetxt(f'./output/BVI_loss_component{i}_{self.postfix}.txt', elbo[:j])
                    end = time.time()
                    print('The elapsed time is: ', end-start)

            new_param = current_param.detach()
            if self.verbose:
                print("Optimization of component " + str(i + 1) + " complete")

            # add it to the matrix of flattened parameters
            if self.params.shape[0] == 0:
                self.params = new_param[None]
            else:
                self.params = torch.cat((self.params, new_param[None]), 0)

            # compute the new weights and add to the list
            if self.verbose and self.weight_method != 0:
                print('Updating weights...')
            self.weights_prev = self.weights.clone()
            # try:
            self.weights = self._compute_weights()

            if self.verbose and self.weight_method != 0:
                print('Weight update complete...')

            # estimate current error
            error_str, self.error = self._error()

            # print out the current error
            print('Component ' + str(self.params.shape[0]) + ':')
            print(error_str +': ' + str(self.error.data.numpy()) + '\n')
            if self.verbose:
                results = self._get_mixture()
                weights = results['weights'].detach().numpy()
                mus = results['mus'].detach().numpy()
                covariances = results['covariances'].detach().numpy()
                np.savetxt(f'./output/BVI_weights_{self.postfix}.txt', weights)
                np.savetxt(f'./output/BVI_mus_{self.postfix}.txt', mus)
                np.savetxt(f'./output/BVI_covariances_{self.postfix}.txt', covariances)
                np.savetxt(f'./output/BVI_loss_component{i}_{self.postfix}.txt', elbo)
            
        # update self.N to the new # components
        self.N = N

        output = self._get_mixture()
        output['obj'] = self.error
        return output
        
    def _initialize(self, itr, n_samples):
        best_param = None
        best_objective = float('inf')

        # try initializing n_init times
        for n in range(self.n_init):
            current_param = self.component_dist.params_init(
                self.params, self.weights, self.init_inflation)
            best_param = current_param
        #     current_objective = self._objective(current_param, itr, n_samples)
            
        #     if current_objective < best_objective or best_param is None:
        #         best_param = current_param
        #         best_objective = current_objective
        #     if self.verbose:
        #         if (n == 0 or n == self.n_init - 1):
        #             if n == 0:
        #                 print("{:^30}|{:^30}".format('Iteration', 'Best objective'))
        #             print("{:^30}|{:^30}".format(n, str(best_objective.data.numpy())))
        # if best_param is None:
        #     # if every single initialization had an infinite objective, just raise an error
        #     raise ValueError
        
        # return the initialized result
        return best_param
    
    def _load_previous_results(self):
        print(f'Load from previous results at iteration {self.N}\n')
        PATH = './output/60_100/bvi/bvi_equal_w_infla_3/'
        postfix = self.postfix
        weights = np.loadtxt(PATH + 'BVI_weights_' + str(postfix) + '.txt')
        mus = np.loadtxt(PATH + 'BVI_mus_' + str(postfix) + '.txt')
        covariances = np.loadtxt(PATH + 'BVI_covariances_' + str(postfix) + '.txt')
        
        if weights.size == 1:
            self.weights = torch.tensor([1.])
            self.params = torch.from_numpy(np.hstack((mus[None], covariances[None])))
        else:
            self.weights = torch.from_numpy(weights[:self.N] / weights[:self.N].sum()) # weights
            self.params = torch.from_numpy(np.hstack((mus[:self.N], covariances[:self.N])))
    
    def get_weights(self):
        return self.weights

    def _compute_weights(self):
        raise NotImplementedError
        
    def _objective(self, itr):
        raise NotImplementedError
        
    def _error(self):
        raise NotImplementedError
  
    def _get_mixture(self):
        raise NotImplementedError