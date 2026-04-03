import math
import time
import numpy as np 
import torch
import torch.nn as nn


class SteinMixture():
    '''
    A class to build a Stein Mixture Variational Inference (SMVI) model,
    used to approximate the posterior distribution by a mixture of components
    Currently, the main purpose of this class is used to evaulate the gradient of mixture ELBO for each component, 
    the training (updating) of each component is performed using VIP package
    Future update will include the training of each component within this class using SVGD update rule
    '''
    def __init__(
        self, componentDistribution, log_posterior, num_component = 5, elbo_type = 1
    ):
        self.log_posterior = log_posterior
        self.component = componentDistribution
        self.num_component = num_component
        self.elbo_type = elbo_type

    def grad_elbo(self, param_numpy, nsamples = 10, chunks = None):
        '''
        Calculate the gradient of ELBO for the mixture distribution
        This function is designed such that it can be called by VIP package to perform SVGD update rule
        Input:
            param_numpy: numpy parameters for each component distribution, with shape (num_component, num_parameters)
                            num_parameters is determined by the component distribution
            nsamples: number of samples to use for Monte Carlo integration to estimate the ELBO value and its gradient
        Return:
            elbo: a vector of ELBO values, each element corresponds to the ELBO value for each component distribution
            gradient: each row contains the gradient for each compoment parameter, with shape (num_component, num_parameters)
            mask: an auxilary mask array for SVGD optimization, can be safely ignored - here return None
        '''
        # Convert param from numpy (reguired by VIP) to torch tensor and set requires_grad = True for AD
        param_torch = torch.from_numpy(param_numpy).detach().requires_grad_(True)
        # Calculate the ELBO value by calling the self._elbo function
        # which performs forward evaluation and Monte Carlo integration
        if self.elbo_type == 1:
            elbo = self._elbo(param_torch, nsamples, chunks)
        elif self.elbo_type == 2:
            elbo = self._elbo2(param_torch, nsamples, chunks)
        elbo.sum().backward()
        gradient = param_torch.grad.numpy().copy()

        return elbo.detach().numpy().copy(), gradient, None

    def _elbo2(self, param, nsamples, chunks):
        '''
        Calculate the ELBO value for the mixture distribution
        The difference between this function and self._elbo is that this function calculate the ELBO value using 
        different sampeles from each component distribution
        Input:
            param: parameters for each component distribution, with shape (num_component, num_parameters)
            nsamples: number of samples to use for Monte Carlo integration to estimate the ELBO value
            chunks: chunk size used to calculate mixture density for all samples, 
                    if None, then calculate all samples together; otherwise, calculate in chunks to save memory
        Return:
            elbo: the ELBO value for each component distribution, with shape (num_component,)
        '''
        # First, draw samples from base distribution
        samples = self.component.sample_from_base(nsamples * self.num_component)
        samples = samples.reshape(self.num_component, nsamples, self.component.dim)
        # Second, forward pass of component distribution to get model samples and its probability value log_gt
        samples, logq, logdet_real_2_const = self.component.log_prob2(param, samples, chunks)
        # samples: ncomponents * nsamples * dim, logq: ncomponents * nsamples, logdet_real_2_const: ncomponents * nsamples

        # Third, compute log posterior probability values for model samples by performing forward evaluation
        logp = self.log_posterior(samples.reshape(self.num_component * nsamples, self.component.dim))
        logp = logp.reshape(self.num_component, nsamples) + logdet_real_2_const
        elbo = (logp - logq).mean(dim = -1)

        return elbo

    def _elbo(self, param, nsamples, chunks):
        '''
        Calculate the ELBO value for the mixture distribution
        Input:
            param: parameters for each component distribution, with shape (num_component, num_parameters)
            nsamples: number of samples to use for Monte Carlo integration to estimate the ELBO value
            chunks: chunk size used to calculate mixture density for all samples, 
                    if None, then calculate all samples together; otherwise, calculate in chunks to save memory
        Return:
            elbo: the ELBO value for each component distribution, with shape (num_component,)
        '''
        # First, draw samples from base distribution
        samples = self.component.sample_from_base(nsamples * self.num_component)
        samples = samples.reshape(self.num_component, nsamples, self.component.dim)
        # Second, forward pass of component distribution to get model samples and its probability value log_gt
        samples, logq = self.component.log_prob(param, samples, chunks)
        # samples: ncomponents * nsamples * dim, logq: ncomponents * nsamples

        # Third, compute log posterior probability values for model samples by performing forward evaluation
        logp = self.log_posterior(samples.reshape(self.num_component * nsamples, self.component.dim))
        logp = logp.reshape(self.num_component, nsamples)
        elbo = (logp - logq).mean(dim = -1)

        return elbo

    def _get_mixture(self, loss = None):
        # get the unflattened params and weights
        output = self.component_dist.unflatten(self.params)
        output.update([('weights', self.weights)])
        if loss is not None:
            output.update([('losses', loss)])
        return output
    
    def sample(self, nsamples, component = -1):        
        '''
        Sample from the obtained mixture models or specific components
        Argument
            nsmaples:   total number of samples to be drawn
            component:  indicator stating samples are drawn from which component
                        int or np.ndarray or list
                        if int: -1 means sample from all components - that is sample from the BVI model
                                otherwise should be in [0, self.N) means which component to be sampled 
                                (0 means the 1st, 1 means 2nd, and so on)
                        if np.ndarray / list: sample from listed components
        Return
            Posterior samples
        '''
        if isinstance(component, np.ndarray) or isinstance(component, list):
            component_index = component
        elif isinstance(component, int):
            if component == -1:
                component_index = np.arange(self.N)
            elif 0 <= component < self.N:
                component_index = [component]
            else:
                raise ValueError('Component should be a numpy.ndarray with integer elements or an integer from -1 to self.N - 1')
        else:
            raise ValueError('Component should be a numpy.ndarray with integer elements or an integer from -1 to self.N - 1')

        samples = np.empty([0, self.component_dist.dim])
        probs = (self.weights[component_index] / self.weights[component_index].sum()).numpy()
        k = np.random.choice(component_index, size = nsamples, p = probs)
        for i in component_index:
            nsamples_i = (k == i).sum()
            samples_i = self.component_dist.sample_from_base(nsamples_i)
            samples_i, _ = self.component_dist.log_prob_gt(self.params[i], samples_i)
            samples = np.vstack([samples, samples_i.data.numpy()])
        return samples
