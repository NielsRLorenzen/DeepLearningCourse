# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:54:23 2022

@author: Niels

Distributions for vaes
"""

#%%
import torch
from torch import Tensor
from torch.distributions import Distribution
import math

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        #z = mu(x) + eta * sigma(x)
        z = self.mu + self.sigma * self.sample_epsilon()
        return z
    
    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        var = self.sigma ** 2
        log_sigma = self.sigma.log()
        log_p = -((z - self.mu) ** 2) / (2 * var) - log_sigma - math.log(math.sqrt(2 * math.pi))
        return log_p