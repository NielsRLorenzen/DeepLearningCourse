# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:07:26 2023

@author: Niels
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Binomial
#%%
#RadialFlow class taken from https://github.com/fmu2/flow-VAE
class RadialFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of radial flow.
        Reference:
        Variational Inference with Normalizing Flows
        Danilo Jimenez Rezende, Shakir Mohamed
        (https://arxiv.org/abs/1505.05770)
        Args:
            dim: input dimensionality.
        """
        super(RadialFlow, self).__init__()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        def m(x):
            return F.softplus(x)
        def h(r):
            return 1. / (a + r)
        def h_prime(r):
            return -h(r)**2

        a = torch.exp(self.a)
        b = -a + m(self.b)
        r = (x - self.c).norm(dim=1, keepdim=True)
        tmp = b * h(r)
        x = x + tmp * (x - self.c)
        log_det = (self.d - 1) * torch.log(1. + tmp) + torch.log(1. + tmp + b * h_prime(r) * r)

        return x, log_det

class Flow(nn.Module):
    def __init__(self, dim, type, length):
        """Instantiates a chain of flows.
        Args:
            dim: input dimensionality.
            type: type of flow.
            length: length of flow.
        """
        super(Flow, self).__init__()
        
        if type == 'radial':
            self.flow = nn.ModuleList([RadialFlow(dim) for _ in range(length)])
        else:
            self.flow = nn.ModuleList([])

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        [B, _] = list(x.size())
        log_det = torch.zeros(B, 1).to(x.device)
        for i in range(len(self.flow)):
            x, inc = self.flow[i](x)
            log_det = log_det + inc

        return x, log_det
#%%
class DenseLayer(nn.Module):
    def __init__(self, n_in,n_out,batch_norm = False):
        '''Instantiate a fully connected layer.
        '''
        super(DenseLayer,self).__init__()
        self.Layer = nn.Sequential(nn.Linear(n_in,n_out),nn.ReLU())
        if batch_norm:
            self.Layer.append(nn.BatchNorm1d(n_out))
            
    def forward(self,x):
        return self.Layer(x)
        
class Encoder(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, width = 'decaying',batch_norm = False, ncond = 0):
        '''Instantiates an encoder that only handles flat inputs.
        
        Params:
            n_in:(int) Input length
            n_out:(int) Number of output dimensions
            n_hidden:(int) Number of hidden layers 
            width: Specifies the width of hidden layers. 
                   Accepted values are 'decaying'(default),
                   an int or list of ints.
                   'decaying' specifies the size of hidden 
                   layer h to be n_in*2^-(h).
                   Passing an int will make all hidden layers
                   that width.
                   A list of ints (must have length n_hidden) 
                   will make the hidden layers with the 
                   specified sizes.
            batch_norm: (bool) If False (default) layers 
                        are not batch normalized.
        '''
        super(Encoder,self).__init__()
        
        layer_widths = [n_in+ncond]
        if width == 'decaying':
            for i in range(n_hidden):
                layer_widths.append(layer_widths[-1]//2)
                
        elif type(width) == int:
            hidden_width = [width for _ in range(n_hidden)]
            layer_widths = layer_widths.extend(hidden_width)
            
        elif type(width) == list:
            assert len(width) == n_hidden, 'The len of the width list must be same as n_hidden'
            layer_widths.extend(width)
        
        self.encode = nn.Sequential()
        for i in range(len(layer_widths)-1):
            self.encode.append(DenseLayer(layer_widths[i],
                                          layer_widths[i+1],
                                          batch_norm=batch_norm))
            
        self.mean_layer = nn.Linear(layer_widths[-1],n_out)
        self.logvar_layer = nn.Linear(layer_widths[-1],n_out)
            
    def forward(self,x,y=None):
        if y is not None:
            x = torch.cat([x,y])
        x = self.encode(x)
        return self.mean_layer(x), self.logvar_layer(x)
    
class Decoder(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, width = 'increasing',batch_norm = False, ncond = 0):
        '''Instantiates a decoder.
        
        Params:
            n_in:(int) Input length
            n_out:(int) Number of output dimensions
            n_hidden:(int) Number of hidden layers 
            width: Specifies the width of hidden layers. 
                   Accepted values are 'Increasing'(default),
                   an int or list of ints.
                   'increasing' specifies the size of hidden 
                   layer h to be n_in*2^(h).
                   Passing an int will make all hidden layers
                   that width.
                   A list of ints (must have length n_hidden) 
                   will make the hidden layers with the 
                   specified sizes.
            batch_norm: (bool) If False (default) layers 
                        are not batch normalized.
        '''
        super(Decoder,self).__init__()
        
        layer_widths = [n_in+ncond]
        if width == 'increasing':
            for i in range(n_hidden):
                layer_widths.append(layer_widths[-1]*2)
                
        elif type(width) == int:
            hidden_width = [width for _ in range(n_hidden)]
            layer_widths = layer_widths.extend(hidden_width)
            layer_widths.append(n_out)
            
        elif type(width) == list:
            assert len(width) == n_hidden, 'The len of the width list must be same as n_hidden'
            layer_widths.extend(width)
            layer_widths.append(n_out)
            
        self.decode = nn.Sequential()
        for i in range(len(layer_widths)-1):
            self.decode.append(DenseLayer(layer_widths[i],
                                          layer_widths[i+1],
                                          batch_norm=batch_norm))
                        
    def forward(self,z,y=None):
        if y is not None:
            z = torch.cat([z,y])
        return self.decode(z)
            
class VariationalAutoencoder(nn.Module):
    def __init__(self, n_in, n_latent, n_hidden, 
                 width = ('decaying','increasing'),
                 batch_norm = True,flow = False, 
                 nconditions = 0, init_weights = True):
        '''Instatiates a Variational autoencoder. This class provides easy 
        network shape flexibility through the width argument of the 
        Decoder and Encoder classes.
        
        Params:
            n_in(int): Number of input features.
            n_latent(int): Number of latent features.
            n_hidden(int/list-like): If int the encoder and decoder
                                     will both have that number
                                     of hidden layers. If list
                                     index 0 is the hidden layers
                                     of the encoder and 1 for
                                     the decoder.
            width(tuple): A two-tuple of width arguments 
                          for the encoder and decoder,
                          respectively.
            batch_norm(bool):If True (default) all dense 
                            layers in the encoder and decoder 
                            are batch normalized.
            flow(bool): Whether or not to use normalizing flow
            conditions(int): Number of conditions
            init_weights(bool): Wheter to init all weights with 
                                the torch.nn.init.xavier_normal_()
                                function
        '''
        super(VariationalAutoencoder,self).__init__()
        
        self.encoder = Encoder(n_in,n_latent,n_hidden, width = width[0], batch_norm=batch_norm, ncond = nconditions)
        if flow:    
            self.flow = Flow(n_latent,'radial',5)
        self.decoder = Decoder(n_latent,n_in,n_hidden, width = width[1], batch_norm=batch_norm, ncond = nconditions)
        
        if init_weights:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    torch.nn.init.constant_(m.bias,0)

            self.encoder.apply(init_weights)
            self.decoder.apply(init_weights)

        
    def sample_posterior(self, mean, logvar):
        '''Sample the distribution q(z|x)'''
        eps = torch.randn(mean.shape).to(mean.device)
        sigma = torch.exp(0.5*logvar)
        return mean + sigma*eps
    
    def generate(self, n, y=None):
        '''Generate samples from the prior.'''
        z = torch.randn(n,self.n_latent)
        if y is not None:
            z = torch.cat([z,y],dim=1)
        px_logits = self.decoder(z)
        px = Binomial(total_count=2,logits=px_logits, validate_args=False)
        return px
    
    def forward(self, x, y=None,beta=1):
        if y is not None:
            mean, logvar = self.encoder(torch.cat([x,y],dim=1))
        else:
            mean, logvar = self.encoder(x)
 
        z = self.sample_posterior(mean,logvar)
                
        kl = -0.5 * torch.sum(1. + logvar - mean.pow(2) - logvar.exp(), dim=1, keepdim=True)
    
        
        if hasattr(self,'flow'):
            z,log_det = self.flow(z)
            kl = kl-log_det
        
        if y is not None:
            px_logits = self.decoder(torch.cat([z,y],dim=1))
        else:
            px_logits = self.decoder(z)
        
        px = Binomial(total_count=2,logits=px_logits, validate_args=False)
        
        log_px = torch.nansum(px.log_prob(x))
      
        elbo = log_px - beta*kl.sum()

        loss = -elbo

        outputs = {'px':px, 
                    'z':z, 
                    'mean':mean, 
                    'logvar':logvar} 
        diagnostics = {'elbo':elbo,
                       'log_px':log_px,
                       'kl':kl.sum()}
        return loss, outputs, diagnostics
#%%
from torch.nn import Softmax                  
class Classifier(nn.Module):
    def __init__(self, n_in, n_hidden, widths, n_classes, 
                 batch_norm = True):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(DenseLayer(n_in, widths[0]))
        widths.append(n_classes)
        for i in range(n_hidden):
            self.net.append(DenseLayer(widths[i], 
                                       widths[i+1],
                                       batch_norm=batch_norm))
    def forward(self,x):
        x = self.net(x)
        print(x.size())
        return Softmax(x)