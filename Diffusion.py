# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:42:59 2022

@author: Admin
"""

import torch
from IPython.display import display
from torchvision import transforms

from Unet import Unet

class OptimizerData():
    def __init__(self,
                 optimizer_type : torch.optim,
                 **kwargs):
        self.optimizer_type = optimizer_type
        self.kwargs         = kwargs
        
class SchedulerData():
    def __init__(self,
                 scheduler_type : torch.optim.lr_scheduler,
                 **kwargs):
        self.scheduler_type = scheduler_type
        self.kwargs         = kwargs

class Diffusion(torch.nn.Module):
    def __init__(self,
                 n_channels : int   = 1,        #image infos
                 x_sz       : int   = 28, 
                 y_sz       : int   = 28,
                 att_type   : str   = 'FLAVOR_SDP',
                 m          : int   = 64,
                 redraw_steps : int = 1000,
                 
                 sample_iters : int = 1000,     #diffusion steps
                 
                 t_size     : int   = 256,      
                 device             = 'cuda',
                 b_1        : float = 10e-4,    #beta ranges
                 b_T        : float = 0.02,
                 
                 opt_data   : OptimizerData = OptimizerData(torch.optim.Adam, lr = 1e-4), #training infos
                 sched_data : SchedulerData = SchedulerData(torch.optim.lr_scheduler.ExponentialLR, gamma = 1),
                 loss       : torch.nn      = torch.nn.MSELoss(),
                 batch_size : int   = 1,
                 data_slice : int   = 0,
                 n_iters    : int   = 10,
                 
                 verbose    : int   = 1,
                 
                 dim_mults          = (1, 2, 4, 8),
                 resnet_block_groups= 8
                 ):
        super(Diffusion, self).__init__()
        self.n_channels = n_channels
        self.x_sz      = x_sz
        self.y_sz      = y_sz
        self.att_type  = att_type
        
        self.sample_iters = sample_iters
        self.t_size     = t_size
        self.device     = device
        
        
        self.net        = Unet(
            dim                 = x_sz,
            dim_mults           = dim_mults,
            channels            = n_channels,
            resnet_block_groups = resnet_block_groups,
            time_dim            = t_size,
            att_type            = att_type,
            m                   = m,
            redraw_steps        = redraw_steps,
            device              = device
        )
        
        self.add_module('unet', self.net) #out of safety
        
        self.beta       = torch.linspace(start=b_1, end = b_T, steps = sample_iters).to(self.device)
        
        self.alphas     = 1.0 - self.beta
        self.alphas_sgn = torch.cumprod(self.alphas, dim = 0)
        alphas_sgn_prev = torch.roll(self.alphas_sgn, 1, 0)
        alphas_sgn_prev[0] = 1
        
        self.a_sgn_sqrt       = torch.sqrt(self.alphas_sgn)     #sqrt(alpha sign)
        self.one_m_a_sgn_sqrt = torch.sqrt(1 - self.alphas_sgn) #sqrt(1 - alpha sign)
        
        #on page 3 there is a more complex formula but this works ok. NB: it's sigma and not sigma^2
        # self.sigma      = torch.sqrt(self.beta)
        self.sigma      = torch.sqrt(self.beta*(1. - alphas_sgn_prev)/(1. - self.alphas_sgn))
        
        self.opt_data   = opt_data
        self.loss       = loss
        self.n_iters    = n_iters
         
        self.verbose    = verbose
        
        self.batch_size = batch_size
        self.data_slice = data_slice
        
        self.optimizer  = opt_data.optimizer_type(self.parameters(), **(opt_data.kwargs))
        self.scheduler  = sched_data.scheduler_type(self.optimizer, **(sched_data.kwargs))
    
    def sample(self):
        self.eval()
        
        transform = transforms.ToPILImage()
        x_T = torch.randn(size = (1, self.n_channels, self.x_sz, self.y_sz)).to(self.device)
        
        for t in reversed(range(self.sample_iters)):
            if(t == 0):
                z = torch.zeros_like(x_T).to(self.device)
            else:
                z = torch.randn_like(x_T).to(self.device)
            
            model_weight = self.beta[t]/self.one_m_a_sgn_sqrt[t]
            
            alpha_sqrt = torch.sqrt(self.alphas[t])
            predicted = self.net(x_T, torch.Tensor([t]).to(self.device)).detach()
            x_T = (x_T - model_weight*predicted)/alpha_sqrt + z*self.sigma[t]
            
            if(self.verbose >= 1 and t%100 == 0):
                print(t)
                r = transform(x_T[0,:,:,:])
                display(r)
        
        self.train()
        return x_T[0, :, :, :]
    
    def fit(self, 
            x    : torch.utils.data.Dataset,
            x_vl : torch.utils.data.Dataset = None):
        data_loader = torch.utils.data.DataLoader(dataset = x, 
                                                  batch_size = self.batch_size,
                                                  shuffle = True)
        data_loader_vl = None
        
        if(self.data_slice > 0):
            subset = list(range(0, self.data_slice))
            trainset_1 = torch.utils.data.Subset(data_loader.dataset, subset)
            data_loader = torch.utils.data.DataLoader(dataset = trainset_1, 
                                                      batch_size = self.batch_size,
                                                      shuffle = True)
        
        if(not x_vl is None):
            data_loader_vl = torch.utils.data.DataLoader(dataset = x_vl, 
                                                         batch_size = self.batch_size,
                                                         shuffle = True)
            if(self.data_slice > 0):
                subset = list(range(0, self.data_slice))
                trainset_1 = torch.utils.data.Subset(data_loader_vl.dataset, subset)
                data_loader_vl = torch.utils.data.DataLoader(dataset = trainset_1, 
                                                             batch_size = self.batch_size,
                                                             shuffle = True)
            

        for epoch in range(self.n_iters):
            tr_loss = torch.tensor([0]).type(torch.FloatTensor).to(self.device)
            for i, data in enumerate(data_loader, 0):
                #we don't really care about the labels
                images, _ = data
                images = images.to(self.device)
                
                #batch size:
                b = images.size(0)
                
                #time tensor
                t = torch.randint(low = 0, high = self.sample_iters, size = (b,)).to(self.device)
                
                #noise matrix
                e = torch.randn_like(images).to(self.device)
                
                sqrt_a         = torch.zeros((b, 1, 1, 1)).to(self.device)
                sqrt_1_minus_a = torch.zeros((b, 1, 1, 1)).to(self.device)
                
                for j in range(b):
                    sqrt_a[j, 0, 0, 0]         = self.a_sgn_sqrt[t[j]]
                    sqrt_1_minus_a[j, 0, 0, 0] = self.one_m_a_sgn_sqrt[t[j]]
                
                net_in = sqrt_a*images + sqrt_1_minus_a*e
                
                net_out = self.net(net_in, t)
                    
                loss = self.loss(net_out, e)
                    
                loss_det = loss.clone().detach()
                tr_loss += loss_det
                
                if(self.verbose >= 2):
                    print("epoch ", epoch, " step ", i, " loss = ", loss_det.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            tr_loss /= (len(x)/self.batch_size)
            if(self.verbose >= 1):
                print("epoch ", epoch, " tr loss = ", tr_loss.item())
            
            if(not x_vl is None):
                vl_loss = torch.tensor([0]).type(torch.FloatTensor).to(self.device)
                for i, data in enumerate(data_loader_vl, 0):
                    self.eval()
                    
                    #we don't really care about the labels
                    images, _ = data
                    images = images.to(self.device)
                    
                    #batch size:
                    b = images.size(0)
                    
                    #time tensor
                    t = torch.randint(low = 0, high = self.sample_iters, size = (b,)).to(self.device)
                    
                    #noise matrix
                    e = torch.randn_like(images).to(self.device)
                    
                    sqrt_a         = torch.zeros((b, 1, 1, 1)).to(self.device)
                    sqrt_1_minus_a = torch.zeros((b, 1, 1, 1)).to(self.device)
                    
                    for j in range(b):
                        sqrt_a[j, 0, 0, 0]         = self.a_sgn_sqrt[t[j]]
                        sqrt_1_minus_a[j, 0, 0, 0] = self.one_m_a_sgn_sqrt[t[j]]
                        
                    net_in = sqrt_a*images + sqrt_1_minus_a*e
                    
                    net_out = self.net(net_in, t).detach()
                    
                    loss = self.loss(net_out, e).detach()
                    
                    vl_loss += loss
                    
                    self.train()
                
                vl_loss /= (len(x_vl)/self.batch_size)
                
                if(self.verbose >= 1):
                    print("epoch ", epoch, " vl loss = ", vl_loss.item())
                    
                del vl_loss
            
            self.scheduler.step()
            
    def debug_sample(self,
                     x : torch.utils.data.Dataset,
                     index = 0,
                     sampling_time = 1,
                     verbose_module = 100):
        transform = transforms.ToPILImage()
        
        data_loader = torch.utils.data.DataLoader(dataset = x, 
                                                  batch_size = self.batch_size,
                                                  shuffle = True)
        subset = [index]
        trainset_1 = torch.utils.data.Subset(data_loader.dataset, subset)
        data_loader = torch.utils.data.DataLoader(dataset = trainset_1, 
                                                  batch_size = self.batch_size,
                                                  shuffle = True)
        
        for i, data in enumerate(data_loader, 0):
            #we don't really care about the labels
            images, _ = data
            images = images.to(self.device)
            
            print("Original:")
            r = transform(images[0,:,:,:])
            display(r)
                
            self.eval()
            
            #time tensor
            curr_t = torch.zeros(size = (1,)).type(torch.IntTensor).to(self.device)
            curr_t[0] = sampling_time
            print(curr_t.size())
    
            #noise matrix
            e = torch.randn_like(images).to(self.device)
    
            sqrt_a         = torch.zeros((1, 1, 1, 1)).to(self.device)
            sqrt_1_minus_a = torch.zeros((1, 1, 1, 1)).to(self.device)
    
            sqrt_a[0, 0, 0, 0]         = self.a_sgn_sqrt[curr_t[0].item()]
            sqrt_1_minus_a[0, 0, 0, 0] = self.one_m_a_sgn_sqrt[curr_t[0].item()]
    
            x_T = sqrt_a*images + sqrt_1_minus_a*e
    
            r = transform(x_T[0,:,:,:])
            display(r)
            
            for t in reversed(range(sampling_time)):
                if(t == 0):
                    z = torch.zeros_like(x_T).to(self.device)
                else:
                    z = torch.randn_like(x_T).to(self.device)
    
                model_weight = (1 - self.alphas[t])/(torch.sqrt(1 - self.alphas_sgn[t]))
    
                alpha_sqrt = torch.sqrt(self.alphas[t])
                predicted = self.net(x_T, torch.Tensor([t]).to(self.device)).detach()
                x_T = (x_T - predicted*model_weight)/alpha_sqrt + z*self.sigma[t]
                
                if(t % verbose_module == 0):
                    r = transform(x_T[0,:,:,:])
                    display(r)
            
            self.train()