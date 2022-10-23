# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from IPython.display import display
from torchvision import transforms

from Unet import Unet

"""
A wrapper for the optimizer. Useful if we want to instantiate it at the __init__() call
"""
class OptimizerData():
    def __init__(self,
                 optimizer_type : torch.optim,
                 **kwargs):
        self.optimizer_type = optimizer_type
        self.kwargs         = kwargs
        
"""
A wrapper for the optimizer. Useful if we want to instantiate it at the __init__() call
"""
class SchedulerData():
    def __init__(self,
                 scheduler_type : torch.optim.lr_scheduler,
                 **kwargs):
        self.scheduler_type = scheduler_type
        self.kwargs         = kwargs

"""
Class defining a diffusion model. Contains everything needed for performing sampling.

Parameters:
n_channels    : int   = 1,           input image channels
x_sz          : int   = 28,          horizontal pixels (at the moment the class only supports square images so it's also the number of columns)
att_type      : str   = 'FAVOR_SDP', 'SDP', 'FAVOR_SDP' or 'FAVOR_RELU'
m             : int   = None,        Number of random orthogonal features to use. None = head_sz*log(head_sz) 
redraw_steps  : int   = 1000,        Number of steps before redrawing the random orthogonal features
init_dim      : int   = None,        Number of channels in the input after the first Conv2D of the Unet. If None, it defaults to x_sz
h             : int   = 4,           Number of attention heads
head_sz       : int   = 32,          Attention head size
 
sample_iters  : int   = 1000,        T in the paper

t_size        : int   = 256,         Dimensionality of the time embeddings (before passing through a linear layer)
device                = 'cuda',      'cpu' for CPU training, 'cuda' for GPU training
b_1           : float = 10e-4,       $\beta_1$ in the DDPM paper
b_T           : float = 0.02,        $\beta_T$ in the DDPM paper

opt_data      : OptimizerData = OptimizerData(torch.optim.Adam, lr = 1e-4),                        The optimizer
sched_data    : SchedulerData = SchedulerData(torch.optim.lr_scheduler.ExponentialLR, gamma = 1),  The optimizer's schedules
loss          : torch.nn      = torch.nn.MSELoss(), #The loss function
batch_size    : int   = 1,
data_slice_tr : int   = 0,           if > 0, use only the first data_slice_tr samples in the training set
data_slice_vl : int   = 0,           if > 0, use only the first data_slice_vl samples in the validation set
n_iters       : int   = 10,          Number of training epochs

verbose       : int   = 1,           If 0, no logs will be printed with the exception of the progress bar

dim_mults             = (1, 2, 4),   Used by the unet (refer to Unet.py for further infos)
resnet_block_groups   = 8,
use_original : bool   = False,       If true, uses the original attention code as used in the paper
clipnorm     : float  = 10000.0      Gradient clip norm. Not particularly important.
"""
class Diffusion(torch.nn.Module):
    def __init__(self,
                 n_channels    : int   = 1,        #input image channels
                 x_sz          : int   = 28,       #horizontal pixels (at the moment the class only supports square images so it's also the number of columns)
                 att_type      : str   = 'FAVOR_SDP', #'SDP', 'FAVOR_SDP' or 'FAVOR_RELU'
                 m             : int   = None,     #Number of random orthogonal features to use. None = head_sz*log(head_sz) 
                 redraw_steps  : int   = 1000,     #Number of steps before redrawing the random orthogonal features
                 init_dim      : int   = None,     #Number of channels in the input after the first Conv2D of the Unet. If None, it defaults to x_sz
                 h             : int   = 4,        #Number of attention heads
                 head_sz       : int   = 32,       #Attention head size
                  
                 sample_iters  : int   = 1000,     #T in the paper
                 
                 t_size        : int   = 256,      #Dimensionality of the time embeddings (before passing through a linear layer)
                 device                = 'cuda',   #'cpu' for CPU training, 'cuda' for GPU training
                 b_1           : float = 10e-4,    #$\beta_1$ in the DDPM paper
                 b_T           : float = 0.02,     #$\beta_T$ in the DDPM paper
                 
                 opt_data      : OptimizerData = OptimizerData(torch.optim.Adam, lr = 1e-4),                        #The optimizer
                 sched_data    : SchedulerData = SchedulerData(torch.optim.lr_scheduler.ExponentialLR, gamma = 1),  #The optimizer's schedules
                 loss          : torch.nn      = torch.nn.MSELoss(), #The loss function
                 batch_size    : int   = 1,
                 data_slice_tr : int   = 0,       #if >0, use only the first data_slice samples in the training/vl set
                 data_slice_vl : int   = 0,       #
                 n_iters       : int   = 10,      #Number of training epochs
                 
                 verbose       : int   = 1,       #If 0, no logs will be printed with the exception of the progress bar
                 
                 dim_mults             = (1, 2, 4), #used by the unet
                 resnet_block_groups   = 8,
                 use_original : bool   = False,   #If true, uses the original attention code as used in the paper
                 clipnorm     : float  = 10000.0  #Gradient clip norm. Not particularly important.
                 ):
        super(Diffusion, self).__init__()
        self.n_channels = n_channels
        self.x_sz      = x_sz
        self.att_type  = att_type
        
        self.sample_iters = sample_iters
        self.t_size     = t_size
        self.device     = device
        
        #Creates the Unet
        self.net        = Unet(
            x_sz                = x_sz,
            init_dim            = init_dim,
            dim_mults           = dim_mults,
            channels            = n_channels,
            resnet_block_groups = resnet_block_groups,
            time_dim            = t_size,
            
            h                   = h,
            head_sz             = head_sz,
            
            att_type            = att_type,
            m                   = m,
            redraw_steps        = redraw_steps,
            device              = device,
            use_original        = use_original
        )
        self.add_module('unet', self.net) #out of safety
        
        #initializes the betas
        self.beta       = torch.linspace(start=b_1, end = b_T, steps = sample_iters).to(self.device)
        
        #initializes the alphas and the alpha_hats
        self.alphas     = 1.0 - self.beta
        self.alphas_sgn = torch.cumprod(self.alphas, dim = 0)
        alphas_sgn_prev = torch.roll(self.alphas_sgn, 1, 0)
        alphas_sgn_prev[0] = 1
        
        #square root of the various alpha_hats and 1-alpha_hats (so that we do not need to calculate them every time)
        self.a_sgn_sqrt       = torch.sqrt(self.alphas_sgn)     #sqrt(alpha sign)
        self.one_m_a_sgn_sqrt = torch.sqrt(1 - self.alphas_sgn) #sqrt(1 - alpha sign)
        
        #on page 3 there is a more complex formula but this works ok. NB: it's sigma and not sigma^2
        self.sigma      = torch.sqrt(self.beta*(1. - alphas_sgn_prev)/(1. - self.alphas_sgn))
        
        #various default initializations
        self.opt_data   = opt_data
        self.loss       = loss
        self.n_iters    = n_iters
         
        self.verbose    = verbose
        
        self.batch_size = batch_size
        self.data_slice_tr = data_slice_tr
        self.data_slice_vl = data_slice_vl
        
        self.optimizer  = opt_data.optimizer_type(self.parameters(), **(opt_data.kwargs))
        self.scheduler  = sched_data.scheduler_type(self.optimizer, **(sched_data.kwargs))
        
        self.clipnorm   = clipnorm
    
    """
    The standard sampling algorithm as shown in the paper
    Parameters:
        
    verbose_steps : int = 0: if > 0, shows the intermediate results each <verbose_steps> steps
    """
    def sample(self, 
               verbose_steps : int = 0): #If > 0, displays the intermediate results each <verbose_steps> steps
        self.eval()
        
        #display utility
        transform = transforms.ToPILImage()
        
        #creates one image made of pure noise
        x_T = torch.randn(size = (1, self.n_channels, self.x_sz, self.x_sz)).to(self.device)
        
        #backward procedure
        for t in reversed(range(self.sample_iters)):
            #choses the right z
            if(t == 0):
                z = torch.zeros_like(x_T).to(self.device)
            else:
                z = torch.randn_like(x_T).to(self.device)
            
            #x_{t-1} constants
            model_weight = self.beta[t]/self.one_m_a_sgn_sqrt[t]
            alpha_sqrt = torch.sqrt(self.alphas[t])
            
            #calculates the noise
            predicted = self.net(x_T, torch.Tensor([t]).to(self.device)).detach()
            
            #removes the noise
            x_T = (x_T - model_weight*predicted)/alpha_sqrt + z*self.sigma[t]
            
            #displays the image if asked
            if(verbose_steps > 0):
                if(t % verbose_steps == 0):
                    print(t)
                    r = transform(x_T[0,:,:,:])
                    display(r)
        
        self.train()
        return x_T[0, :, :, :]
    
    """
    The training algorithm.
    
    x    : torch.utils.data.Dataset,       #Training set
    x_vl : torch.utils.data.Dataset = None #Validation set (if any)
    """
    def fit(self, 
            x    : torch.utils.data.Dataset,
            x_vl : torch.utils.data.Dataset = None):
        
        #pytorch's dataloader for the training set
        data_loader = torch.utils.data.DataLoader(dataset = x, 
                                                  batch_size = self.batch_size,
                                                  shuffle = True)
        
        #training history
        self.tr_loss_arr = []
        self.vl_loss_arr = []
        
        #if requested, loads the dataloader with only a slice of the training set
        if(self.data_slice_tr > 0):
            subset = list(range(0, self.data_slice_tr))
            trainset_1 = torch.utils.data.Subset(data_loader.dataset, subset)
            data_loader = torch.utils.data.DataLoader(dataset = trainset_1, 
                                                      batch_size = self.batch_size,
                                                      shuffle = True)
        
        #same procedure for the dataloader for the validation set.
        data_loader_vl = None
        if(not x_vl is None):
            data_loader_vl = torch.utils.data.DataLoader(dataset = x_vl, 
                                                         batch_size = self.batch_size,
                                                         shuffle = True)
            if(self.data_slice_vl > 0):
                subset = list(range(0, self.data_slice_vl))
                trainset_1 = torch.utils.data.Subset(data_loader_vl.dataset, subset)
                data_loader_vl = torch.utils.data.DataLoader(dataset = trainset_1, 
                                                             batch_size = self.batch_size,
                                                             shuffle = True)
            
        #training loop
        for epoch in range(self.n_iters):
            #training loss vector (useful for the print)
            tr_loss = torch.tensor([0]).type(torch.FloatTensor).to(self.device)
            
            #wraps the dataloader in a tqdm structure for the smart progress bar
            enumerator = tqdm(data_loader)
            for i, data in enumerate(enumerator):
                #we don't really care about the labels
                images, _ = data
                
                #safety measure
                images = images.to(self.device) 
                
                #batch size:
                b = images.size(0)
                
                #time tensor
                t = torch.randint(low = 0, high = self.sample_iters, size = (b,)).to(self.device)
                
                #noise matrix
                e = torch.randn_like(images).to(self.device)
                
                #initializes the sqrt(a_sgn) and sqrt(1-a_sgn) tensors. One element
                #for each sample in the batch
                sqrt_a         = torch.zeros((b, 1, 1, 1)).to(self.device)
                sqrt_1_minus_a = torch.zeros((b, 1, 1, 1)).to(self.device)
                
                #sets the sqrt(a_sgn) and sqrt(1-a_sgn) to their respective values
                for j in range(b):
                    sqrt_a[j, 0, 0, 0]         = self.a_sgn_sqrt[t[j]]
                    sqrt_1_minus_a[j, 0, 0, 0] = self.one_m_a_sgn_sqrt[t[j]]
                
                #batch images with the extra noise
                net_in = sqrt_a*images + sqrt_1_minus_a*e
                
                #calculates the noise
                net_out = self.net(net_in, t)
                
                #loss calculation and backpropagation as usual
                loss = self.loss(net_out, e)
                    
                #pytorch's optimizer routine
                loss_det = loss.clone().detach()
                tr_loss += loss_det
                
                enumerator.set_description("epoch %i step %i loss = %.5f" %(epoch, i, loss_det.item()))
                
                self.optimizer.zero_grad()
                loss.backward()
                
                #norm clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipnorm)

                self.optimizer.step()
            
            #average loss
            tr_loss /= ((len(x) if self.data_slice_tr == 0 else self.data_slice_tr)/self.batch_size)
            self.tr_loss_arr.append(tr_loss.item())
            to_print = "epoch {:d} tr loss = {:.5f}".format(epoch, tr_loss.item())
            
            #string containing the vl loss infos (if any)
            to_add = ""
            
            #same thing as above but with the validation set and gradient stuff disabled
            if(not x_vl is None):
                vl_loss = torch.tensor([0]).type(torch.FloatTensor).to(self.device)
                
                self.eval()
                enumerator = tqdm(data_loader_vl)
                for i, data in enumerate(enumerator):
                    
                    
                    #we don't really care about the labels
                    images, _ = data
                    
                    #safety measure
                    images = images.to(self.device)
                    
                    #batch size:
                    b = images.size(0)
                    
                    #time tensor
                    t = torch.randint(low = 0, high = self.sample_iters, size = (b,)).to(self.device)
                    
                    #noise matrix
                    e = torch.randn_like(images).to(self.device)
                    
                    #initializes the sqrt(a_sgn) and sqrt(1-a_sgn) tensors. One element
                    #for each sample in the batch
                    sqrt_a         = torch.zeros((b, 1, 1, 1)).to(self.device)
                    sqrt_1_minus_a = torch.zeros((b, 1, 1, 1)).to(self.device)
                    
                    #sets the sqrt(a_sgn) and sqrt(1-a_sgn) to their respective values
                    for j in range(b):
                        sqrt_a[j, 0, 0, 0]         = self.a_sgn_sqrt[t[j]]
                        sqrt_1_minus_a[j, 0, 0, 0] = self.one_m_a_sgn_sqrt[t[j]]
                    
                    #batch images with the extra noise
                    net_in = sqrt_a*images + sqrt_1_minus_a*e
                    
                    #calculates the noise. Detach() since otherwise it will be kept
                    #in the memory and eventually cause cuda to blow up the vram
                    net_out = self.net(net_in, t).detach()
                    
                    #standard loss calculation
                    loss = self.loss(net_out, e).detach()
                    vl_loss += loss
                    
                    enumerator.set_description("epoch %i step %i vl loss = %.5f" %(epoch, i, loss.item()))
                    
                self.train()
                
                vl_loss /= ((len(x_vl) if self.data_slice_vl == 0 else self.data_slice_vl)/self.batch_size)
                self.vl_loss_arr.append(vl_loss.item())
                    
                to_add = ", vl_loss = {:.5f}".format(vl_loss.item())
                
                del vl_loss
                
            to_print = to_print + to_add
            if(self.verbose >= 1):
                print(to_print)
            
            print("==========================================================")
            
            self.scheduler.step()
            
    """
    Kept only for debugging purposes. Given a dataset x and an index <index> plus a
    sampling time <sampling_time>, creates (x[index])_{sampling_time} and attempts to denoise it
    
    Params:
    x : torch.utils.data.Dataset, #Dataset
    index          = 0,           #Sample index in x
    sampling_time  = 1,           #Sample timestep
    verbose_module = 100          #if > 0, displays the intermediate result every <verbose_module> timesteps
    """
    def debug_sample(self,
                     x : torch.utils.data.Dataset,
                     index          = 0,
                     sampling_time  = 1,
                     verbose_module = 100):
        transform = transforms.ToPILImage()
        
        #loads the training sample
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
            
            #displays the original image
            print("Original:")
            r = transform(images[0,:,:,:])
            display(r)
                
            #disables training overhead
            self.eval()
            
            #time tensor
            curr_t = torch.zeros(size = (1,)).type(torch.IntTensor).to(self.device)
            curr_t[0] = sampling_time
            print(curr_t.size())
    
            #noise matrix
            e = torch.randn_like(images).to(self.device)
    
            #alphas and sqrt(1 - alpha)
            sqrt_a         = torch.zeros((1, 1, 1, 1)).to(self.device)
            sqrt_1_minus_a = torch.zeros((1, 1, 1, 1)).to(self.device)
    
            sqrt_a[0, 0, 0, 0]         = self.a_sgn_sqrt[curr_t[0].item()]
            sqrt_1_minus_a[0, 0, 0, 0] = self.one_m_a_sgn_sqrt[curr_t[0].item()]
            
            #x_<sampling_time>
            x_T = sqrt_a*images + sqrt_1_minus_a*e
            
            #displays x_<sampling_time>
            r = transform(x_T[0,:,:,:])
            display(r)
            
            #inverse sampling procedure
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
           
    """
    Plots training and validation losses
    Parameters:
        
    name: list of strings indicating what we want to display in the graph
          'tr_loss' and 'vl_loss' are currently supported.
    ylim: y limiter
    dpi:  image quality
    """
    def plot(self, name : list, ylim = None, dpi : int = 500):
        """ Plot the results """
        plt.figure(dpi = dpi)
        plt.xlabel('epoch')
        
        assert(len(name) <= 2), "NeuralNetwork.plot: max 2 losses!"
        
        
        plot_colors = ['b-', 'r--']
        c = 0
        
        for line in name:
            if(line == 'tr_loss'):
                plt.plot(self.tr_loss_arr, plot_colors[c], label='Training loss')
                plt.ylabel("Loss")
            elif(line == 'vl_loss'):
                plt.plot(self.vl_loss_arr, plot_colors[c], label='Validation loss')
                plt.ylabel("Loss")
            else:
                raise Exception("Unrecognized loss name")
            c = c + 1
        
        if(not ylim is None):
            plt.ylim(ylim)
        
        plt.legend(fontsize  = 'large')
        plt.show()