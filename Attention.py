# -*- coding: utf-8 -*-

###############################################################################
#FOR THE RECORDS: L = total number of pixels in the image (rows*columns)
###############################################################################

import torch
import math
import numpy as np
import einops

"""
An abstract class for the attention mechanism it only contains some sanity checks
for the head size
"""
class Attention(torch.nn.Module):
    def __init__(self,
                 device        = 'cuda',
                 d_model : int = 128,   #h*(heads dimention).
                 h       : int = 4,     #number of heads
                 ):
        super(Attention, self).__init__()
        
        self.device  = device
        self.d_model = d_model
        self.h       = h
        
        #d_model must be a multiple of h
        assert(self.d_model % self.h == 0), "MultiHeadAttention: self.d_model % self.h != 0"
        self.head_sz = self.d_model // self.h
    
    """
    Quick reminder that here Q, K and V have already been through a conv2d
    They have size [b, h, L, head_sz = d_k] where L = total number of pixels in the image
    """
    def forward(self, Q, K, V, mask = None):
        pass
    
"""
Standard Scale Dot Product attention.
Parameters:

device             = 'cuda', 'cpu' for CPU training, 'cuda' for GPU training
d_model    : int   = 512,    h*head_sz
h          : int   = 8,      How many attention heads we want to use. d_model must be a multiple of it.
 
dropout_p  : float = 0.1     One weird thing is that both tensor2tensor and bert 
                             apparently applies dropout to V...
"""
class SDPAttention(Attention):
    def __init__(self,
                 device             = 'cuda',
                 d_model    : int   = 512,
                 h          : int   = 8,
                 
                 dropout_p  : float = 0.1
                 ):
        super(SDPAttention, self).__init__(device,
                                        d_model,
                                        h)
        
        self.d_k_sqrt = torch.tensor([self.head_sz], device = self.device)
        self.dropout = torch.nn.Dropout(p = dropout_p)
    
    def forward(self, Q, K, V, mask = None):
        #[b, h, L, L]
        #Q*K^T
        prod = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.d_k_sqrt
        
        #unused in the final code
        if mask is not None:
            prod = prod.masked_fill(mask == 0, -1e10)
            
        #applies softmax
        att = torch.softmax(prod, dim = -1)
        
        #one weird thing is that both tensor2tensor and bert apparently applies dropout to V...
        #[b, h, L, d_v]
        res = torch.matmul(self.dropout(att), V)
        
        return res, att
    
"""
Performer's code (also known as FAVOR+)

Parameters:
device                 = 'cuda',       'cpu' for CPU training, 'cuda' for GPU training
d_model        : int   = 512,
h              : int   = 4,            How many attention heads we want to use. d_model must be a multiple of it.
kernel_type    : str   = 'FAVOR_SDP',  FAVOR_SDP or FAVOR_RELU
num_stabilizer : float = .0001,        The numerical stability constant
redraw_steps   : int   = 1000,         Number of steps before redrawing the random orthogonal features 
m              : int   = None,         Number of random ortohogonal features
"""
class FAVORplus(Attention):
    def __init__(self,
                 device                 = 'cuda',
                 d_model        : int   = 512,
                 h              : int   = 4,            #How many attention heads we want to use. d_model must be a multiple of it.
                 kernel_type    : str   = 'FAVOR_SDP',  #FAVOR_SDP or FAVOR_RELU supported
                 num_stabilizer : float = .0001,        #The numerical stability constant
                 redraw_steps   : int   = 1000,         #Number of steps before redrawing the random orthogonal features
                 m              : int   = None,         #Number of random ortohogonal features
                 ):
        super(FAVORplus, self).__init__(device,
                                        d_model,
                                        h)
        
        self.device         = device
        self.d_model        = d_model
        self.h              = h
        self.kernel_type    = kernel_type
        self.num_stabilizer = num_stabilizer
        self.redraw_steps   = redraw_steps
        
        #sets m to the default value if None
        if(m is not None): 
            self.m = m
        else:
            #m = d*log(d) is the optimal m size in terms of kernel approx.
            self.m = self.head_sz*math.ceil(math.log(self.head_sz))
            print(math.log(self.head_sz))
            print(self.m)
        
        #Constants useful in the softmax_kernel
        self.sq_d    = torch.sqrt(torch.tensor([self.head_sz], dtype=torch.float32)).to(self.device)
        self.sq_sq_d = torch.sqrt(self.sq_d).to(self.device)
        self.sq_m    = torch.sqrt(torch.tensor([self.m], dtype=torch.float32)).to(self.device)
        
        self.projection_matrix = self.redraw_proj_matrix()
        self.n_calls = 0
        
    #returns a [m, head_sz] matrix where the rows of each sub-blocks of
    #size <head_size> are orthogonal between each other.
    def redraw_proj_matrix(self):
        n_subblocks = int(self.m / self.head_sz)
        
        #it is easier to store the subblocks in an array and then just use cat
        subblocks = []
        
        for i in range(n_subblocks):
            #creates a [head_sz, head_sz] subblock of the random ortohogonal features
            start_matrix = torch.randn((self.head_sz, self.head_sz))
            q, _ = torch.linalg.qr(start_matrix) #it's already in mode "thin qr" 
            
            #transposes q
            q = q.permute((1, 0)).to(self.device)
            
            subblocks.append(q)
            
        #if m is not a multiple of head_sz, the last subblock contains less
        #than head_sz rows and we need to act accordingly
        remaining_rows = self.m - n_subblocks*self.head_sz
        if(remaining_rows > 0):
            #creates a [head_sz, head_sz] subblock of the random ortohogonal features
            start_matrix = torch.randn((self.head_sz, self.head_sz))
            q, _ = torch.linalg.qr(start_matrix) #it's already in mode "thin qr" 
            
            #transposes q
            q = q.permute((1, 0)).to(self.device)
            
            #gets only the first <remaining_rows> rows
            subblocks.append(q[:remaining_rows, :])
        
        #stacks them vertically
        to_return = torch.cat(subblocks)
        
        return to_return

    """
    Softmax kernel (positive features)
    
    x size: [b, h, L, head_size]
    """
    def softmax_kernel(self, x):
        #x/d**(0.25)
        arr = x/self.sq_sq_d
        
        #altough projection_matrix^T has size [head_sz, m] and arr is 
        #[b, L, h, head_sz], the multiplication automatically duplicates the
        #projection matrix into 2 new axis before proceeding.
        arr = arr @ self.projection_matrix.permute((1, 0)) #size: [b,L,h,m]
        
        #(||x||^2)/(2*sqrt(head_sz))
        #final size: [b, L, h, 1]
        g = x**2
        g = torch.sum(g, dim = -1, keepdim = True)
        g = g/(2*self.sq_d)
        
        #since both g and arr's single members are exponentials, and we are
        #supposed to multiply them, we can just sum them inside the exponent)
        to_return = torch.exp(arr - g + self.num_stabilizer)/self.sq_m
        
        #final size: [b, h, L, m]
        return to_return
    
    """
    ReLU kernel
    
    x size: [b, h, L, head_size]
    """
    def relu_kernel(self, x):
        #altough projection_matrix^T has size [head_sz, m] and arr is 
        #[b, h, L, head_sz], the multiplication automatically duplicates the
        #projection matrix into 2 new axis before proceeding.
        arr = x @ self.projection_matrix.permute((1, 0)) #size: [b,h,Lm]
        arr = arr/self.sq_m
        arr = torch.nn.functional.relu(arr) + torch.tensor([self.num_stabilizer]).to(self.device)
        
        return arr
    
    #q,k and v size: [b, h, L, head_size]
    #they are already multiplied by W_q etc
    def forward(self, q, k, v):
        if(self.n_calls > self.redraw_steps):
            self.n_calls = 0
            self.projection_matrix = self.redraw_proj_matrix()
        self.n_calls += 1
        
        #final size: [b, h, L, r] (r = m in softmax and relu)
        #calculates phi(Q) and phi(K)
        if(self.kernel_type == 'FAVOR_RELU'):
            phi_q = self.relu_kernel(q)
            phi_k = self.relu_kernel(k)
        else:
            phi_q = self.softmax_kernel(q)
            phi_k = self.softmax_kernel(k)
        
        
        #"official implementation". Left here only for benchmarks. My implementation seems faster :D
        """
        phi_k_sum = phi_k.sum(dim = -2)
        D         = torch.einsum('...nd,...d->...n', phi_q, phi_k_sum.type_as(phi_q))
        D_inv     = 1.0/D
        context   = torch.einsum('...nd,...ne->...de', phi_k, v)
        to_return = torch.einsum('...de,...nd,...n->...ne', context, phi_q, D_inv)
        """
        
        #(K'^T)*(1L)
        #size: [b, h, r, 1]
        phi_k_sum = phi_k.sum(dim = -2).unsqueeze(-1)
        
        #D = Q' @ phi_k_sum
        #size: [b, h, L, r]x[b, h, r, 1] = [b, h, L, 1]
        D         = phi_q @ phi_k_sum
        D_inv     = 1.0/D
        
        #(K'^T) @ V => [b, h, r, L]x[b, h, L, head_sz] = [b, h, r, head_sz]
        to_return = phi_k.permute((0, 1, 3, 2)) @ v
        
        #phi_q @ to_return => [b, h, L, r]x[b, h, r, head_sz] = [b, h, L, head_sz]
        to_return = phi_q @ to_return
        
        #D^{-1} * to_return => [b, h, L, 1] * [b, h, L, head_sz] = [b, h, L, head_sz]
        to_return = D_inv * to_return
        
        return to_return, None