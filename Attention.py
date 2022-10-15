# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:55:59 2022

@author: Admin
"""

###############################################################################
#FOR THE RECORDS: L = total number of pixels in the image (rows*columns)
###############################################################################

import torch
import math
import numpy as np
import einops

#abstract Attention class
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
        self.head_dim = self.d_model // self.h
    
    """
    Quick reminder that here Q, K and V have already been through a conv2d
    They have size [b, h, L, head_dim = d_k] where L = total number of pixels in the image
    """
    def forward(self, Q, K, V, mask = None):
        pass
    
"""
Standard Scale Dot Product attention
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
        
        self.d_k_sqrt = torch.tensor([self.head_dim], device = self.device)
        self.dropout = torch.nn.Dropout(p = dropout_p)
    
    def forward(self, Q, K, V, mask = None):
        #[b, h, L, L]
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
Performers code (also known as FAVOR+)
"""
class FAVORplus(Attention):
    def __init__(self,
                 device                 = 'cuda',
                 d_model        : int   = 512,
                 h              : int   = 4,            #number of head
                 kernel_type    : str   = 'FAVOR_SDP',  #FAVOR_SDP or FAVOR_RELU supported
                 num_stabilizer : float = .0001,        #used in cases such as relu giving us a whole row equal to 0
                 redraw_steps   : int   = 1000,         #redraw after how many steps?
                 m              : int   = None,         #number of random ortohogonal features
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
        
        if(m is not None): 
            self.m = m
        else:
            #m = d*log(d) is the optimal m size in terms of kernel approx.
            self.m = self.head_sz*math.ceil(math.log(self.head_sz))
        
        #useful in softmax_kernel
        self.sq_d    = torch.sqrt(torch.tensor([self.head_sz], dtype=torch.float32)).to(self.device)
        self.sq_sq_d = torch.sqrt(self.sq_d).to(self.device)
        self.sq_m    = torch.sqrt(torch.tensor([self.m], dtype=torch.float32)).to(self.device)
        
        self.projection_matrix = self.redraw_proj_matrix()
        self.n_calls = 0
        
    #returns a [m, head_sz] matrix where each sub-blocks of
    #rows of size <head_size> are orthogonal between each other.
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
            
        remaining_rows = self.m - n_subblocks*self.d_model
        if(remaining_rows != 0):
            #creates a [head_sz, head_sz] subblock of the random ortohogonal features
            start_matrix = torch.randn((self.head_sz, self.head_sz))
            q, _ = torch.linalg.qr(start_matrix) #it's already in mode "thin qr" 
            
            #transposes q
            q = q.permute((1, 0)).to(self.device)
            
            subblocks.append(q[:remaining_rows, :])
        
        #stacks them vertically
        to_return = torch.cat(subblocks)
        
        return to_return

    #x size: [b, L, h, head_size] (converted by the forward method)
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
        
        #final size: [b, L, h, m]
        return to_return
    
    def relu_kernel(self, x):
        #altough projection_matrix^T has size [head_sz, m] and arr is 
        #[b, L, h, head_sz], the multiplication automatically duplicates the
        #projection matrix into 2 new axis before proceeding.
        arr = x @ self.projection_matrix.permute((1, 0)) #size: [b,L,h,m]
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
        
        q1 = einops.rearrange(q, 'b h L d -> b L h d')
        k1 = einops.rearrange(k, 'b h L d -> b L h d')
        
        #final size: [b, L, h, r] (r = m in softmax and relu)
        if(self.kernel_type == 'FAVOR_RELU'):
            phi_q = self.relu_kernel(q1)
            phi_k = self.relu_kernel(k1)
        else:
            phi_q = self.softmax_kernel(q1)
            phi_k = self.softmax_kernel(k1)
        
        
        #we now need to calculate D = diag(Q'(K'^T * 1L))
        
        #K'^T * 1L effectively deletes the "L" dimention => to do the same
        #with attention heads and batches, we need to bring the L axis on
        #the extreme left => [L, b, h, r]
        phi_q = einops.rearrange(phi_q, 'b L h r -> L b h r')
        phi_k = einops.rearrange(phi_k, 'b L h r -> L b h r')
        
        #k'^T * 1L is effectively the sum of the [b, h, r] sub-blocks
        #(hence, along axis 0)
        #resulting size: [b, h, r]
        phi_k_sum = torch.sum(phi_k, dim = 0)
        
        #Q'(phi_k_sum) is slightly trickier because of how the axis order but
        #it comes down to:
        #1) make an hadamard product between each axis 0 slice of phi_q against phi_k_sum
        #2) sum the columns (last axis) of the result of operation (1)
        #final size: [L, b, h]
        D = phi_q * phi_k_sum
        D = torch.sum(D, dim = -1)
        
        #sets back D to the "proper" axis order where batch is first
        D = einops.rearrange(D, 'L b h -> b L h')
        
        #reshapes as [b, L, h, 1].
        #this has the advantage that if A has shape [b, L, h, d], if we perform
        #A/D we are effectively dividing each row of a single slice [i, j,k,:] of A
        #against the correct normalizer [i, j, k]
        D = torch.unsqueeze(D, -1)
        
        
        #now for the attention matrix
        #A = (d^-1)(Q'((K'^T)V))
        
        #we reshape V to be compatible with q'
        v1 = einops.rearrange(v, 'b h L d -> L b h d')
        
        A = phi_k.permute((0, 1, 3, 2)) @ v1 #[L,b,r,h]x[L,b,h,d] = [L,b,r,d]
        A = phi_q @ A                        #[L,b,h,r]x[L,b,r,d] = [L,b,h,d]
           
        #sets back to the "proper" shape [b, L, h, d]
        A = einops.rearrange(A, 'L b h d -> b L h d')
        
        #[b,h,L,d]/[b, L, h, 1] = [b,h,L,d]
        to_return = A / D
        
        to_return = einops.rearrange(to_return, 'b L h d -> b h L d')
        return to_return, None