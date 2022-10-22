# -*- coding: utf-8 -*-

"""
This file defines the class MultiHeadAttention
"""

import torch

from einops import rearrange

from Attention import *

"""
The class which implements multi head attention (not the attention mechanism itself!).
It is essentially a wrapper of the Attention mechanisms defined in Attention.py
which performs the usual linear projections before performing the Attention itself.
Parameters:

device              = 'cuda',      #'cpu' for CPU training, 'cuda' for GPU training
dim          : int  = 3,           #Input image's channels
d_model      : int  = 128,         #Number of channels in the sample after applying Conv2D (the output will still be of <dim> channels)
h            : int  = 8,           #How many attention heads we want to use. d_model must be a multiple of it.
bias         : bool = False,       #Whether the Conv2D should use bias
att_type     : str  = 'FAVOR_SDP', #'SDP' for standard attention, else 'FAVOR_SDP' or 'FAVOR_RELU'
m            : int  = None,        #Number of random orthogonal features to use. None = head_sz*log(head_sz) 
redraw_steps : int  = 1000         #Number of steps before redrawing the random orthogonal features 
"""
class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 device              = 'cuda',      #'cpu' for CPU training, 'cuda' for GPU training
                 dim          : int  = 3,           #Input image's channels
                 d_model      : int  = 128,         #Number of channels in the sample after applying Conv2D (the output will still be of <dim> channels)
                 h            : int  = 8,           #How many attention heads we want to use. d_model must be a multiple of it.
                 bias         : bool = False,       #Whether the Conv2D should use bias
                 
                 att_type     : str  = 'FAVOR_SDP', #'SDP' for standard attention, else 'FAVOR_SDP' or 'FAVOR_RELU'
                 m            : int  = None,        #Number of random orthogonal features to use. None = head_sz*log(head_sz) 
                 redraw_steps : int  = 1000         #Number of steps before redrawing the random orthogonal features 
                 ):
        super(MultiHeadAttention, self).__init__()
        
        #default setup
        self.device         = device
        self.d_model        = d_model
        self.h              = h
        
        self.bias           = bias
        
        self.att_type = att_type
        
        #sanity check
        assert(self.d_model % self.h == 0), "MultiHeadAttention: self.d_model % self.h != 0"
        self.head_sz = self.d_model // self.h #d_k/q/v in the paper
        
        #standard attention uses a simple linear layer, but the DDPM
        #uses a Conv2d instead (which makes sense since we're using images)
        self.W_q = torch.nn.Conv2d(dim, d_model, 1, bias = self.bias).to(self.device)
        self.W_k = torch.nn.Conv2d(dim, d_model, 1, bias = self.bias).to(self.device)
        self.W_v = torch.nn.Conv2d(dim, d_model, 1, bias = self.bias).to(self.device)
        
        self.W_O = torch.nn.Conv2d(d_model, dim, 1, bias = self.bias).to(self.device)
        
        #initializes the correct attention module.
        if(self.att_type == 'SDP'): 
            print("Using standard attention")
            self.attention_module = SDPAttention(device = self.device, d_model = self.d_model, h = self.h)
        else: 
            if(self.att_type == 'FAVOR_SDP'):
                print("Using FAVOR+ with a softmax Kernel")
            else:
                print("Using FAVOR+ with a RELU Kernel")
            self.attention_module = FAVORplus(device = self.device, d_model = self.d_model, 
                                              h = self.h, kernel_type = self.att_type, 
                                              m = m, redraw_steps = redraw_steps)
    
    #x size: [b, c, x, y]
    #b = batch size, c = dim, x = horizontal pixels count and y = vertical pixels count 
    #note that in all our experiments x = y
    def forward(self, x):
        x_sz = x.size(-1) #x_sz = y = x 
        
        #resulting size: [b, d_model = h*d_k, x, y]
        q1 = self.W_q(x)
        k1 = self.W_k(x)
        v1 = self.W_v(x)
        
        #resulting size: [b, h, L, head_sz = d_k]
        #basically: each batch contains a 3d tensor where a slice is a 2D tensor 
        #representing an attention head. For each head (hence, 2D tensor), each 
        #row is a specific pixel and contains the channels value of that specific pixel
        q1 = rearrange(q1, "b (h s) x y -> b h (x y) s", h = self.h)
        k1 = rearrange(k1, "b (h s) x y -> b h (x y) s", h = self.h)
        v1 = rearrange(v1, "b (h s) x y -> b h (x y) s", h = self.h)
        
        #real attention application (refer to the Attention.py file for further infos)
        #res has size [b, h, x_sz*y_sz, head_dim].
        #att has size [b, h, x_sz     , x_sz    ]
        res, att = self.attention_module(q1, k1, v1)
        
        #reshapes as [b, d_model, x, y]
        res = rearrange(res, "b h (x y) s -> b (h s) x y", x = x_sz)
        
        #reverts to the original shape [b, c, x, y]
        res = self.W_O(res)
        
        return res