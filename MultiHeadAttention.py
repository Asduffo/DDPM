# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:41:35 2022

@author: Admin
"""

import torch

from einops import rearrange

from Attention import *

#multi head attention class
class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 device              = 'cuda',      #device
                 dim          : int  = 3,           #number of input channels
                 d_model      : int  = 128,         #number of channels after applying conv2D (the output will still be of dim channels)
                 h            : int  = 8,           #attention heads. d_model must be a multiple of it.
                 bias         : bool = False,       #wether the Conv2D should use bias
                 
                 att_type     : str  = 'FAVOR_SDP', #'SDP' for standard attention, else FAVOR_SDP/FLVOR_RELU
                 m            : int  = None,        #number of orthogonal random features for FLAVOR+
                 redraw_steps : int  = 1000         #after how many steps should we redraw the orthogonal random features 
                 ):
        super(MultiHeadAttention, self).__init__()
        
        self.device         = device
        self.d_model        = d_model
        self.h              = h
        
        self.bias           = bias
        
        self.att_type = att_type
        
        #sanity check
        assert(self.d_model % self.h == 0), "MultiHeadAttention: self.d_model % self.h != 0"
        self.head_dim = self.d_model // self.h #d_k/q/v in the paper
        
        #the output size in theory is h*d_k (= d_model)
        #standard attention uses a simple linear layer, but the author of the paper
        #uses a Conv2d instead (which makes sense since we're using images)
        self.W_q = torch.nn.Conv2d(dim, d_model, 1, bias = self.bias).to(self.device)
        self.W_k = torch.nn.Conv2d(dim, d_model, 1, bias = self.bias).to(self.device)
        self.W_v = torch.nn.Conv2d(dim, d_model, 1, bias = self.bias).to(self.device)
        
        self.W_O = torch.nn.Conv2d(d_model, dim, 1, bias = self.bias).to(self.device)
        
        #initializes the correct attention module.
        if(self.att_type == 'SDP'): 
            print("Using standard attention")
            self.attention_module = SDPAttention(self.device, self.d_model, self.h)
        else: 
            print("using FAVOR+")
            self.attention_module = FAVORplus(self.device, self.d_model, self.h, self.att_type, 
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
        
        #resulting size: [b, h, L, head_dim = d_k]
        #basically: each batch contains a 3d tensor where a slice is a 2D tensor 
        #representing an attention head. For each head, each row is a specific
        #pixel and contains the channels value of that specific pixel
        q1 = rearrange(q1, "b (h s) x y -> b h (x y) s", h = self.h)
        k1 = rearrange(k1, "b (h s) x y -> b h (x y) s", h = self.h)
        v1 = rearrange(v1, "b (h s) x y -> b h (x y) s", h = self.h)
        
        #real attention application (refer to the Attention.py file for further infos)
        #res has size [b, h, x_sz*x_sz, head_dim].
        #att has size [b, h, x_sz     , x_sz    ]
        res, att = self.attention_module(q1, k1, v1)
        
        #reshapes as [b, d_model, x, y]
        res = rearrange(res, "b h (x y) s -> b (h s) x y", x = x_sz)
        
        #reverts to the original shape [b, c, x, y]
        res = self.W_O(res)
        
        return res