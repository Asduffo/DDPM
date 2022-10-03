# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:41:35 2022

@author: Admin
"""

import torch

from einops import rearrange

from Attention import *

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 device              = 'cuda',
                 dim          : int  = 3,      #input channels size
                 d_model      : int  = 128,    #mid channels size
                 h            : int  = 8,
                 bias         : bool = False,
                 
                 att_type     : str  = 'FLAVOR_SDP',
                 m            : int  = 64,
                 redraw_steps : int  = 1000
                 ):
        super(MultiHeadAttention, self).__init__()
        
        self.device         = device
        self.d_model        = d_model
        self.h              = h
        
        self.bias           = bias
        
        self.att_type = att_type
        
        assert(self.d_model % self.h == 0), "MultiHeadAttention: self.d_model % self.h != 0"
        self.head_dim = self.d_model // self.h #d_k/q/v in the paper
        
        #the output size in theory is h*d_k (= d_model)
        self.W_q = torch.nn.Conv2d(dim, d_model, 1, bias = self.bias).to(self.device)
        self.W_k = torch.nn.Conv2d(dim, d_model, 1, bias = self.bias).to(self.device)
        self.W_v = torch.nn.Conv2d(dim, d_model, 1, bias = self.bias).to(self.device)
        
        self.W_O = torch.nn.Conv2d(d_model, dim, 1, bias = self.bias).to(self.device)
        
        if(self.att_type == 'SDP'): 
            print("Using standard attention")
            self.attention_module = SDPAttention(self.device, self.d_model, self.h)
        else: 
            print("using FLAVOR+")
            self.attention_module = FLAVORplus(self.device, self.d_model, self.h, self.att_type, 
                                               m = m, redraw_steps = redraw_steps)
        
    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    #x size: [b, c, x, y]   
    def forward(self, x):
        x_sz = x.size(-1) #x_sz = x
        
        #resulting size: [b, d_model = h*d_k, x, y]
        q1 = self.W_q(x)
        k1 = self.W_k(x)
        v1 = self.W_v(x)
        
        #resulting size: [b, h, L, head_dim = d_k]
        q1 = rearrange(q1, "b (h s) x y -> b h (x y) s", h = self.h)
        k1 = rearrange(k1, "b (h s) x y -> b h (x y) s", h = self.h)
        v1 = rearrange(v1, "b (h s) x y -> b h (x y) s", h = self.h)
        
        #res has size [b, h, L, d_v].
        #att has size [b, h, L, L]
        res, att = self.attention_module(q1, k1, v1)
        
        
        res = rearrange(res, "b h (x y) s -> b (h s) x y", x = x_sz)
        res = self.W_O(res)
        
        return res