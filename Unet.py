# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:54:05 2022

@author: Admin
"""

from inspect import isfunction
from functools import partial
from einops import rearrange
from torch import einsum

import torch
import math


from MultiHeadAttention import MultiHeadAttention

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out = None):
    return torch.nn.Sequential(
        torch.nn.Upsample(scale_factor = 2, mode = 'nearest'),
        torch.nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return torch.nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class Block(torch.nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = torch.nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = torch.nn.GroupNorm(groups, dim_out)
        self.act = torch.nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    #input: [batch_size, 1] > [batch_size, dim] (the rest of the 
    #sinusoidal NN turns it into [batch_size, time_dim])
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResnetBlock(torch.nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.SiLU(),
			
            #dim_out * 2 since it then uses time_emb.chunk(2), splitting it into 2 tensors
            torch.nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = torch.nn.Conv2d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()

    def forward(self, x, time_emb = None):
        #x has size [b, layer_channels, layer_dim, layer_dim]
        #
        scale_shift = None
        
        #time_emb size = [batch_size, time_emb_dim]
        if exists(time_emb):
            #[batch_size, time_emb_dim] => [batch_size, dim_out * 2]
            time_emb = self.mlp(time_emb)
            
            #[batch_size, dim_out * 2] => [batch_size, dim_out * 2, 1, 1]
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            
            #returna 2 tensori di size [batch_size, dim_out, 1, 1]
            scale_shift = time_emb.chunk(2, dim = 1)
        
        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
    
class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class Unet(torch.nn.Module):
    def __init__(
        self,
        dim,                #initial number of channels for the image
        dim_mults           = (1, 2, 4, 8),
        channels            = 3,
        resnet_block_groups = 8,            #groups in the group norm
        time_dim            = 256,
        
        h            : str  = 4,            #attention heads
        att_type     : str  = 'FLAVOR_SDP',
        m            : int  = 64,
        redraw_steps : int  = 1000,
        device              = 'cuda'
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        
        #number of channels obtained by the first convolution
        input_channels = channels

        #init_dim = dim
        init_dim = dim
        
        #initial convolution. Returns a convolution with <init_dim> channels
        self.init_conv = torch.nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        #if(for example) dim_mults = (1, 2, 4, 8), dim = 32 and init_dim = 20, 
        #then dims = [20, 32, 64, 128, 256] = [20, dim*1, dim*2, dim*4, dim*8]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings. The embeddings in the original paper have a size
        #equal to 4*dim. Here it's customizable.
        self.time_dim = time_dim
        
        #the time embeddings are really executed in 2 steps: in the first
        #we retrieve the sinusoidal embeddings which have size dim 
        #(eg: one position has embeddings of size dim) and then does a linear transformation
        #to turn them into time_dim.
        self.time_mlp = torch.nn.Sequential(
            #[batch_sz, 1] => [batch_sz, dim]
            SinusoidalPosEmb(dim),
            
            #[batch_sz, dim] => [batch_sz, time_dim]
            torch.torch.nn.Linear(dim, self.time_dim),
            
            #[batch_sz, time_dim] => [batch_sz, time_dim]
            torch.torch.nn.GELU(),
            
            #[batch_sz, time_dim] => [batch_sz, time_dim]
            torch.torch.nn.Linear(self.time_dim, self.time_dim)
        )

        # layers
        self.downs = torch.nn.ModuleList([])  #compressor
        self.ups = torch.nn.ModuleList([])    #decompressor
        num_resolutions = len(in_out)   #number of layers

        #creates <num_resolutions> layers for the compressor
        for ind, (dim_in, dim_out) in enumerate(in_out):
            #checks if it's the last layer
            is_last = ind == (num_resolutions - 1)
            
            #the stuff inside this ModuleList is all a single  layer's content
            self.downs.append(torch.nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = self.time_dim),
                Residual(PreNorm(dim_in,
                                 MultiHeadAttention(device, dim_in, 32*h, h, False, att_type, m, redraw_steps))),
                Downsample(dim_in, dim_out) if not is_last else torch.nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        
        #mid_dim = size of the compressor's last output
        mid_dim = dims[-1]
        
        #the compressor is made by a series block > attention > block
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim,
                                         MultiHeadAttention(device, mid_dim, 32*h, h, False, 'SDP', m, redraw_steps)))
        #Attention(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = self.time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            #is this the bottomest layer? (happens at the last iteration)
            is_last = ind == (num_resolutions - 1)

            self.ups.append(torch.nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = self.time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = self.time_dim),
                Residual(PreNorm(dim_out, 
                                 MultiHeadAttention(device, dim_out, 32*h, h, False, att_type, m, redraw_steps))),
                Upsample(dim_out, dim_in) if not is_last else  torch.nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels
        self.out_dim = default_out_dim

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = self.time_dim)
        self.final_conv = torch.nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        #input size = [batch_size, channels, dim, dim]
    
        #[batch_size, init_dim, dim2, dim2]
        #(dim2 = dim after convolution)
        x = self.init_conv(x)
        
        #used in the upper-most layer of the decompressor for concatenation
        r = x.clone()

        #size: [batch_size, time_dim]
        t = self.time_mlp(time)

        #saves the residuals of the compressor. They will be concatenated
        #in the decompressor's inputs
        h = []

        #iterates in the compressor (top-bottom)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)
            
            #downsample at the end is just a conv2d which halves the
            #output channels
            x = downsample(x)

        #bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        #decompressor (bottom-up)
        for block1, block2, attn, upsample in self.ups:
            #concatenates the input with the decompressor's output
            #from the same "level" in the Unet (in this case it's the
            #output of the FIRST block).
            x = torch.cat((x, h.pop()), dim = 1)
            
            #first block
            x = block1(x, t)

            #concatenates the input with the decompressor's output
            #from the same "level" in the Unet (in this case it's the
            #output of the SECOND block).
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            
            #attention
            x = attn(x)
            
            #upsample
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
class Attention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = torch.nn.Sequential(torch.nn.Conv2d(hidden_dim, dim, 1), 
                                          torch.nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

