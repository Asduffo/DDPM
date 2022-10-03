# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:55:59 2022

@author: Admin
"""

import torch
import math
import numpy as np

class Attention(torch.nn.Module):
    def __init__(self,
                 device           = 'cuda',
                 d_model    : int = 512,
                 h          : int = 8,
                 ):
        super(Attention, self).__init__()
        
        self.device  = device
        self.d_model = d_model
        self.h       = h
        
        assert(self.d_model % self.h == 0), "MultiHeadAttention: self.d_model % self.h != 0"
        self.head_dim = self.d_model // self.h
        
    #switches from [b, L, d_model = d_k*h] to [b, h, L, d_k]
    def transpose_heads(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    """
    Quick reminder that here Q, K and V have already been multiplied by a linear matrix
    They have size [b, L, d_model]
    """
    def forward(self, Q, K, V, mask = None):
        pass
    
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
        #useful below
        batch_size = Q.size(0)
        
        #[b, h, L, L]
        prod = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.d_k_sqrt
        
        if mask is not None:
            prod = prod.masked_fill(mask == 0, -1e10)
            
        #applies softmax
        att = torch.softmax(prod, dim = -1)
        
        #one weird thing is that both tensor2tensor and bert apparently applies dropout to V...
        #[b, h, L, d_v]
        res = torch.matmul(self.dropout(att), V)
        
        #[b, L, h, d_v]
        # res = res.permute(0, 2, 1, 3).contiguous()
        
        #[b, L, h*d_v = d_model]
        # res = res.view(batch_size, -1, self.d_model)
        
        return res, att
    
class FLAVORplus(Attention):
    def __init__(self,
                 device                       = 'cuda',
                 d_model              : int   = 512,
                 h                    : int   = 4,            #number of head
                 kernel_type          : str   = 'FLAVOR_SDP', #FLAVOR_SDP or FLAVOR_RELU supported
                 numerical_stabilizer : float = .0001,        #used in cases such as relu giving us a whole row equal to 0
                 redraw_steps         : int   = 1000,         #redraw after how many steps?
                 m                    : int   = 64,           #number of random ortohogonal features
                 renormalize          : bool  = False
                 ):
        super(FLAVORplus, self).__init__(device,
                                         d_model,
                                         h)
        
        self.f = [] 
        self.g = None #h in the paper
        self.m = m
        
        
        self.kernel_type          = kernel_type
        self.numerical_stabilizer = numerical_stabilizer
        self.redraw_steps         = redraw_steps
        self.curr_redraw_step     = 0
        self.renormalize          = renormalize
        
        self.w = self.redraw_w()
        
        self.f = []
        self.f_count = 2
        if(self.kernel_type == 'FLAVOR_SDP'):
            self.f_count = 2
            self.f.append(torch.exp)
            self.f.append(torch.exp)
        else: #default is RELU
            self.f_count = 1
            self.f.append(torch.nn.ReLU())
    
    def redraw_w(self):
        #it is easy to create it with numpy
        H = np.random.rand(self.h, self.head_dim, self.m)
        
        #orthogonalizes the heads
        for i in range(self.h):
            u, s, vh = np.linalg.svd(H[i,:,:], full_matrices=False)
            H[i,:,:] = u @ vh
        
        #converts to torch tensor
        w = torch.from_numpy(H).to(self.device)
        w.double()
        w.require_grad = False
        
        return w
    
    #This is the real complicated part: given M of size [b, h, L, head_size], performs
    #a random feature map into its rows
    def phi(self, M : torch.Tensor):
        #multiplies the matrix by the orthogonal features
        #final size = [b, h, L, m]
        #CORRECT
        o = M.double() @ self.w
        
        #creates an array of tensors where each one is the o tensor mapped to a certain function
        mapped_blocks = []
        if(self.kernel_type == 'FLAVOR_SDP'):
            mapped_blocks.append(self.f[0](o))
            mapped_blocks.append(self.f[1](-o))
        else:
            mapped_blocks.append(self.f[0](o))
        
        #concatenates the various blocks by columns. Final size: [b, h, L, m*self.f_count]
        #CORRECT
        big_o = mapped_blocks[0]
        for i in range(1, self.f_count):
            # dim = -1 means to concatenate by the columns of the last axis
            big_o = torch.cat((big_o, mapped_blocks[i]), dim = -1)
        
        big_o = big_o + self.numerical_stabilizer
        
        #DEBUG
        # [b, h, L, m*self.f_count]
        # big_o = torch.randn((2, 3, 4, 5))
        
        M_size = M.size()
        M_b = M_size[0]
        M_h = M_size[1]
        M_L = M_size[2]
        
        # torch.autograd.set_detect_anomaly(True)
        
        #creates a column tensor of L elements where row i contains g(M[i-th row])
        g_vector = torch.ones((M_b, M_h, M_L, 1)).to(self.device)
        
        
        if(self.kernel_type == 'FLAVOR_SDP'):
            g_vector  = torch.norm(M, p = 2, dim = -1, keepdim = True) #norm
            g_vector  = -torch.square(g_vector)/2                      #squared norm, divide by -1/2
            g_vector  = torch.exp(g_vector)/math.sqrt(2)               #exponent, divide by 2
        #the else is (at the moment) useless since the other implemented kernel is RELU and g is the identity function
        #NB: dividing by sqrt(m) is done below
        
        # print(g_vector.size())
        
        #multiplies each row (meaning, penultimate axis) in the big_o tensor by
        #the corresponding element in g, hadamard style.
        #CORRECT (or so it seems)
        big_o = (big_o * g_vector) / math.sqrt(self.m)
        
        #unfortunately looks like there are no functions which allows for mapping
        #each row of each batch to a certain 
        return big_o
    
    def forward(self, Q, K, V, mask = None):
        #they are all expected to be of size [batch_size, h, L, head_size]
        #NB: r = m*self.f_count
        
        self.curr_redraw_step += 1
        if(self.curr_redraw_step % self.redraw_steps == 0):
            self.w = self.redraw_w()
            
        #new size: [b, h, L, r]
        Q_p = self.phi(Q).float()
        K_p = self.phi(K).float()
        
        
        #we get V's shape and set the last dimension (the columns) to 1, which is
        #the shape of 1L => in theory it should be of size [b, h, L, 1]
        oneL_size = list(V.size())
        oneL_size[-1] = 1
        ones = torch.ones(oneL_size).to(self.device)
        
        #size = [b, h, L, d_v + 1]
        C = torch.cat((V, ones), dim = -1)
        
        Buf2 = None
        
        #K_p.permute((-1, -2)) transposes the last two axis
        #size: [b, h, r, (d_v + 1)]
        Buf1 = K_p.permute((0, 1, 3, 2)) @ C
        
        #size: [b, h, L, (d_v + 1)]
        Buf2 = Q_p @ Buf1
        
        
        Buf3 = Buf2[:, :, :,  :-1]  #Buf2 except for the last column              => [b, h, L, d_v]
        buf4 = Buf2[:, :, :, [-1]]  #Buf2 last column. The [] preserves the shape => [b, h, L,   1]
        
        #buf3 size = [b, h, L, d_v]
        #buf4 size = [b, h, L, 1]
        #=> each column in buf4 should be used for the diagonal matrices that the paper calls D
        #=> in theory, D has size [b, h, L, L] where the last 2 axis are all diagonal matrices.
        #BUT we can just multiply the columns of the last 2 axis in hadamard style
        #against the column in buf4. This saves complexity since it goes from O(L^2 * d) to O(Ld)
        
        #it multiplies (hadamard style) the columns of the last 2 axis of Buf3 by the
        #column vector in the last 2 axis of buf4
        #final size: [b, h, L, d_v]
        
        #the paper says that they have also tested the model without renormalization.
        #normalization is prohibited in ReLU, since if one whole row in Q or K is zero,
        #we end up with zeros on buf4 as well and 1/buf4 is not valid here.
        if(self.renormalize):
            #same shape as buf4 but each element has numerator and denominator inverted.
            to_return = 1/buf4 * Buf3
        else:
            to_return = Buf3
        
        return to_return, None
