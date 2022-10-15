# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:15:12 2022

@author: Admin
"""

from Diffusion import Diffusion

from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

"""
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    transforms.Resize((32, 32))
])
"""

image_size = 32

transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(), # turn into Numpy array of shape HWC, divide by 255
    Lambda(lambda t: (t * 2) - 1),
    
])


mnist_tr_set = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
mnist_ts_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

d = Diffusion(batch_size          = 1,
              n_channels          = 1,
              x_sz                = 32,
              y_sz                = 32,
              verbose             = 1,
              n_iters             = 10,
              data_slice_tr       = 10,
              data_slice_vl       = 5,
              att_type            = 'FAVOR_RELU',
              resnet_block_groups = 8,
              m                   = None,
              use_original        = False)
d.to('cuda')
d.fit(mnist_tr_set, mnist_ts_set)

# transform = transforms.ToPILImage()
# r = transform(d.sample())