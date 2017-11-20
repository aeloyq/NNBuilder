# -*- coding: utf-8 -*-
"""
Created on  十一月 16 13:38 2017

@author: aeloyq
"""
from simple import *


class RBFKernel(LayerBase):
    def __init__(self, unit, **kwargs):
        LayerBase.__init__(self, **kwargs)
        self.k_dim = unit
        self.kernelweight = Parameter(self, 'KernelWeight', Parameter.kernelweight, Parameter.randn)

    def set_params(self):
        self.kernelweight.shape = [self.k_dim, self.in_dim]

    def apply(self, X):
        tile_list = [1] + [self.k_dim] * (self.kernelweight().ndim - 1) + [1]
        indice_list = [slice(None, None, None)] + [None] * (self.kernelweight().ndim - 1) + [slice(None, None, None)]
        x_tile = T.tile(X[indice_list], tile_list)
        return x_tile - self.kernelweight
