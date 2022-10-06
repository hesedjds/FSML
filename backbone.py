# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Basic ResNet model

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

        
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

    
class Conv2d_mtl(nn.Conv2d): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_mtl, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.mtl_weight = Parameter(torch.ones(out_channels, in_channels, 1, 1))
        self.mtl_weight.fast = None 
        self.weight.requires_grad = False
        
        if self.bias:
            self.mtl_bias = Parameter(torch.zeros(out_channels))
            self.mtl_bias.fast = None 
            self.bias.requires_grad = False
        
    def forward(self, x):
        mtl_weight = self.mtl_weight.fast if self.mtl_weight.fast is not None else self.mtl_weight
        mtl_weight_expand = mtl_weight.expand(self.weight.shape)
        scaled_weight = self.weight.mul(mtl_weight_expand)

        if self.bias:
            mtl_bias = self.mtl_bias.fast if self.mtl_bias.fast is not None else self.mtl_bias 
            shifted_bias = self.bias + mtl_bias 
        else: 
            shifted_bias = None

        out = F.conv2d(x, scaled_weight, shifted_bias, stride=self.stride, padding=self.padding)
            
        return out 

# Simple ResNet Block
class SimpleBlock(nn.Module):
    mtl = False
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.mtl:
            self.C1 = Conv2d_mtl(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = Conv2d_mtl(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
            for p in self.BN1.parameters():
                p.requires_grad = False
            for p in self.BN2.parameters():
                p.requires_grad = False
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.mtl:
                self.shortcut = Conv2d_mtl(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)
                for p in self.BNshortcut.parameters():
                    p.requires_grad = False
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out
       

class ResNet(nn.Module):
    mtl = False
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.mtl:
            conv1 = Conv2d_mtl(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)
            for p in bn1.parameters():
                p.requires_grad = False
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

def ResNet18( flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

class DimReduction(nn.Module):
    mtl = False

    def __init__(self, model_func, dim=128):
        super(DimReduction, self).__init__()
        self.base = model_func()
        self.final_feat_dim = dim
        if self.mtl:
            self.dim_reduction = nn.Linear(self.base.final_feat_dim, dim)
            for p in self.dim_reduction.parameters():
                p.requires_grad = False
        else:
            self.dim_reduction = nn.Linear(self.base.final_feat_dim, dim)

    def forward(self, x):
        feature = self.base(x)
        return self.dim_reduction(F.relu(feature))

def DimReduce(model_func, dim=128):
    return DimReduction(model_func, dim)
