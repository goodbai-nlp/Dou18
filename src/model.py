#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: model.py
@time: 2019/5/23 16:07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import to_variable
class VAE(nn.Module):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.drop1 = nn.Dropout(0.1)
        self.emb_dim = params.g_input_size
        self.map1 = nn.Linear(self.emb_dim, params.g_size, bias=False)
        self.map2 = nn.Linear(params.g_size, self.emb_dim, bias=False)
        self.map3 = nn.Linear(params.g_size, self.emb_dim, bias=False)
        # nn.init.eye(self.map1.weight)

    def encode(self, x):
        z1,z2 = self.map1(x),self.map2(x)
        return z1,z2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = to_variable(torch.randn(std.size()))
        return mu + eps * std

    def decode(self, z):
        decoded = F.linear(z, self.map1.weight.t(), bias=None)
        # decoded = self.map3(z)
        return decoded

    def forward(self, x):
        # mu,logvar = self.encode(x)
        z,_ = self.encode(x)
        # z = self.reparameterize(mu,logvar)
        return self.decode(z),z

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.drop1 = nn.Dropout(0.1)
        self.activation1 = nn.LeakyReLU(0.2)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map21 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(0)  # As per the fb implementation
        self.activation2 = nn.LeakyReLU(0.2)
        self.map3 = nn.Linear(hidden_size, output_size)

    def gaussian(self, ins, mean, stddev):
        noise = torch.autograd.Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins * noise

    def forward(self, x):
        x = self.activation1(self.map1(self.drop1(x)))  # Input dropout
        x = self.drop2(self.activation2(self.map2(x)))
        return torch.sigmoid(self.map3(x)).view(-1)
