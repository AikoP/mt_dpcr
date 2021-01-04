#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""

import torch
import torch.nn as nn
from utils.activations import Mish

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class Model(nn.Module):

    def __init__(self, d = 3, k = 10, emb_dims = 1024, dropout = 0.5):

        super(Model, self).__init__()

        self.emb_dims = emb_dims
        self.k = k
        self.dropout = dropout
        self.d = d

        self.conv1 = nn.Sequential(nn.Conv2d(2*self.d,              64,             kernel_size=1, bias=False), nn.BatchNorm2d(64),Mish())
        self.conv2 = nn.Sequential(nn.Conv2d(64,                    64,             kernel_size=1, bias=False), nn.BatchNorm2d(64),Mish())

        self.conv3 = nn.Sequential(nn.Conv2d(64*2,                  64,             kernel_size=1, bias=False), nn.BatchNorm2d(64),Mish())
        self.conv4 = nn.Sequential(nn.Conv2d(64,                    64,             kernel_size=1, bias=False), nn.BatchNorm2d(64),Mish())

        # self.conv5 = nn.Sequential(nn.Conv2d(64*2,                  64,             kernel_size=1, bias=False), nn.BatchNorm2d(64),Mish())

        # self.conv6 = nn.Sequential(nn.Conv1d(64*3,                  self.emb_dims,  kernel_size=1, bias=False), nn.BatchNorm1d(self.emb_dims),Mish())
        self.conv6 = nn.Sequential(nn.Conv1d(64*2,                  self.emb_dims,  kernel_size=1, bias=False), nn.BatchNorm1d(self.emb_dims),Mish())
        # self.conv7 = nn.Sequential(nn.Conv1d(self.emb_dims+64*3,    512,            kernel_size=1, bias=False), nn.BatchNorm1d(512),Mish())
        self.conv7 = nn.Sequential(nn.Conv1d(self.emb_dims+64*2,    512,            kernel_size=1, bias=False), nn.BatchNorm1d(512),Mish())
        self.conv8 = nn.Sequential(nn.Conv1d(512,                   256,            kernel_size=1, bias=False), nn.BatchNorm1d(256),Mish())

        self.conv9 = nn.Conv1d(256, 2, kernel_size=1, bias=False)

    def forward(self, x):

        # batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=False)  # (batch_size, 3, num_points) -> (batch_size, 2*3, num_points, k)
        x = self.conv1(x)                               # (batch_size, 2*3, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                               # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # x = get_graph_feature(x2, k=self.k)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # x = self.conv5(x)                               # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        # x3 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # x = torch.cat((x1, x2, x3), dim=1)              # (batch_size, 64*3, num_points)
        x = torch.cat((x1, x2), dim=1)                  # (batch_size, 64*2, num_points)

        x = self.conv6(x)                               # (batch_size, 64*2, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]              # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)                  # (batch_size, 1024, num_points)
        # x = torch.cat((x, x1, x2, x3), dim=1)           # (batch_size, 1024+64*3, num_points)
        x = torch.cat((x, x1, x2), dim=1)               # (batch_size, 1024+64*2, num_points)

        x = self.conv7(x)                               # (batch_size, 1024+64*2, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                               # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.conv9(x)                               # (batch_size, 256, num_points) -> (batch_size, 2, num_points)

        return x