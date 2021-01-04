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

from utils.model_utils import get_graph_feature, getConvLayer, getActivation

class Model(nn.Module):

    def __init__(self, d = 3, F = 10, out_channels = 2, activation_type='mish', activation_args={}):

        super(Model, self).__init__()

        self.F = F
        self.d = d
        self.out_channels = out_channels

        self.conv1 = getConvLayer(self.d*2, 64, getActivation(activation_type, args=activation_args), d=2)
        self.conv2 = getConvLayer(64*2, 64, getActivation(activation_type, args=activation_args), d=2)
        self.conv3 = getConvLayer(64*2, 128, getActivation(activation_type, args=activation_args), d=2)

        self.conv4 = getConvLayer(128, 128, getActivation(activation_type, args=activation_args), d=1)

        self.conv5 = getConvLayer(256, 128, getActivation(activation_type, args=activation_args), d=1)
        self.conv6 = getConvLayer(128, 64, getActivation(activation_type, args=activation_args), d=1)
        self.conv7 = getConvLayer(128, 64, getActivation(activation_type, args=activation_args), d=1)
        self.conv8 = getConvLayer(128, 64, getActivation(activation_type, args=activation_args), d=1)

        self.conv9 = nn.Conv1d(64, self.out_channels, kernel_size=1, bias=False)

    def forward(self, x):

        x = get_graph_feature(x, k=self.F, dim9=False)  # (batch_size, d, num_points) -> (batch_size, 2*d, num_points, k)
        x = self.conv1(x)                               # (batch_size, 2*d, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.F)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                               # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.F)             # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                               # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv4(x3)                              # (batch_size, 128, num_points) -> (batch_size, 128, num_points)

        x = torch.cat((x, x3), 1)                       # (batch_size, 128, num_points) -> (batch_size, 256, num_points)
        x = self.conv5(x)                               # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv6(x)                               # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x = torch.cat((x, x2), 1)                       # (batch_size, 64, num_points) -> (batch_size, 128, num_points)
        x = self.conv7(x)                               # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x = torch.cat((x, x1), 1)                       # (batch_size, 64, num_points) -> (batch_size, 128, num_points)
        x = self.conv8(x)                               # (batch_size, 128, num_points) -> (batch_size, 64, num_points)

        x = self.conv9(x)                               # (batch_size, 64, num_points) -> (batch_size, out_channels, num_points)

        return x