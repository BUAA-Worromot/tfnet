import tensorflow.python.keras.layers as layers
import math
import collections
from itertools import repeat
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn


def _ntuple(n):
    def parse(x):
        # if isinstance(x, container_abcs.Iterable):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class Conv2dX100():
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 transposed=False,
                 output_padding=0):
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.transposed = transposed
        self.output_padding = _pair(output_padding)
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = tf.Variable(
                np.random.random((in_channels, out_channels // groups,
                             *self.kernel_size)))
        else:
            self.weight = tf.Variable(
                np.random.random((out_channels, in_channels // groups,
                             *self.kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
