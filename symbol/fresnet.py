# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import mxnet as mx
import numpy as np
import symbol_utils
import memonger
import sklearn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

NDEV = 16
def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body


def Act(data, act_type, name):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body

def residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    #print('in unit1')
    if bottle_neck:
        conv1 = Conv(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=stride, pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.contrib.SyncBatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', key=name+'_bn1', ndev=16)
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.contrib.SyncBatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', key=name+'_bn2', ndev=16)
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.contrib.SyncBatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', key=name+'_bn3', ndev=16)

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.contrib.SyncBatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc', key=name+'_sc', ndev=16)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn3 + shortcut, act_type=act_type, name=name + '_relu3')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.contrib.SyncBatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1', key=name+'_bn1', ndev=16)
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.contrib.SyncBatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2', key=name+'_bn2', ndev=16)
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn2 = mx.symbol.broadcast_mul(bn2, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.contrib.SyncBatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc', key=name+'_sc', ndev=16)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn2 + shortcut, act_type=act_type, name=name + '_relu3')

def residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    #print('in unit1')
    if bottle_neck:
        conv1 = Conv(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.contrib.SyncBatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', key=name+'_bn1', ndev=16)
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.contrib.SyncBatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', key=name+'_bn2', ndev=16)
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.contrib.SyncBatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', key=name+'_bn3', ndev=16)

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.contrib.SyncBatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc', key=name+'_sc', ndev=16)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn3 + shortcut, act_type=act_type, name=name + '_relu3')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.contrib.SyncBatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1', key=name+'_bn1', ndev=16)
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.contrib.SyncBatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2', key=name+'_bn2', ndev=16)
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn2 = mx.symbol.broadcast_mul(bn2, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.contrib.SyncBatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc', key=name+'_sc', ndev=16)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn2 + shortcut, act_type=act_type, name=name + '_relu3')

def residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    #print('in unit2')
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.contrib.SyncBatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', key=name+'_bn1', ndev=16)
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv1 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.contrib.SyncBatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', key=name+'_bn2', ndev=16)
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        conv2 = Conv(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.contrib.SyncBatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', key=name+'_bn3', ndev=16)
        act3 = Act(data=bn3, act_type=act_type, name=name + '_relu3')
        conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=conv3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          conv3 = mx.symbol.broadcast_mul(conv3, body)
        if dim_match:
            shortcut = data
        else:
            shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.contrib.SyncBatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1', key=name+'_bn1', ndev=16)
        act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
        conv1 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.contrib.SyncBatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2', key=name+'_bn2', ndev=16)
        act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
        conv2 = Conv(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=conv2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          conv2 = mx.symbol.broadcast_mul(conv2, body)
        if dim_match:
            shortcut = data
        else:
            shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    #print('in unit3')
    if bottle_neck:
        bn1 = mx.sym.contrib.SyncBatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', key=name+'_bn1', ndev=16)
        conv1 = Conv(data=bn1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.contrib.SyncBatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', key=name+'_bn2', ndev=16)
        act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.contrib.SyncBatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', key=name+'_bn3', ndev=16)
        act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn4 = mx.sym.contrib.SyncBatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4', key=name+'_bn4', ndev=16)

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn4, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn4 = mx.symbol.broadcast_mul(bn4, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.contrib.SyncBatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc', key=name+'_sc', ndev=16)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return bn4 + shortcut
    else:
        bn1 = mx.sym.contrib.SyncBatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', key=name+'_bn1', ndev=16)
        conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.contrib.SyncBatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', key=name+'_bn2', ndev=16)
        act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.contrib.SyncBatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', key=name+'_bn3', ndev=16)
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.contrib.SyncBatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc', key=name+'_sc', ndev=16)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return bn3 + shortcut

def residual_unit_v3_x(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    
    """Return ResNeXt Unit symbol for building ResNeXt
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    assert(bottle_neck)
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    num_group = 32
    #print('in unit3')
    bn1 = mx.sym.contrib.SyncBatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', key=name+'_bn1', ndev=16)
    conv1 = Conv(data=bn1, num_group=num_group, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.contrib.SyncBatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', key=name+'_bn2', ndev=16)
    act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
    conv2 = Conv(data=act1, num_group=num_group, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.contrib.SyncBatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', key=name+'_bn3', ndev=16)
    act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
    conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    bn4 = mx.sym.contrib.SyncBatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4', key=name+'_bn4', ndev=16)

    if use_se:
      #se begin
      body = mx.sym.Pooling(data=bn4, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
      body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                name=name+"_se_conv1", workspace=workspace)
      body = Act(data=body, act_type=act_type, name=name+'_se_relu1')
      body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                name=name+"_se_conv2", workspace=workspace)
      body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
      bn4 = mx.symbol.broadcast_mul(bn4, body)
      #se end

    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_conv1sc')
        shortcut = mx.sym.contrib.SyncBatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc', key=name+'_sc', ndev=16)
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return bn4 + shortcut


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
  uv = kwargs.get('version_unit', 3)
  version_input = kwargs.get('version_input', 1)
  if uv==1:
    if version_input==0:
      return residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    else:
      return residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  elif uv==2:
    return residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  elif uv==4:
    return residual_unit_v4(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  else:
    return residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
#  def loc(data, filter_loc=[24,48,96,64], num_group=1, workspace=256):
#      body = mx.sym.Pooling(data=data, kernel=(2,2), stride=(2,2), pool_type='avg', name='loc_pool0')
#      body = mx.sym.Convolution(data=body, num_filter=filter_loc[0], kernel=(3,3), stride=(1,1), name='loc_conv1', workspace=workspace)
#      body = mx.sym.Activation(data=body, act_type='relu', name='loc_relu1')
#      body = mx.sym.Pooling(data=body, kernel=(2,2), stride=(2,2), pool_type='max', name='loc_pool1')
#      body = mx.sym.Convolution(data=body, num_filter=filter_loc[1], kernel=(3,3), stride=(1,1), name='loc_conv2', workspace=workspace)
#      body = mx.sym.Activation(data=body, act_type='relu', name='loc_relu2')
#      body = mx.sym.Pooling(data=body, kernel=(2,2), stride=(2,2), pool_type='max', name='loc_pool2')
#      body = mx.sym.Convolution(data=body, num_filter=filter_loc[2], kernel=(3,3), stride=(1,1), name='loc_conv3', workspace=workspace)
#      body = mx.sym.Activation(data=body, act_type='relu', name='loc_relu3')
#      body = mx.sym.Pooling(data=body, kernel=(2,2), stride=(2,2), pool_type='max', name='loc_pool3')
#      body = mx.sym.Flatten(data=body)
#      body = mx.sym.FullyConnected(data=body, num_hidden=filter_loc[3], attr={'lr_mult':0.001}, name='loc_fc4')
#      body = mx.sym.Activation(data=body, act_type='relu', name='loc_relu4')
#      body = mx.sym.FullyConnected(data=body, num_hidden=6, attr={'lr_mult':0.001}, name='stn_loc')
#      return body

def loc(data, filter_loc, bn_mom, num_group=1, workspace=256, use_global_stats=False, stn_lr_mult=0.0):
    print('For debug, stn_lr_mutl:%f , filter_loc : %s, num_group: %d, bn_mom:%.2f, use_global_stats:%d '% \
            (stn_lr_mult, str(filter_loc), num_group, bn_mom, use_global_stats))
    # first 
    loc = mx.symbol.Pooling(data=data, kernel=(2,2), stride=(2,2), pool_type='avg', name='loc_pool0')
    loc = mx.sym.Convolution(data=loc, num_filter=filter_loc[0], kernel=(3,3), stride=(1,1), no_bias=True, num_group=1, \
                             name='loc_conv1', workspace=workspace, attr={'lr_mult':str(stn_lr_mult)})   #input is rgb, so num_group=1 here
    loc = mx.contrib.sym.SyncBatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=use_global_stats, name='loc_bn1', \
                                       attr={'lr_mult':str(stn_lr_mult)}, ndev=NDEV, key='loc_bn1')
    loc = mx.sym.Activation(data=loc, act_type='relu', name='loc_relu1')

    #second
    loc = mx.symbol.Pooling(data=loc, kernel=(2,2), stride=(2,2), pool_type='max', name='loc_pool1')
    loc = mx.sym.Convolution(data=loc, num_filter=filter_loc[1], kernel=(3,3), stride=(1,1), no_bias=True, num_group=num_group, \
                             name='loc_conv2', workspace=workspace, attr={'lr_mult':str(stn_lr_mult)})  
    loc = mx.contrib.sym.SyncBatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=use_global_stats, name='loc_bn2', \
                                       attr={'lr_mult':str(stn_lr_mult)}, ndev=NDEV, key='loc_bn2')
    loc = mx.sym.Activation(data=loc, act_type='relu', name='loc_relu2')
    
    #third
    loc = mx.symbol.Pooling(data=loc, kernel=(2,2), stride=(2,2), pool_type='max', name='loc_pool2')
    loc = mx.sym.Convolution(data=loc, num_filter=filter_loc[2], kernel=(3,3), stride=(1,1), no_bias=True, num_group=num_group, \
                             name='loc_conv3', workspace=workspace, attr={'lr_mult':str(stn_lr_mult)})  
    loc = mx.contrib.sym.SyncBatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=use_global_stats, name='loc_bn3', \
                                       attr={'lr_mult':str(stn_lr_mult)}, ndev=NDEV, key='loc_bn3')
    loc = mx.sym.Activation(data=loc, act_type='relu', name='loc_relu3')

    #forth
    loc = mx.symbol.Pooling(data=loc, kernel=(2,2), stride=(2,2), pool_type='max', name='loc_pool3')
    loc = mx.symbol.Flatten(data=loc)
    loc = mx.symbol.FullyConnected(data=loc, num_hidden=filter_loc[3], name='loc_fc4', attr={'lr_mult':str(stn_lr_mult)})
    loc = mx.contrib.sym.SyncBatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=use_global_stats, name='loc_bn4', \
                                       attr={'lr_mult':str(stn_lr_mult)}, ndev=NDEV, key='loc_bn4')
    loc = mx.sym.Activation(data=loc, act_type='relu', name='loc_relu4')

    #init
    _w = mx.symbol.Variable('locw', shape=(6, filter_loc[3]), init=mx.init.Zero(), attr={'lr_mult':str(stn_lr_mult)})
    _b = mx.symbol.Variable('locb', shape=(6), init=mx.init.Constant([1., 0., 0., 0., 1., 0.]), attr={'lr_mult': str(stn_lr_mult)})
    loc = mx.sym.FullyConnected(data=loc, weight=_w, bias=_b, num_hidden=6, attr={'lr_mult':str(stn_lr_mult)}, name='stn_loc')

    return loc



#  def resnet(units, num_stages, filter_list, num_classes, bottle_neck):
def resnet(units, num_stages, filter_list, num_classes, bottle_neck, filter_loc, target_shape, use_global_stats=False, bn_mom=0.9, workspace=256, stn_lr_mult=0.0):
    #  bn_mom = config.bn_mom
    #  workspace = config.workspace
    kwargs = {'version_se' : config.net_se,
        'version_input': config.net_input,
        'version_output': config.net_output,
        'version_unit': config.net_unit,
        'version_act': config.net_act,
        'bn_mom': bn_mom,
        'workspace': workspace,
        'memonger': config.memonger,
        }
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    version_se = kwargs.get('version_se', 1)
    version_input = kwargs.get('version_input', 1)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 3)
    act_type = kwargs.get('version_act', 'prelu')
    memonger = kwargs.get('memonger', False)
    print('version_se:{}, version_input:{}, version_output:{}, act_type:{}, memonger:{}'.format(
            version_se, version_input, version_output, version_unit, act_type, memonger))
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    if version_input==0:
      #data = mx.sym.contrib.SyncBatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data', key='bn_data', ndev=16)
      data = mx.sym.identity(data=data, name='id')
      data = data-127.5
      data = data*0.0078125
      body = Conv(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.contrib.SyncBatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='sybn_01', key='sybn_01', ndev=16)
      body = Act(data=body, act_type=act_type, name='relu0')
      #body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    elif version_input==2:
      data = mx.sym.contrib.SyncBatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data', key='bn_data', ndev=16)
      body = Conv(data=data, num_filter=filter_list[0], kernel=(3,3), stride=(1,1), pad=(1,1),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.contrib.SyncBatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='sybn_02', key='sybn_02', ndev=16)
      body = Act(data=body, act_type=act_type, name='relu0')
    elif version_input==3:
      print("STN")
      #  data = data-127.5
      #  data = data*0.0078125
      #  body = mx.sym.SpatialTransformer(data=data, loc=loc(data), target_shape=(128,112), \
      #                                   transform_type="affine", sampler_type="bilinear", name="stn")
      body = mx.sym.identity(data=data, name='id')
      body = mx.sym.SpatialTransformer(data=body, loc=loc(data, filter_loc, bn_mom, workspace=workspace, \
                                       use_global_stats=use_global_stats, stn_lr_mult=stn_lr_mult),
                                       target_shape=target_shape,
                                       transform_type="affine", sampler_type="bilinear", name="stn")

      #  body1 = loc(data, filter_loc, bn_mom, workspace=workspace, use_global_stats=use_global_stats, stn_lr_mult=stn_lr_mult)
      #  body = mx.sym.SpatialTransformer(data=body, loc=body1, target_shape=target_shape,
      #                                   transform_type="affine", sampler_type="bilinear", name="stn")

      #  tmp_b = mx.symbol.Variable('tmp_b', shape=(1,6), init=mx.init.Constant([[1., 0., 0., 0., 1., 0.]]))
      #  tmp_b_brd = mx.symbol.broadcast(shape(config._per_batch_size, 6))
      #  loss_tmp = mx.symbol.sum(mx.symbol.sum(mx.symbol.square(tmp_b_brd - body1), axis=1), axis=0)
      #  stn_loss = mx.sym.MakeLoss(loss_tmp)

      body = Conv(data=body, num_filter=filter_list[0], kernel=(3,3), stride=(1,1), pad=(1,1),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.contrib.SyncBatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0', key='bn0', ndev=NDEV)
      body = Act(data=body, act_type=act_type, name='relu0')
    else: # 1
      data = mx.sym.identity(data=data, name='id')
      data = data-127.5
      data = data*0.0078125
      body = data
      body = Conv(data=body, num_filter=filter_list[0], kernel=(3,3), stride=(1,1), pad=(1, 1),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.contrib.SyncBatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='sybn_04', key='sybn_04', ndev=16)
      body = Act(data=body, act_type=act_type, name='relu0')

    for i in range(num_stages):
      #if version_input==0:
      #  body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
      #                       name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      #else:
      #  body = residual_unit(body, filter_list[i+1], (2, 2), False,
      #    name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      body = residual_unit(body, filter_list[i+1], (2, 2), False,
        name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      for j in range(units[i]-1):
        body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i+1, j+2),
          bottle_neck=bottle_neck, **kwargs)

    if bottle_neck:
      body = Conv(data=body, num_filter=512, kernel=(1,1), stride=(1,1), pad=(0,0),
                                no_bias=True, name="convd", workspace=workspace)
      body = mx.sym.contrib.SyncBatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bnd', key='bnd', ndev=16)
      body = Act(data=body, act_type=act_type, name='relud')

    fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    return fc1

def get_symbol():
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    num_classes = config.emb_size
    num_layers = config.num_layers
    
    # add by 1205
    bn_mom = config.bn_mom
    workspace = config.workspace
    use_global_stats = False
    target_shape = (config.stn_h, config.stn_w)
    filter_loc = [24, 28, 96, 64]

    if num_layers >= 500:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 98:
        units = [3, 4, 38, 3]
    elif num_layers == 99:
        units = [3, 8, 35, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 134:
        units = [3, 10, 50, 3]
    elif num_layers == 136:
        units = [3, 13, 48, 3]
    elif num_layers == 140:
        units = [3, 15, 48, 3]
    elif num_layers == 124:
        units = [3, 13, 40, 5]
    elif num_layers == 160:
        units = [3, 24, 49, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
    
    print('for debug, get symbole resnet before')
    net = resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  bottle_neck = bottle_neck,
                  filter_loc  = filter_loc,
                  target_shape = target_shape,
                  use_global_stats = use_global_stats,
                  bn_mom = bn_mom,
                  workspace = workspace,
                  stn_lr_mult = config.stn_lr_mult)
    print('for debug, get symbol resnet done')
    if config.memonger:
      dshape = (config.per_batch_size, config.image_shape[2], config.image_shape[0], config.image_shape[1])
      net_mem_planned = memonger.search_plan(net, data=dshape)
      old_cost = memonger.get_cost(net, data=dshape)
      new_cost = memonger.get_cost(net_mem_planned, data=dshape)

      print('Old feature map cost=%d MB' % old_cost)
      print('New feature map cost=%d MB' % new_cost)
      net = net_mem_planned
    return net


