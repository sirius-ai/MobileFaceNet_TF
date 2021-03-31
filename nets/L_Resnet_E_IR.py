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
implement ReNet50 for face recognition.
Original author Andy.Wei

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import tensorflow as tf
import argparse
import sys

variable_scope = 'ResNet'

def residual_unit_v3(data, out_filter, stride, dim_match, trainable, name):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    out_filter : int
        Number of output channels
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    trainable: Boolean
        trainning or testing, True is trainning, otherwise is testing.
    name : str
        Base name of the operators
    """
    shape = [3, 3]
    in_filter= data.get_shape().as_list()[-1]
    shape.append(int(in_filter))
    shape.append((out_filter))

    # print(name)

    bn1 = batch_normalization(data, variance_epsilon=2e-5, trainable=trainable, name=name + '_bn1')
    bn1_pad = tf.pad(bn1, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    conv1 = convolution(bn1_pad, group=1, shape=shape, strides=[1, 1], padding='VALID', trainable=trainable, name=name + '_conv1')
    bn2 = batch_normalization(conv1, variance_epsilon=2e-5, trainable=trainable, name=name + '_bn2')
    relu1 = prelu(bn2, trainable=trainable, name=name + '_relu1')
    relu1_pad = tf.pad(relu1, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
    shape[-2] = relu1_pad.get_shape().as_list()[-1]
    conv2 = convolution(relu1_pad, group=1, shape=shape, strides=stride, padding='VALID', trainable=trainable, name=name + '_conv2')
    bn3 = batch_normalization(conv2, variance_epsilon=2e-5, trainable=trainable, name=name + '_bn3')

    if dim_match:
        shortcut = data
    else:
        shape = [1, 1]
        in_filter = data.get_shape().as_list()[-1]
        shape.append(int(in_filter))
        shape.append((out_filter))

        conv1sc = convolution(data, group=1, shape=shape, strides=stride, padding='VALID', trainable=trainable, name=name + '_conv1sc')
        shortcut = batch_normalization(conv1sc, variance_epsilon=2e-5, trainable=trainable, name=name + '_sc')

    return bn3 + shortcut


def residual_unit(data, out_filter, stride, dim_match, trainable, name, **kwargs):
    return residual_unit_v3(data, out_filter, stride, dim_match, trainable, name=name, **kwargs)

def prelu(input, trainable, name):
    gamma = tf.get_variable(initializer=tf.constant(0.25,dtype=tf.float32,shape=[input.get_shape()[-1]]), trainable=trainable, name=name + "_gamma")
    return tf.maximum(0.0, input) + gamma * tf.minimum(0.0, input)

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op

def batch_normalization(input, trainable, name, **kwargs):
    input_shape = input.get_shape()
    shape = input_shape.as_list()[-1::]
    axis = list(range(len(input_shape) - 1))
    moving_mean = tf.get_variable(shape=shape, initializer=tf.zeros_initializer, trainable=trainable, name=name + "_mean")
    moving_variance = tf.get_variable(shape=shape, initializer=tf.ones_initializer, trainable=trainable, name=name + "_var")
    offset = tf.get_variable(shape=shape, initializer=tf.zeros_initializer, trainable=trainable, name=name + "_bias")
    scale = tf.get_variable(shape=shape, initializer=tf.ones_initializer, trainable=trainable, name=name + "_scale") if name != 'fc1' else None

    mean, variance = tf.nn.moments(input, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    is_training = tf.convert_to_tensor(trainable, dtype='bool', name='is_training')
    mean, variance = control_flow_ops.cond(is_training,
        lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name=name, **kwargs)

def convolution(input, group, shape, trainable, name, **kwargs):
    w = tf.get_variable(initializer=tf.truncated_normal(shape, stddev=0.1), trainable=trainable, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, pruning.apply_mask(w, name + "_weight"), **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, pruning.apply_mask(weight, name + "_weight_groups"), **kwargs) for
                     (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if name.endswith('_sc'):
        b = tf.get_variable(initializer=tf.truncated_normal(input.get_shape().as_list()[-1::], stddev=0.1), trainable=trainable, name=name + "_bias")
        layer = layer + b
    return layer

def resnet(inputs, w_init, units, num_stages, filter_list, trainable, reuse=False):
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
    """
    #version_se = kwargs.get('version_se', 1)
    #version_input = kwargs.get('version_input', 1)
    #assert version_input >= 0
    #version_output = kwargs.get('version_output', 'E')
    #version_unit = kwargs.get('version_unit', 3)
    #print(version_se, version_input, version_output, version_unit)
    num_unit = len(units)
    assert (num_unit == num_stages)
    inputs = inputs - 127.5
    inputs = inputs * 0.0078125

    with tf.variable_scope(variable_scope, reuse=reuse):
        net = tf.pad(inputs, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        net = convolution(net, group=1, strides=[1, 1], shape=[3, 3, 3, 64], padding='VALID', trainable=trainable, name='conv0')
        net = batch_normalization(net, variance_epsilon=2e-5, trainable=trainable, name='bn0')
        net = prelu(net, trainable=trainable, name='relu0')

        body = net
        for i in range(num_stages):
            body = residual_unit(body, filter_list[i + 1], (2, 2), False, trainable=trainable, name='stage%d_unit%d' % (i + 1, 1))
            for j in range(units[i] - 1):
                body = residual_unit(body, filter_list[i + 1], (1, 1), True, trainable=trainable, name='stage%d_unit%d' % (i + 1, j + 2))

        bn1 = batch_normalization(body, variance_epsilon=2e-5, trainable=trainable, name='bn1')
        bn1_shape = bn1.get_shape().as_list()
        bn1 = tf.reshape(bn1, shape=[-1, bn1_shape[1] * bn1_shape[2] * bn1_shape[3]], name='E_Reshapelayer')
        pre_fc1 = tf.layers.dense(bn1, units=512, kernel_initializer=w_init, use_bias=True)
        fc1 = batch_normalization(pre_fc1, variance_epsilon=2e-5, trainable=trainable, name='fc1')

    return fc1, pre_fc1


def get_resnet(inputs, w_init, num_layers, trainable, reuse=False):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    if num_layers >= 101:
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
    elif num_layers == 100:
        units = [3, 13, 30, 3]
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

    return resnet(inputs=inputs,
                  w_init=w_init,
                  units=units,
                  num_stages=num_stages,
                  filter_list=filter_list,
                  trainable=trainable,
                  reuse=reuse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, help='Load a pretrained model before training starts.')
    parser.add_argument('--ckpt_path', type=str, help='the checkpoint path to save model.')
    args = parser.parse_args(sys.argv[1:])

    with tf.Graph().as_default():
        with tf.Session() as sess:
            input = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input')
            trainable_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None, name='trainable')
            w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
            prelogits = get_resnet(input, w_init=w_init_method, num_layers=50, trainable=trainable_placeholder)

            embeddings = tf.identity(prelogits, name='embeddings')

            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
            print(args.pretrained_model)
            ckpt = tf.train.get_checkpoint_state(args.pretrained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)
            saver.save(sess, args.ckpt_path, global_step=0)

    print('test finish!')
