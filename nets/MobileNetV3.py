# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Copyright 2019 aiboy.wei Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Implementation of paper Searching for MobileNetV3, https://arxiv.org/abs/1905.02244
author: aiboy.wei@outlook.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

MobileNetV3_Small_Spec = [
    # Op            k    exp    out    SE     NL        s
    [ "ConvBnAct",  3,   False, 16,    False, "hswish", 2 ],
    [ "bneck",      3,   16,    16,    True,  "relu",   2 ],
    [ "bneck",      3,   72,    24,    False, "relu",   2 ],
    [ "bneck",      3,   88,    24,    False, "relu",   1 ],
    [ "bneck",      5,   96,    40,    True,  "hswish", 2 ],
    [ "bneck",      5,   240,   40,    True,  "hswish", 1 ],
    [ "bneck",      5,   240,   40,    True,  "hswish", 1 ],
    [ "bneck",      5,   120,   48,    True,  "hswish", 1 ],
    [ "bneck",      5,   144,   48,    True,  "hswish", 1 ],
    [ "bneck",      5,   288,   96,    True,  "hswish", 2 ],
    [ "bneck",      5,   576,   96,    True,  "hswish", 1 ],
    [ "bneck",      5,   576,   96,    True,  "hswish", 1 ],
    [ "ConvBnAct",  1,   False, 576,   True,  "hswish", 1 ],
    [ "pool",       7,   False, False, False, "None",   1 ],
    [ "ConvNBnAct", 1,   False, 1280,  False, "hswish", 1 ],
    [ "ConvNBnAct", 1,   False, 1000,  False, "None",   1 ],
]

MobileNetV3_Large_Spec = [
    # Op            k    exp    out    SE     NL        s
    [ "ConvBnAct",  3,   False, 16,    False, "hswish", 2 ],
    [ "bneck",      3,   16,    16,    False, "relu",   1 ],
    [ "bneck",      3,   64,    24,    False, "relu",   2 ],
    [ "bneck",      3,   72,    24,    False, "relu",   1 ],
    [ "bneck",      5,   72,    40,    True,  "relu",   2 ],
    [ "bneck",      5,   120,   40,    True,  "relu",   1 ],
    [ "bneck",      5,   120,   40,    True,  "relu",   1 ],
    [ "bneck",      3,   240,   80,    False, "hswish", 2 ],
    [ "bneck",      3,   200,   80,    False, "hswish", 1 ],
    [ "bneck",      3,   184,   80,    False, "hswish", 1 ],
    [ "bneck",      3,   184,   80,    False, "hswish", 1 ],
    [ "bneck",      3,   480,   112,   True,  "hswish", 1 ],
    [ "bneck",      3,   672,   112,   True,  "hswish", 1 ],
    [ "bneck",      5,   672,   160,   True,  "hswish", 2 ],
    [ "bneck",      5,   960,   160,   True,  "hswish", 1 ],
    [ "bneck",      5,   960,   160,   True,  "hswish", 1 ],
    [ "ConvBnAct",  1,   False, 960,   False, "hswish", 1 ],
    [ "pool",       7,   False, False, False, "None",   1 ],
    [ "ConvNBnAct", 1,   False, 1280,  False, "hswish", 1 ],
    [ "ConvNBnAct", 1,   False, 1000,  False, "None",   1 ],
]

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v

class Linear(tf.keras.layers.Layer):
    def __init__(self, name="Linear"):
        super(Linear, self).__init__(name=name)

    def call(self, input):
        return input

class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, name="HardSigmoid"):
        super(HardSigmoid, self).__init__(name=name)
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6")

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0

class HardSwish(tf.keras.layers.Layer):
    def __init__(self, name="HardSwish"):
        super(HardSwish, self).__init__(name=name)
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6")

    def call(self, input):
        return input * self.relu6(input + 3.0) / 6.0

_available_activation = {
            "relu": tf.keras.layers.ReLU(name="ReLU"),
            "relu6": tf.keras.layers.ReLU(max_value=6, name="ReLU6"),
            "hswish": HardSwish(),
            "hsigmoid": HardSigmoid(),
            "softmax": tf.keras.layers.Softmax(name="Softmax"),
            "None": Linear(),
        }

class SENet(tf.keras.layers.Layer):
    def __init__(self, reduction=4, l2=2e-4, name="SENet"):
        super(SENet, self).__init__(name=name)
        self.reduction = reduction
        self.l2_reg = l2

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name=f'AvgPool{h}x{w}')
        self.fc1 = tf.keras.layers.Dense(units=c//self.reduction, activation="relu", use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name="Squeeze")
        self.fc2 = tf.keras.layers.Dense(units=c, activation=HardSigmoid(), use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name="Excite")
        self.reshape = tf.keras.layers.Reshape((1, 1, c), name=f'Reshape(-1, 1, 1, {c})')

        super().build(input_shape)

    def call(self, input):
        output = self.gap(input)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.reshape(output)
        return input * output

class ConvBnAct(tf.keras.layers.Layer):
    def __init__(self, k, exp, out, SE, NL, s, l2, name="ConvBnAct"):
        super(ConvBnAct, self).__init__(name=name)
        self.conv2d = tf.keras.layers.Conv2D(filters=out, kernel_size=k, strides=s, activation=None, padding="same",
                                             kernel_regularizer=tf.keras.regularizers.l2(l2), name="conv2d")
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.99, name="BatchNormalization")
        self.act = _available_activation[NL]

    def call(self, input):
        output = self.conv2d(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class ConvNBnAct(tf.keras.layers.Layer):
    def __init__(self, k, exp, out, SE, NL, s, l2, name="ConvNBnAct"):
        super(ConvNBnAct, self).__init__(name=name)
        self.act = _available_activation[NL]
        self.fn = tf.keras.layers.Conv2D(filters=out, kernel_size=k, strides=s, activation=self.act, padding="same",
                                         kernel_regularizer=tf.keras.regularizers.l2(l2),name="conv2d")

    def call(self, input):
        output = self.fn(input)
        return output

class Pool(tf.keras.layers.Layer):
    def __init__(self, k, exp, out, SE, NL, s, l2, name="Pool"):
        super(Pool, self).__init__(name=name)
        self.gap = tf.keras.layers.AveragePooling2D(pool_size=(k, k), strides=1, name=f'AvgPool{k}x{k}')

    def call(self, input):
        output = self.gap(input)
        return output

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, k, exp, out, SE, NL, s, l2, name="BottleNeck"):
        super(BottleNeck, self).__init__(name=name)
        self.use_se = SE
        self.stride = s
        self.exp_ch = exp
        self.out_ch = out

        self.expand = ConvBnAct(k=1, exp=exp, out=exp, SE=SE, NL=NL, s=1, l2=l2, name="BottleNeckExpand")
        self.depthwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=k,
            strides=s,
            padding="same",
            use_bias=False,
            depthwise_regularizer=tf.keras.regularizers.l2(l2),
            name=f'Depthwise{k}x{k}',
        )
        self.pointwise = tf.keras.layers.Conv2D(
            filters=out,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            name=f'Pointwise1x1',
        )
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.99, name="BatchNormalization")

        if self.use_se:
            self.se = SENet(name="SEBottleneck",)

        self.act = _available_activation[NL]

    def call(self, input):
        output = self.expand(input)
        output = self.depthwise(output)
        output = self.bn(output)
        if self.use_se:
            output = self.se(output)
        output = self.act(output)
        output = self.pointwise(output)
        output = self.bn(output)

        if self.stride == 1 and self.exp_ch == self.out_ch:
            return input + output
        else:
            return output

_available_mobilenetv3_spec = {
            "small": MobileNetV3_Small_Spec,
            "large": MobileNetV3_Large_Spec,
        }

_available_operation = {
            "ConvBnAct":  ConvBnAct,
            "bneck":      BottleNeck,
            "pool":       Pool,
            "ConvNBnAct": ConvNBnAct,
        }

class MobileNetV3(tf.keras.Model):
    def __init__(self, type="large", classes_numbers=1000, width_multiplier=1.0, divisible_by=8,
                 l2_reg=2e-5, dropout_rate=0.2, name="MobileNetV3"):
        super(MobileNetV3, self).__init__()
        self.spec = _available_mobilenetv3_spec[type]
        self.spec[-1][3] = classes_numbers # bottlenet layer size or class numbers
        self._name = name+"_"+type
        self.backbone = tf.keras.Sequential(name="arch")
        for i, params in enumerate(self.spec):
            Op, k, exp, out, SE, NL, s = params
            inference_op = _available_operation[Op]

            if isinstance(exp, int):
                exp_ch = _make_divisible(exp * width_multiplier, divisible_by)
            else:
                exp_ch = None
            if isinstance(out, int):
                out_ch = _make_divisible(out * width_multiplier, divisible_by)
            else:
                out_ch = None

            name = f'{Op}_{i}'
            self.backbone.add(inference_op(k, exp_ch, out_ch, SE, NL, s, l2_reg, name))
            if (type == "small" and i == 14) or (type == "large" and i == 18):
                self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name=f'{self._name}/Dropout')
                self.backbone.add(self.dropout)

    def call(self, input):
        output = self.backbone(input)
        return output


if __name__ == "__main__":
    import numpy as np

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
    model = MobileNetV3(type="small")

    data = np.zeros((10, 224, 224, 3), dtype=np.float32)
    pre = model(data)
    print(model.variables)
    model.save_weights("./mobilenetv3.h5")
    print(model.summary())