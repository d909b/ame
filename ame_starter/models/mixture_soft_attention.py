"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from keras.initializers import Constant
from keras import initializers, activations, constraints, regularizers
from ame_starter.models.soft_attention import SoftAttention


class MixtureSoftAttention(Layer):
    """
    Mixture Stream attention layer for attentive neural networks.
    Using a mixture of softmaxes as explained in (for RNNs):
    Breaking the Softmax Bottleneck: A High-Rank RNN Language Model
    by Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen.
    """
    def __init__(self,
                 num_softmaxes=1,
                 activation='tanh',
                 use_bias=True,
                 w_initializer='glorot_uniform',
                 u_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 w_regularizer=None,
                 u_regularizer=None,
                 bias_regularizer=None,
                 w_constraint=None,
                 u_constraint=None,
                 bias_constraint=None,
                 attention_dropout=0.,
                 seed=909,
                 stream_size=128,
                 normalised=False,
                 similarity_metric="dot",
                 skip_embedding=False,
                 num_independent_attention_mechanisms=0,
                 **kwargs):
        self.supports_masking = True

        self.w_initializer = initializers.get(w_initializer)
        self.u_initializer = initializers.get(u_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.w_constraint = constraints.get(w_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.w_regularizer = regularizers.get(w_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.attention_dropout = max(min(1., float(attention_dropout)), 0)
        self.seed = seed
        self.P = None
        self.M = None
        self.weight_magnitude = None
        self.stream_size = stream_size
        self.num_softmaxes = num_softmaxes
        self.normalised = normalised
        self.similarity_metric = similarity_metric
        self.skip_embedding = skip_embedding
        self.num_independent_attention_mechanisms = num_independent_attention_mechanisms
        self.child_layers = []

        super(MixtureSoftAttention, self).__init__(**kwargs)

    def get_config(self):
        config = super(MixtureSoftAttention, self).get_config()
        new_config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'w_initializer': initializers.serialize(self.w_initializer),
            'u_initializer': initializers.serialize(self.u_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'w_regularizer': regularizers.serialize(self.w_regularizer),
            'u_regularizer': regularizers.serialize(self.u_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'w_constraint': constraints.serialize(self.w_constraint),
            'u_constraint': constraints.serialize(self.u_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'attention_dropout': self.attention_dropout,
            'stream_size': self.stream_size,
            'num_softmaxes': self.num_softmaxes,
            'normalised': self.normalised,
            'similarity_metric': self.similarity_metric,
            'skip_embedding': self.skip_embedding,
            'num_independent_attention_mechanisms': self.num_independent_attention_mechanisms
        }
        config.update(new_config)
        return config

    def build(self, input_shape):
        if self.num_softmaxes > 1:
            hidden_state_size = input_shape[-1]

            self.P = self.add_weight((self.num_softmaxes, hidden_state_size, hidden_state_size),
                                     initializer=self.w_initializer,
                                     name='{}_w'.format(self.name),
                                     regularizer=self.w_regularizer,
                                     constraint=self.w_constraint)

            self.M = self.add_weight((hidden_state_size, self.num_softmaxes),
                                     initializer=self.w_initializer,
                                     name='{}_w'.format(self.name),
                                     regularizer=self.w_regularizer,
                                     constraint=self.w_constraint)

            if self.normalised:
                self.weight_magnitude = self.add_weight((self.num_softmaxes,),
                                                        initializer=Constant(np.sqrt(1. / input_shape[-1])),
                                                        name="{}_g".format(self.name))

        super(MixtureSoftAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None  # Any masking is removed at this layer.

    def get_stream_attention_layer(self):
        layer = SoftAttention(
            kernel_initializer=self.w_initializer,
            embedding_initializer=self.u_initializer,
            embedding_activation=self.activation,
            w_regularizer=self.w_regularizer,
            u_regularizer=self.u_regularizer,
            normalised=self.normalised,
            similarity_metric=self.similarity_metric,
            skip_embedding=self.skip_embedding,
            attention_dropout=self.attention_dropout,
            use_bias=self.use_bias,
            seed=self.seed,
            num_independent_attention_mechanisms=self.num_independent_attention_mechanisms
        )
        self.child_layers.append(layer)
        return layer

    @property
    def trainable_weights(self):
        own_weights = super(MixtureSoftAttention, self).trainable_weights
        child_weights = []
        for child in self.child_layers:
            child_weights += child.trainable_weights
        return own_weights + child_weights

    @property
    def non_trainable_weights(self):
        own_weights = super(MixtureSoftAttention, self).non_trainable_weights
        child_weights = []
        for child in self.child_layers:
            child_weights += child.non_trainable_weights
        return own_weights + child_weights

    def call(self, input, input_mask=None, training=None):
        if self.num_softmaxes == 1:
            return self.get_stream_attention_layer()(input)
        else:
            p_i = tf.unstack(self.P, axis=0)
            h_k = []
            for p in p_i:
                h = K.dot(input, p)
                h = self.activation(h)
                h_k.append(h)

            h_k = tf.stack(h_k, axis=-2)  # h_k = (num_softmaxes, len(h)[-1])

            if self.normalised:
                normed_m = self.weight_magnitude * self.M / \
                           (K.sqrt(K.sum(K.square(self.M)) + K.epsilon()) + K.epsilon())
            else:
                normed_m = self.M

            pi_k = K.softmax(K.dot(input, normed_m))

            h_k_split = tf.unstack(h_k, axis=-2)

            all_attention_factors = []
            for h in h_k_split:
                a_i = self.get_stream_attention_layer()(h)
                a_i = K.squeeze(a_i, axis=-1)
                all_attention_factors.append(a_i)
            attention_factors = tf.stack(all_attention_factors, axis=-1)
            attention_factors = K.batch_dot(attention_factors, K.expand_dims(pi_k, axis=-1))
            return attention_factors

    def get_output_shape_for(self, input_shape):
        if self.num_independent_attention_mechanisms == 0:
            return None, input_shape[1]
        else:
            return None, self.num_independent_attention_mechanisms

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)
