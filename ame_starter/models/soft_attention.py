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
import numpy as np
import keras.backend as K
from keras.layers import Layer
from keras import initializers, activations, regularizers
from keras.initializers import Constant


class SoftAttention(Layer):
    """
    Soft attention layer.

    Following the method from:
    Yang, Z., Yang, D., Dyer, C., He, X., Smola, A. J., & Hovy, E. H. (2016).
    Hierarchical Attention Networks for Document Classification. In HLT-NAACL (pp. 1480-1489).
    """
    def __init__(self,
                 kernel_initializer="glorot_uniform",
                 embedding_initializer="zeros",
                 embedding_activation="tanh",
                 w_regularizer=None,
                 u_regularizer=None,
                 normalised=False,
                 similarity_metric="dot",
                 skip_embedding=False,
                 attention_dropout=0.,
                 use_bias=True,
                 seed=909,
                 num_independent_attention_mechanisms=0,
                 **kwargs):
        self.supports_masking = True
        # self.uses_learning_phase = True

        self.initial_weights_w = initializers.get(kernel_initializer)
        self.initial_weights_u = initializers.get(embedding_initializer)
        self.w_regularizer = regularizers.get(w_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.embedding_activation = activations.get(embedding_activation)
        self.use_bias = use_bias
        self.attention_dropout = attention_dropout
        self.seed = seed
        self.normalised = normalised
        self.w = None
        self.u = None
        self.b = None
        self.weight_magnitude = None
        self.similarity_metric = similarity_metric
        self.skip_embedding = skip_embedding
        self.num_independent_attention_mechanisms = num_independent_attention_mechanisms

        super(SoftAttention, self).__init__(**kwargs)

    def get_config(self):
        config = super(SoftAttention, self).get_config()
        config_new = {
            'use_bias': self.use_bias,
            'seed': self.seed,
            'w_initializer': initializers.serialize(self.initial_weights_w),
            'u_initializer': initializers.serialize(self.initial_weights_u),
            'w_regularizer': regularizers.serialize(self.w_regularizer),
            'u_regularizer': regularizers.serialize(self.u_regularizer),
            'attention_dropout': self.attention_dropout,
            'embedding_activation': activations.serialize(self.embedding_activation),
            'normalised': self.normalised,
            'similarity_metric': self.similarity_metric,
            'skip_embedding': self.skip_embedding,
            'num_independent_attention_mechanisms': self.num_independent_attention_mechanisms
        }
        config.update(config_new)
        return config

    def build(self, input_shape):
        hidden_state_size = input_shape[-1]
        if not self.skip_embedding:
            self.w = self.add_weight((hidden_state_size, hidden_state_size),
                                     initializer=self.initial_weights_w,
                                     name='{}_w'.format(self.name))
            if self.use_bias:
                self.b = self.add_weight((hidden_state_size,),
                                         initializer='zero',
                                         name='{}_b'.format(self.name))

        if self.num_independent_attention_mechanisms == 0:
            context_shape = (hidden_state_size,)
            magnitude_shape = (1,)
        else:
            context_shape = (hidden_state_size, self.num_independent_attention_mechanisms)
            magnitude_shape = (self.num_independent_attention_mechanisms,)

        self.u = self.add_weight(context_shape,
                                 initializer=self.initial_weights_u,
                                 name='{}_u'.format(self.name))

        if self.normalised:
            self.weight_magnitude = self.add_weight(magnitude_shape,
                                                    initializer=Constant(np.sqrt(1. / input_shape[-1])),
                                                    name="{}_g".format(self.name))

        super(SoftAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None  # Any masking is removed at this layer.

    def compute_similarity(self, x, u, gamma=0.1, degree=2, soft_margin_constant=1):
        get_magnitude = lambda v: K.sqrt(K.sum(K.square(v), axis=-1) + K.epsilon())
        subtract_rowwise = lambda l, r: l - K.expand_dims(r, axis=0)
        if self.similarity_metric == "cosine":
            magnitude_x = get_magnitude(x)
            magnitude_u = get_magnitude(u)
            return dot_product(x, u) / (magnitude_x * magnitude_u + K.epsilon())
        elif self.similarity_metric == "euclidean":
            return get_magnitude(subtract_rowwise(x, u))
        elif self.similarity_metric == "rbf":
            return K.exp(-gamma*K.square(get_magnitude(subtract_rowwise(x, u))))
        elif self.similarity_metric == "polynomial":
            return K.pow(dot_product(x, u) + soft_margin_constant, degree)
        else:
            return dot_product(x, u)

    def call(self, input, input_mask=None, training=None):
        if self.skip_embedding:
            u_it = input
        else:
            u_it = K.dot(input, self.w)
            if self.use_bias:
                u_it += self.b

            u_it = self.embedding_activation(u_it)

        if self.normalised:
            # Calculate u as the product of a weight_magnitude and a normalised direction vector.
            # See "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks",
            # https://arxiv.org/pdf/1602.07868.pdf
            # Add K.epsilon to prevent NaN in K.sqrt and division by 0.
            normed_u = self.weight_magnitude * self.u / (K.sqrt(K.sum(K.square(self.u)) + K.epsilon()) + K.epsilon())
            attention_weights = K.exp(self.compute_similarity(u_it, normed_u))
        else:
            attention_weights = K.exp(self.compute_similarity(u_it, self.u))

        if input_mask is not None:
            attention_weights *= K.cast(input_mask, K.floatx())

        if 0. < self.attention_dropout < 1.:
            def dropped_inputs():
                return K.dropout(attention_weights, self.attention_dropout, seed=self.seed)

            attention_weights = K.in_train_phase(dropped_inputs, attention_weights, training=training)

        # Add a small value to avoid division by zero if sum of weights is very small.
        attention_weights /= K.cast(K.sum(attention_weights, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attention_weights = K.expand_dims(attention_weights)
        return attention_weights

    def get_output_shape_for(self, input_shape):
        if self.num_independent_attention_mechanisms == 0:
            return None, input_shape[1]
        else:
            return None, self.num_independent_attention_mechanisms

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


class AfterSoftAttention(Layer):
    """
    We split the soft attention mechanism into two logical layers
    to be able to easily extract the attention factors.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(AfterSoftAttention, self).__init__(**kwargs)

    def get_config(self):
        config = super(AfterSoftAttention, self).get_config()
        return config

    def build(self, input_shape):
        super(AfterSoftAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None  # Any masking is removed at this layer.

    def call(self, input, input_mask=None, training=None):
        input, attention_weights = input
        # Weight initial input by attention.
        weighted_input = input * attention_weights
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        # The temporal dimension is collapsed via attention.
        return tuple(shape for i, shape in enumerate(input_shape[0]) if i != 1)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


def dot_product(x, kernel):
    if K.backend() == 'tensorflow' and len(x.shape) == 3:
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
